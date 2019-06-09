import random
import pickle
import os
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from model import BasicClassifier
from arguments import get_args
from utils import save_ckpt, GOATLogger, compute_score

class yn_dataset(Dataset):

    def __init__(self, root, train, seqlen=14):
        """
        root (str): path to data directory
        train (bool): training or validation
        seqlen (int): maximum words in a question
        """
        if train:
            prefix = 'train'
            datapath = os.path.join(root, 'data_non_yesno', 'non_yn_preds.pkl')
            datapath1 = os.path.join(root, 'data_yesno', prefix + '_qa.pkl')
            j_path = os.path.join(root, 'data_non_yesno', 'non_yn_val_joint_feats.pkl')
            j_path1 = os.path.join(root, 'data_yesno', prefix + '_joint_feats.pkl')
        else:
            prefix = 'val'
            datapath = os.path.join(root, 'data_yesno', prefix + '_qa.pkl')
            j_path = os.path.join(root, 'data_yesno', prefix + '_joint_feats.pkl')
        print("Loading preprocessed files... ({})".format(prefix))
        #qas = pickle.load(open(os.path.join(root, prefix + '_qa.pkl'), 'rb'))
        idx2ans, ans2idx = pickle.load(open(os.path.join(root, 'data_yesno', 'dict_ans.pkl'), 'rb'))

        #joint_embed = pickle.load(open(os.path.join(root, prefix + '_joint_feats.pkl'), 'rb'))
        
        if train:
            qass = [pickle.load(open(datapath, 'rb')), pickle.load(open(datapath1, 'rb'))]
            joint_embed = [pickle.load(open(j_path, 'rb')), pickle.load(open(j_path1, 'rb'))]
        else:
            qass = [pickle.load(open(datapath, 'rb'))]
            joint_embed = [pickle.load(open(j_path, 'rb'))]

        print("Setting up everything... ({})".format(prefix))
        self.vqas = []
        for idxx, qas in enumerate(qass): 
            for qa in tqdm(qas):
                ans = np.zeros(len(idx2ans), dtype=np.float32)
                for a, s in qa['answer']:
                    ans[ans2idx[a]] = s

                self.vqas.append({
                    'j': joint_embed[idxx][qa['question_id']],
                    'a': ans
                })

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        return torch.from_numpy(self.vqas[idx]['j']), \
               torch.Tensor(self.vqas[idx]['a'])

    @staticmethod
    def get_n_classes(fpath=os.path.join('data', 'data_yesno', 'dict_ans.pkl')):
        idx2ans, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2ans)

    #@staticmethod
    #def get_vocab_size(fpath=os.path.join(root, 'data_non_yesno', 'dict_q.pkl')):
    #    idx2word, _ = pickle.load(open(fpath, 'rb'))
    #    return len(idx2word)


def prepare_data(args):

    train_loader = torch.utils.data.DataLoader(yn_dataset(root=args.data_root, train=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers)

    val_loader = torch.utils.data.DataLoader(yn_dataset(root=args.data_root, train=False),
                                             batch_size=args.vbatch_size,
                                             shuffle=False,
                                             num_workers=args.n_workers)

    #vocab_size = yn_dataset.get_vocab_size()
    num_classes = yn_dataset.get_n_classes()
    return train_loader, val_loader, num_classes


class Model(nn.Module):

    def __init__(self, word_embed_dim, hidden_size, num_answers):

        super(Model, self).__init__()
        self.classifier = BasicClassifier(hidden_size,
                                          word_embed_dim,
                                          num_answers)

    def forward(self, joint_embed):

        outputs = self.classifier(joint_embed)

        return outputs


def evaluate(val_loader, model, epoch, device, logger):
    model.eval()

    batches = len(val_loader)
    for step, (j, a) in enumerate(tqdm(val_loader, ascii=True)):
        j = j.to(device)
        a = a.to(device)

        logits = model(j)

        loss = F.binary_cross_entropy_with_logits(logits, a) * a.size(1)
        score = compute_score(logits, a)

        logger.batch_info_eval(epoch, step, batches, loss.item(), score)

    score = logger.batch_info_eval(epoch, -1, batches)
    return score


def train(train_loader, model, optim, epoch, device, logger):
    model.train()

    batches = len(train_loader)
    start = time.time()
    for step, (j, a) in enumerate(tqdm(train_loader, ascii=True)):
        data_time = time.time() - start

        j = j.to(device)
        a = a.to(device)

        logits = model(j)
        loss = F.binary_cross_entropy_with_logits(logits, a) * a.size(1)

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optim.step()

        batch_time = time.time() - start
        score = compute_score(logits, a)
        logger.batch_info(epoch, step, batches, data_time, loss.item(), score, batch_time)
        start = time.time()


def main():
    parser = get_args()
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

    logger = GOATLogger(args.mode, args.save, args.log_freq)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cpu:
        device = torch.device('cpu')
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable.")

        args.devices = torch.cuda.device_count()
        args.batch_size *= args.devices
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda')
        torch.cuda.manual_seed(args.seed)

    train_loader, val_loader, num_answers = prepare_data(args)

    model = Model(args.word_embed_dim, args.hidden_size, num_answers)
    model = nn.DataParallel(model).to(device)
    logger.loginfo("Parameters: {:.3f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    optim = torch.optim.Adamax(model.parameters(), lr=5e-4)

    last_epoch = 0
    bscore = 0.0

    if args.resume:
        logger.loginfo("Initialized from ckpt: " + args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        last_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optim.load_state_dict(ckpt['optim_state_dict'])

    if args.mode == 'eval':
        _ = evaluate(val_loader, model, last_epoch, device, logger)
        return

    # Train
    for epoch in range(last_epoch, args.epoch):
        train(train_loader, model, optim, epoch, device, logger)
        score = evaluate(val_loader, model, epoch, device, logger)
        bscore = save_ckpt(score, bscore, epoch, model, optim, args.save, logger)

    logger.loginfo("Done")

if __name__ == "__main__":
    main()

