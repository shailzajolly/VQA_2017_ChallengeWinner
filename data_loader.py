from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class VQAv2(Dataset):

    def __init__(self, root, train, seqlen=14):
        """
        root (str): path to data directory
        train (bool): training or validation
        seqlen (int): maximum words in a question
        """
        if train:
            prefix = 'train'
        else:
            prefix = 'val'
        print("Loading preprocessed files... ({})".format(prefix))
        qas = pickle.load(open(os.path.join(root, prefix + '_qa.pkl'), 'rb'))
        idx2word, word2idx = pickle.load(open(os.path.join(root, 'dict_q.pkl'), 'rb'))
        idx2ans, ans2idx = pickle.load(open(os.path.join(root, 'dict_ans.pkl'), 'rb'))

        print("Setting up everything... ({})".format(prefix))
        self.vqas = []
        for qa in tqdm(qas):
            que = []
            for i, word in enumerate(qa['question_toked']):
                if i == seqlen:
                    break
                que.append(word2idx.get(word, 1))

            ans = np.zeros(len(idx2ans), dtype=np.float32)
            for a, s in qa['answer']:
                ans[ans2idx[a]] = s

            self.vqas.append({
                'v': os.path.join('data','img_feats', prefix, prefix + '_{}.npy'.format(qa['image_id'])),
                'q': que,
                'a': ans,
                'q_txt': qa['question'],
                'a_txt': qa['answer'],
                'q_id': qa['question_id']
            })

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        return torch.Tensor(np.load(self.vqas[idx]['v'])), \
               torch.Tensor(self.vqas[idx]['q']).long(), \
               torch.Tensor(self.vqas[idx]['a']), \
               self.vqas[idx]['q_txt'], \
               self.vqas[idx]['a_txt'], \
               self.vqas[idx]['question_id']

    @staticmethod
    def get_n_classes(fpath=os.path.join('data', 'dict_ans.pkl')):
        idx2ans, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2ans)

    @staticmethod
    def get_vocab_size(fpath=os.path.join('data', 'dict_q.pkl')):
        idx2word, _ = pickle.load(open(fpath, 'rb'))
        return len(idx2word)


def collate_fn(data):

    def merge(batch):
        return torch.stack(tuple(b for b in batch), 0)
    
    def merge_seq(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, torch.Tensor(lengths)

    data.sort(key=lambda x: len(x[1]), reverse=True)

    v, q, a, q_txt, a_txt, q_id = zip(*data)

    v = merge(v)
    q, q_lens = merge_seq(q)
    a = merge(a)

    return v, q ,a, q_lens, q_txt, a_txt, q_id

def prepare_data(args):

    train_loader = torch.utils.data.DataLoader(VQAv2(root=args.data_root, train=True),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(VQAv2(root=args.data_root, train=False),
                                             batch_size=args.vbatch_size,
                                             shuffle=False,
                                             num_workers=args.n_workers,
                                             collate_fn=collate_fn)

    vocab_size = VQAv2.get_vocab_size()
    num_classes = VQAv2.get_n_classes()
    return train_loader, val_loader, vocab_size, num_classes
