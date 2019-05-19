import random
import json

import torch
from torch import nn
from tqdm import tqdm

from model import QuestionEncoder, ImageEncoder, JointEmbedding
from data_loader import prepare_data
from arguments import get_args


class Model(nn.Module):

    def __init__(self, vocab_size, word_embed_dim, hidden_size, resnet_out):

        super(Model, self).__init__()
        self.ques_encoder = QuestionEncoder(vocab_size,
                                            word_embed_dim,
                                            hidden_size)

        self.img_encoder = ImageEncoder(resnet_out + hidden_size,
                                        hidden_size)

        self.joint_embed = JointEmbedding(hidden_size,
                                          resnet_out,
                                          hidden_size)

    def forward(self, images, questions, q_lens):

        ques_enc = self.ques_encoder(questions, q_lens)
        img_enc = self.img_encoder(images, ques_enc)
        joint_embed = self.joint_embed(ques_enc, img_enc)

        return joint_embed


def evaluate(eval_loader, model, device, loader_type):
    model.eval()

    feats_data = dict()
    for step, (v, q, _, q_lens, _, _, q_id) in enumerate(tqdm(eval_loader, ascii=True)):
        v = v.to(device)
        q = q.to(device)
        q_lens = q_lens.to(device)

        joint_embed = model(v, q, q_lens)

        for idx, data in zip(joint_embed.cpu().numpy()):
            feats_data[idx] = data

    json.dump(feats_data, open(loader_type+'_joint_feats.json', 'w+'))


def main():
    parser = get_args()
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise NameError("Argument {} not recognized".format(unparsed))

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

    train_loader, val_loader, vocab_size, num_answers = prepare_data(args)

    model = Model(vocab_size, args.word_embed_dim, args.hidden_size, args.resnet_out)
    model = nn.DataParallel(model).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)

    evaluate(train_loader, model, device, "train")
    evaluate(val_loader, model, device, "val")

if __name__ == "__main__":
    main()

