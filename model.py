import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GatedTanh(nn.Module):

    def __init__(self, inp_size, out_size):

        super(GatedTanh, self).__init__()
        self.i2t = nn.Linear(inp_size, out_size)  # input transform
        self.i2g = nn.Linear(inp_size, out_size)  # input to gate

    def forward(self, data):

        inp2transform = torch.tanh(self.i2t(data))
        inp2gate = torch.sigmoid(self.i2g(data))
        gated_transform = torch.mul(inp2transform, inp2gate)

        return gated_transform


class MlpLayer(nn.Module):

    def __init__(self, input_size, output_size):

        super(MlpLayer, self).__init__()
        self.mlp = nn.Linear(input_size, output_size)
        self.acts = ['relu', 'tanh']

    def forward(self, data, act='relu'):
        output = self.mlp(data)
        if act=='relu':
            output = F.relu(output)

        return output


class QuestionEncoder(nn.Module):

    def __init__(self, vocab_size, embed_dim, gru_hidden_size):

        super(QuestionEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size + 1, embed_dim)

        pretrained_wemb = np.zeros((vocab_size + 1, embed_dim), dtype=np.float32)
        pretrained_wemb[:vocab_size] = np.load("data/glove_pretrained_300.npy")
        self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_wemb))

        self.encoder = nn.GRU(embed_dim, gru_hidden_size)
        self.do1 = nn.Dropout(0.1)
        self.do2 = nn.Dropout(0.2)

    def forward(self, data, q_lens):
        data = self.embeddings(data)
        data = self.do1(data)
        data = data.permute(1,0,2)  # 512x14x300 --> 14x512x300
        data = torch.nn.utils.rnn.pack_padded_sequence(data, q_lens)
        self.encoder.flatten_parameters()   # Multi-GPU Training
        outputs, hidden = self.encoder(data)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        return self.do2(hidden.squeeze())


class TopDownAttention(nn.Module):

    def __init__(self, inp_size, hidden_size):

        super(TopDownAttention, self).__init__()
        #self.nonlinear = GatedTanh(inp_size, hidden_size)
        self.nonlinear = MlpLayer(inp_size, hidden_size)
        self.attn_layer = nn.Linear(hidden_size, 1)

    def forward(self, data):

        gated_transform = self.nonlinear(data)
        attn_scores = self.attn_layer(gated_transform)
        attn_probs = torch.softmax(attn_scores, dim=1)

        return attn_probs

class ImageEncoder(nn.Module):

    def __init__(self, inp_size, hidden_size):

        super(ImageEncoder, self).__init__()
        self.attention = TopDownAttention(inp_size, hidden_size)
        self.do1 = nn.Dropout(0.2)

    def forward(self, img_features, ques_features):

        ques_features = ques_features.unsqueeze(1).expand(-1,36,-1)  # N * k * enc_size
        concat_features = torch.cat((ques_features, img_features), 2)  # N * k * (enc_size + img_size)
        attn_probs = self.attention(concat_features)   # N * k * 1
        img_encode = torch.sum(torch.mul(img_features, attn_probs), dim=1)
        img_encode = self.do1(img_encode)

        return img_encode

class JointEmbedding(nn.Module):

    def __init__(self, ques_inp_size, img_inp_size, out_size):

        super(JointEmbedding, self).__init__()
        #self.ques_nonlinear = GatedTanh(ques_inp_size, out_size)
        #self.img_nonlinear = GatedTanh(img_inp_size, out_size)
        self.ques_nonlinear = MlpLayer(ques_inp_size, out_size)
        self.img_nonlinear = MlpLayer(img_inp_size, out_size)

    def forward(self, ques_features, img_features):

        ques_features = self.ques_nonlinear(ques_features)  # N * 512
        img_features = self.img_nonlinear(img_features)     # N * 2048  ->  N * 512
        joint_embed = torch.mul(ques_features, img_features)  # N * 512

        return joint_embed

class HybridClassifier(nn.Module):

    def __init__(self, inp_size, text_embed_size, img_embed_size, num_answers):

        super(HybridClassifier, self).__init__()
        #self.ques_nonlinear = GatedTanh(inp_size, text_embed_size)
        #self.img_nonlinear = GatedTanh(inp_size, img_embed_size)
        self.ques_nonlinear = MlpLayer(inp_size, text_embed_size)
        self.img_nonlinear = MlpLayer(inp_size, img_embed_size)
        
        self.ques_linear = nn.Linear(text_embed_size, num_answers)
        self.img_linear = nn.Linear(img_embed_size, num_answers)

    def forward(self, joint_embed):

        ques_embed = self.ques_nonlinear(joint_embed)
        img_embed = self.img_nonlinear(joint_embed)
        ques_out = self.ques_linear(ques_embed)
        img_out = self.img_linear(img_embed)
        joint_output = torch.sigmoid(torch.add(ques_out, img_out))

        return joint_output


class BasicClassifier(nn.Module):

    def __init__(self, joint_embed_size, text_embed_size, num_answers):

        super(BasicClassifier, self).__init__()
        #self.nonlinear = GatedTanh(joint_embed_size, text_embed_size)
        self.nonlinear = MlpLayer(joint_embed_size, text_embed_size)
        self.classifier = nn.Linear(text_embed_size, num_answers)
        self.do = nn.Dropout(0.5)

    def forward(self, joint_embed):

        output = self.classifier(self.do(self.nonlinear(joint_embed)))

        return output

class Model(nn.Module):

    def __init__(self, vocab_size, word_embed_dim, hidden_size, resnet_out, num_answers):

        super(Model, self).__init__()
        self.ques_encoder = QuestionEncoder(vocab_size,
                                            word_embed_dim,
                                            hidden_size)

        self.img_encoder = ImageEncoder(resnet_out + hidden_size,
                                        hidden_size)

        self.joint_embed = JointEmbedding(hidden_size,
                                          resnet_out,
                                          hidden_size)

        self.classifier = BasicClassifier(hidden_size,
                                          word_embed_dim,
                                          num_answers)

    def forward(self, images, questions, q_lens):

        ques_enc = self.ques_encoder(questions, q_lens)
        img_enc = self.img_encoder(images, ques_enc)
        joint_embed = self.joint_embed(ques_enc, img_enc)
        outputs = self.classifier(joint_embed)

        return outputs















