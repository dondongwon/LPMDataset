import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torchtext
import torchvision
from transformers import BertModel
import pdb

def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)


def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)

    samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(
        mu.unsqueeze(1))
    return samples

class UncertaintyModuleImage(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return {
            'logsigma': out,
            'attention': attn,
        }


class UncertaintyModuleText(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.rnn = nn.GRU(d_in, d_out // 2, bidirectional=True, batch_first=True)
        self.embed_dim = d_out

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, pad_mask=None, lengths=None):
        lengths = lengths.cpu()
        residual, attn = self.attention(x, pad_mask)

        # Forward propagate RNNs
        
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        lengths = lengths.cuda()
        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        gru_out = torch.gather(padded[0], 1, I).squeeze(1)

        out = self.fc(residual) + gru_out

        return {
            'logsigma': out,
            'attention': attn,
        }

class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn


class PIENet(nn.Module):
    """Polysemous Instance Embedding (PIE) module"""

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(PIENet, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.num_embeds > 1:
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
        out = self.layer_norm(out + residual)
        return out, attn, residual

def get_cnn(arch, pretrained):
  return torchvision.models.__dict__[arch](pretrained=pretrained) 

class EncoderImage(nn.Module):
    def __init__(self, config):
        super(EncoderImage, self).__init__()

        embed_dim = int(config.embed_dim)
        self.use_attention = config.img_attention
        self.use_probemb = config.img_probemb

        # Backbone CNN
        self.cnn = get_cnn(config.cnn_type, True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = nn.Linear(cnn_dim, embed_dim)

        self.cnn.fc = nn.Sequential()

        if self.use_attention:
            self.pie_net = PIENet(1, cnn_dim, embed_dim, cnn_dim // 2)

        if self.use_probemb:
            self.uncertain_net = UncertaintyModuleImage(cnn_dim, embed_dim, cnn_dim // 2)

        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = config.img_finetune

        self.n_samples_inference = int(config.n_samples_inference)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = self.fc(pooled)

        output = {}
        out_7x7 = out_7x7.view(-1, self.cnn_dim, 7 * 7)

        if self.use_attention:
            out, attn, residual = self.pie_net(out, out_7x7.transpose(1, 2))
            output['attention'] = attn
            output['residual'] = residual

        if self.use_probemb:
            uncertain_out = self.uncertain_net(pooled, out_7x7.transpose(1, 2))
            logsigma = uncertain_out['logsigma']
            output['logsigma'] = logsigma
            output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)

        if self.use_probemb and self.n_samples_inference:
            output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_samples_inference)
        else:
            output['embedding'] = out

        return output


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
    ind = torch.arange(0, max_length).unsqueeze(0).to(lengths.device)
    mask = (ind >= lengths.unsqueeze(1)) if set_pad_to_one \
        else (ind < lengths.unsqueeze(1))
    mask = mask.to(lengths.device)
    return mask


class EncoderText(nn.Module):
    def __init__(self, word2idx, opt):
        super(EncoderText, self).__init__()

        wemb_type, word_dim, embed_dim = \
            opt.wemb_type, opt.word_dim, int(opt.embed_dim)

        self.wemb_type = wemb_type

        self.embed_dim = embed_dim
        self.use_attention = opt.txt_attention
        self.use_probemb = opt.txt_probemb

        # Word embedding
        self.embed = nn.Embedding(len(word2idx), word_dim)
        self.embed.weight.requires_grad = opt.txt_finetune

        # Sentence embedding
        self.rnn = nn.GRU(word_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        if self.use_attention:
            self.pie_net = PIENet(1, word_dim, embed_dim, word_dim // 2)

        self.uncertain_net = UncertaintyModuleText(word_dim, embed_dim, word_dim // 2)

        if self.wemb_type == 'glove' or self.wemb_type == 'fasttext' :
            self.init_weights(wemb_type, word2idx, word_dim)
        else:
            self.bert_model = BertModel.from_pretrained("bert-base-uncased")


        self.n_samples_inference = int(opt.n_samples_inference)
        

    def init_weights(self, wemb_type, word2idx, word_dim):
        if wemb_type is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText()
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache="../kumon_dep/vector_cache")
                
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            print('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        # Embed word ids to vectors
        if self.wemb_type == 'glove' or self.wemb_type == 'fasttext' :
            wemb_out = self.embed(x)
        if self.wemb_type == 'bert':   
            with torch.no_grad():
                wemb_out = self.bert_model(x)[0]

        # Forward propagate RNNs
        lengths = lengths.cpu()
        packed = pack_padded_sequence(wemb_out, lengths, batch_first=True, enforce_sorted=False)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        lengths = lengths.cuda()
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        out = torch.gather(padded[0], 1, I).squeeze(1)

        output = {}

        if self.use_attention:
            
            pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
            out, attn, residual = self.pie_net(out, wemb_out, pad_mask)
            output['attention'] = attn
            output['residual'] = residual

        if self.use_probemb:
            if not self.use_attention:
                pad_mask = get_pad_mask(wemb_out.shape[1], lengths, True)
            uncertain_out = self.uncertain_net(wemb_out, pad_mask, lengths)
            logsigma = uncertain_out['logsigma']
            output['logsigma'] = logsigma
            output['uncertainty_attention'] = uncertain_out['attention']

        out = l2_normalize(out)

        if self.use_probemb and self.n_samples_inference:
            output['embedding'] = sample_gaussian_tensors(out, logsigma, self.n_samples_inference)
        else:
            output['embedding'] = out

        return output



class PCME(nn.Module):
    """Probabilistic CrossModal Embedding (PCME) module"""
    def __init__(self, word2idx, config):
        super(PCME, self).__init__()

        self.embed_dim = config.embed_dim

        self.n_embeddings = config.n_samples_inference

        self.img_enc = EncoderImage(config)
        self.txt_enc = EncoderText(word2idx, config)

    def forward(self, images, sentences, lengths):
        image_output = self.img_enc(images)
        caption_output = self.txt_enc(sentences, lengths)

        return {
            'image_features': image_output['embedding'],
            'image_attentions': image_output.get('attention'),
            'image_residuals': image_output.get('residual'),
            'image_logsigma': image_output.get('logsigma'),
            'image_logsigma_att': image_output.get('uncertainty_attention'),
            'caption_features': caption_output['embedding'],
            'caption_attentions': caption_output.get('attention'),
            'caption_residuals': caption_output.get('residual'),
            'caption_logsigma': caption_output.get('logsigma'),
            'caption_logsigma_att': caption_output.get('uncertainty_attention'),
        }

    def image_forward(self, images):
        return self.img_enc(images)

    def text_forward(self, sentences, lengths):
        return self.txt_enc(sentences, lengths)