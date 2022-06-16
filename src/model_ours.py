import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torchtext
import torchvision
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pdb
from transformers import BertModel
from transformers import ViltProcessor, ViltModel
from PIL import Image
import torch.nn.functional as F

class Attention(torch.nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

    def forward(self, 
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor, # [seq_length, encoder_dim]
        ):
        weights = self._get_weights(query, values) # [seq_length]
        weights = torch.nn.functional.softmax(weights, dim=0)
        return weights @ values  # [encoder_dim]

class MultiplicativeAttention(Attention):

    def __init__(self, encoder_dim: int, decoder_dim: int):
        super().__init__(encoder_dim, decoder_dim)
        self.W = torch.nn.Parameter(torch.FloatTensor(
            self.decoder_dim, self.encoder_dim).uniform_(-0.1, 0.1))

    def _get_weights(self,
        query: torch.Tensor,  # [decoder_dim]
        values: torch.Tensor, # [seq_length, encoder_dim]
    ):
        weights = query @ self.W @ values.T  # [seq_length]
        return weights/np.sqrt(self.decoder_dim)  # [seq_length]




def get_cnn(arch, pretrained):
  return torchvision.models.__dict__[arch](pretrained=pretrained) 


def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
  return torch.div(x, norm)


def get_pad_mask(max_length, lengths, set_pad_to_one=True):
  ind = torch.arange(0, max_length).unsqueeze(0)
  if torch.cuda.is_available():
    ind = ind.cuda()
  mask = Variable((ind >= lengths.unsqueeze(1))) if set_pad_to_one \
    else Variable((ind < lengths.unsqueeze(1)))
  return mask.cuda() if torch.cuda.is_available() else mask


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
      mask = mask.repeat(self.n_head, 1, 1).permute(1,2,0)
      attn.masked_fill_(mask, -np.inf)
    attn = self.softmax(attn)

    output = torch.bmm(attn.transpose(1,2), x)
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

class Ours_VILT(nn.Module):

  def __init__(self, opt):
    super(Ours_VILT, self).__init__()

    self.mil = opt.num_embeds > 1
    self.fig_enc = ViltModel.from_pretrained("dandelin/vilt-b32-mlm-itm")
    self.txt_enc = EncoderTextNoTrace(opt)

    # self.fc = nn.Sequential(
    #            nn.Linear(opt.word_dim * 185,opt.embed_size),
    #            nn.ReLU()
    #           #  nn.Linear(2048,),
    #           #  nn.ReLU(),
    #            )

  def forward(self,fig_ocr, spoken_output, pointer_target, cap_lengths):
    
    fig_emb = self.fig_enc(**fig_ocr).pooler_output
    # pdb.set_trace()
    # fig_emb = fig_emb.last_hidden_state.flatten(1)
    # # fig_emb = self.fc(fig_emb.last_hidden_state.flatten(1))
    txt_emb, txt_attn, txt_residual = self.txt_enc(spoken_output, cap_lengths)
    return fig_emb, txt_emb, txt_attn, txt_residual

class EncoderTextNoTrace(nn.Module):
  def __init__(self, opt):
    super().__init__()

    wemb_type, word_dim, embed_size, num_embeds = \
      opt.wemb_type, opt.word_dim, opt.embed_size, opt.num_embeds

    self.wemb_type = wemb_type

    self.embed_size = embed_size
    self.use_attention = opt.txt_attention
    self.abs = True if hasattr(opt, 'order') and opt.order else False
    self.legacy = opt.legacy

    # Sentence embedding
    self.rnn = nn.GRU(word_dim, embed_size//2, bidirectional=True, batch_first=True)
    if self.use_attention:
      self.pie_net = PIENet(num_embeds, word_dim, embed_size, word_dim//2, opt.dropout)
    self.dropout = nn.Dropout(opt.dropout)
    
    self.bert_model = BertModel.from_pretrained("bert-base-uncased")

  def forward(self, x, lengths):
    # Embed word ids to vectors
    
    with torch.no_grad():
      wemb_out = self.bert_model(x)[0]
    wemb_out = self.dropout(wemb_out)

    # Forward propagate RNNs
    lengths = lengths.cpu()
    packed = pack_padded_sequence(wemb_out, lengths, batch_first=True, enforce_sorted=False)
    if torch.cuda.device_count() > 1:
      self.rnn.flatten_parameters()

    _, rnn_out = self.rnn(packed)
    # Reshape *final* output to (batch_size, hidden_size)
    rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(-1, self.embed_size)

    out = self.dropout(rnn_out)

    attn, residual = None, None
    if self.use_attention:
      pad_mask = get_pad_mask(wemb_out.shape[1], lengths.cuda(), True)
      out, attn, residual = self.pie_net(out, wemb_out, pad_mask)
    
    out = l2norm(out)
    if self.abs:
      out = torch.abs(out)
    return out, attn, residual


class Ours_VILT_Trace(nn.Module):

  def __init__(self, opt):
    super().__init__()

    self.mil = opt.num_embeds > 1
    self.fig_enc = ViltModel.from_pretrained("dandelin/vilt-b32-mlm-itm")
    self.txt_enc = EncoderTextTrace(opt)
    

    # self.fc = nn.Sequential(
    #            nn.Linear(opt.word_dim * 185,opt.embed_size),
    #            nn.ReLU()
    #           #  nn.Linear(2048,),
    #           #  nn.ReLU(),
    #            )

  def forward(self,fig_ocr, spoken_output, pointer_target, cap_lengths):
    
    fig_emb = self.fig_enc(**fig_ocr).pooler_output
    # pdb.set_trace()
    # fig_emb = fig_emb.last_hidden_state.flatten(1)
    # # fig_emb = self.fc(fig_emb.last_hidden_state.flatten(1))
    txt_emb, txt_attn, txt_residual = self.txt_enc(Variable(spoken_output), pointer_target, cap_lengths)
    return fig_emb, txt_emb, txt_attn, txt_residual

class EncoderTextTrace(nn.Module):
  def __init__(self, opt):
    super().__init__()

    wemb_type, word_dim, embed_size, num_embeds = \
      opt.wemb_type, opt.word_dim, opt.embed_size, opt.num_embeds

    self.wemb_type = wemb_type

    self.embed_size = embed_size
    self.use_attention = opt.txt_attention
    self.abs = True if hasattr(opt, 'order') and opt.order else False
    self.legacy = opt.legacy

    # Sentence embedding
    self.rnn = nn.GRU(word_dim, embed_size//2, bidirectional=True, batch_first=True)
    if self.use_attention:
      self.pie_net = PIENet(num_embeds, word_dim, embed_size, word_dim//2, opt.dropout)
    self.dropout = nn.Dropout(opt.dropout)
    
    self.bert_model = BertModel.from_pretrained("bert-base-uncased")

    self.gamma1 = nn.Linear(512, 128)
    self.gamma2 = nn.Linear(128, 256)
    self.gamma3 = nn.Linear(256, 512)
    self.beta1 = nn.Linear(4, 128)
    self.beta2 = nn.Linear(128, 256)
    self.beta3 = nn.Linear(256, 512)

    self.layernorm_1 = nn.LayerNorm(word_dim)
    self.layernorm_2 = nn.LayerNorm(word_dim)
    self.trace_fc = nn.Linear(word_dim, word_dim)

    self.W = torch.nn.Parameter(torch.FloatTensor(
            1, 512).uniform_(-0.1, 0.1))


  def FiLM(self, x, gamma, beta):

      x = gamma * x #+ beta

      return x

  def forward(self, x, traces, lengths):
    # Embed word ids to vectors
    
    with torch.no_grad():
      wemb_out = self.bert_model(x)[0]

    
    
    fusion = 'gumbel-softmax'

    if fusion == 'addition':
      traces = traces.float().unsqueeze(2).repeat(1,1,768)
      wemb_out += traces

      # wemb_out = traces * wemb_out
      # mask = (traces.sum(-1) == 0)
      # wemb_out = traces * wemb_out * (mask) + wemb_out * (1-mask)

      #attention masks
      # 1 1 0 0 0 0 0


    if fusion == 'gumbel-softmax':
      wemb_out = wemb_out + (F.gumbel_softmax(traces.float(), tau=1, hard=False).unsqueeze(2) * wemb_out)

      # 2 * wemb  

      #   wemb_out * traces 
      # |Fig| x 512 x 768  
      # |Fig| x 512  = [0, 0, 0] 
      
      # if there was an alignment between word and this figure 
      # gumbell soft max (1 x 512)

      
      wemb_out = self.layernorm_1(wemb_out)
      wemb_out = self.trace_fc(wemb_out)
      wemb_out = self.layernorm_2(wemb_out)
    
    if fusion == 'multiplicative-attention':
      traces = traces.float().unsqueeze(2)
      att =  traces @ self.W @ wemb_out
      wemb_out = wemb_out + att

      


    # if fusion == 'additive-attention': #Bahdanau Attention
    #   wemb_out = wemb_out + 

    # wemb_out = self.dropout(wemb_out)
    
    # Forward propagate RNNs
    lengths = lengths.cpu()
    packed = pack_padded_sequence(wemb_out, lengths, batch_first=True, enforce_sorted=False)
    if torch.cuda.device_count() > 1:
      self.rnn.flatten_parameters()

    _, rnn_out = self.rnn(packed)
    # Reshape *final* output to (batch_size, hidden_size)
    rnn_out = rnn_out.permute(1, 0, 2).contiguous().view(-1, self.embed_size)

    out = self.dropout(rnn_out)

    attn, residual = None, None
    if self.use_attention:
      pad_mask = get_pad_mask(wemb_out.shape[1], lengths.cuda(), True)
      out, attn, residual = self.pie_net(out, wemb_out, pad_mask)
    
    out = l2norm(out)
    if self.abs:
      out = torch.abs(out)
    return out, attn, residual


class VILT(nn.Module):

  def __init__(self, opt):
    super(VILT, self).__init__()

    self.mil = opt.num_embeds > 1
    self.fig_enc = ViltModel.from_pretrained("dandelin/vilt-b32-mlm-itm")
    self.txt_enc = EncoderTextNoTrace(opt)

    # self.fc = nn.Sequential(
    #            nn.Linear(opt.word_dim * 185,opt.embed_size),
    #            nn.ReLU()
    #           #  nn.Linear(2048,),
    #           #  nn.ReLU(),
    #            )

  def forward(self,fig_ocr, spoken_output, pointer_target, cap_lengths):
    
    fig_emb = self.fig_enc(**fig_ocr).pooler_output
    # fig_emb = fig_emb.last_hidden_state.flatten(1)
    # # fig_emb = self.fc(fig_emb.last_hidden_state.flatten(1))
    # txt_emb, txt_attn, txt_residual = self.txt_enc(spoken_output, cap_lengths)
    return fig_emb, txt_emb, txt_attn, txt_residual
