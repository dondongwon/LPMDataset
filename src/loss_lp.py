import torch
import torch.nn as nn
import numpy as np
import pdb

def cosine_sim(x, y):
  """Cosine similarity between all the image and sentence pairs. Assumes x and y are l2 normalized"""
  return x.mm(y.t())

def order_sim(x, y):
  """Order embeddings similarity measure $max(0, x-y)$"""
  YmX = (y.unsqueeze(1).expand(y.size(0), x.size(0), y.size(1)) - \
          x.unsqueeze(0).expand(y.size(0), x.size(0), y.size(1)))
  score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
  return score

def l2norm(x):
  """L2-normalize columns of x"""
  norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
  return torch.div(x, norm)

def rbf(x, y, gamma):
  """RBF kernel K(x,y) """
  pdist = torch.norm(x[:, None] - y, dim=2, p=2)
  return torch.exp(-gamma * pdist)


class PVSELoss(nn.Module):

  def __init__(self, opt, reduction='mean'):
    super(PVSELoss, self).__init__()

    self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
    self.num_embeds = opt.num_embeds if hasattr(opt, 'num_embeds') else 1
    self.mmd_weight = opt.mmd_weight if hasattr(opt, 'mmd_weight') else 0.
    self.div_weight = opt.div_weight if hasattr(opt, 'div_weight') else 0.
    self.sim_fn = order_sim if hasattr(opt, 'order') and opt.order else cosine_sim
    self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
    self.reduction = reduction

    if self.num_embeds > 1:
      self.max_pool = torch.nn.MaxPool2d(self.num_embeds)


  def diversity_loss(self, x):
    x = l2norm(x) # Columns of x MUST be l2-normalized
    gram_x = x.bmm(x.transpose(1,2))
    I = torch.autograd.Variable((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1))
    if torch.cuda.is_available():
      I = I.cuda()
    gram_x.masked_fill_(I, 0.0)
    loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (self.num_embeds**2)
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def mmd_rbf_loss(self, x, y, gamma=None):
    if gamma is None:
      gamma = 1./x.size(-1)
    loss = rbf(x, x, gamma) - 2 * rbf(x, y, gamma) + rbf(y, y, gamma)
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def triplet_ranking_loss(self, A, B, I, max_dim):
    loss = (self.margin + A - B).clamp(min=0.0)
    loss.masked_fill_(I, 0.0)
    if self.max_violation:
      loss = loss.max(max_dim)[0]
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def forward(self, img, txt, img_r, txt_r):
    loss, losses = 0, dict()

    # compute image-sentence score matrix
    if self.num_embeds > 1:
      # 
      scores = self.sim_fn(img.view(-1, img.size(-1)), txt.view(-1, txt.size(-1)))
      scores = self.max_pool(scores.unsqueeze(0)).squeeze()
    
    else:
      scores = self.sim_fn(img.squeeze(), txt)
    try:
      diagonal = scores.diag().view(img.size(0), 1)
    except Exception:
      scores = scores.unsqueeze(0).unsqueeze(0)
      diagonal = scores.diag().view(img.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    mask = torch.eye(scores.size(0)) > .5
    I = torch.autograd.Variable(mask)
    if torch.cuda.is_available():
      I = I.cuda()

    # compare every diagonal score to scores in its column (image-to-text retrieval)
    i2t_loss = self.triplet_ranking_loss(scores, d1, I, 1)

    # compare every diagonal score to scores in its row (text-to-image retrieval)
    t2i_loss = self.triplet_ranking_loss(scores, d2, I, 0)

    ranking_loss = i2t_loss + t2i_loss
    loss += ranking_loss
    losses['ranking_loss'] = ranking_loss

    # diversity loss
    if self.num_embeds > 1 and self.div_weight > 0.:
      
      div_loss =  self.diversity_loss(txt_r) #+ self.diversity_loss(img_r) dong: only do diversity loss for text
      loss += self.div_weight * div_loss
      losses['div_loss'] = div_loss

    #domain discrepancy loss
    if self.num_embeds > 1 and self.mmd_weight > 0.:
      mmd_loss = self.mmd_rbf_loss(img.view(-1, img.size(-1)), txt.view(-1, txt.size(-1)), gamma=0.5)
      loss += self.mmd_weight * mmd_loss
      losses['mmd_loss'] = mmd_loss

    return loss, losses


"""Batch-wise efficient probabilistic embedding loss for cross-modal retrieval
PCME
"""

def batchwise_cdist(samples1, samples2, eps=1e-6):
    """Compute L2 distance between each pair of the two multi-head embeddings in batch-wise.
    We may assume that samples have shape N x K x D, N: batch_size, K: number of embeddings, D: dimension of embeddings.
    The size of samples1 and samples2 (`N`) should be either
    - same (each sample-wise distance will be computed separately)
    - len(samples1) = 1 (samples1 will be broadcasted into samples2)
    - len(samples2) = 1 (samples2 will be broadcasted into samples1)
    The following broadcasting operation will be computed:
    (N x 1 x K x D) - (N x K x 1 x D) = (N x K x K x D)
    Parameters
    ----------
    samples1: torch.Tensor (shape: N x K x D)
    samples2: torch.Tensor (shape: N x K x D)
    Returns
    -------
    batchwise distance: N x K ** 2
    """
    if len(samples1.size()) != 3 or len(samples2.size()) != 3:
        raise RuntimeError('expected: 3-dim tensors, got: {}, {}'.format(samples1.size(), samples2.size()))

    if samples1.size(0) == samples2.size(0):
        batch_size = samples1.size(0)
    elif samples1.size(0) == 1:
        batch_size = samples2.size(0)
    elif samples2.size(0) == 1:
        batch_size = samples1.size(0)
    else:
        raise RuntimeError(f'samples1 ({samples1.size()}) and samples2 ({samples2.size()}) dimensionalities '
                           'are non-broadcastable.')

    samples1 = samples1.unsqueeze(1)
    samples2 = samples2.unsqueeze(2)
    return torch.sqrt(((samples1 - samples2) ** 2).sum(-1) + eps).view(batch_size, -1)


def soft_contrastive_nll(logit, matched):
    """Compute the negative log-likelihood of the soft contrastive loss.
    .. math::
        NLL_{ij} = -\log p(m = m_{ij} | z_i, z_j)
                 = -\log \left[ \mathbb{I}_{m_{ij} = 1} \sigma(-a \| z_i - z_j \|_2 + b)
                         +  \mathbb{I}_{m_{ij} = -1} (1 - \sigma(-a \| z_i - z_j \|_2 + b)) \right].
    Note that the matching indicator {m_ij} is 1 if i and j are matched otherwise -1.
    Here we define the sigmoid function as the following:
    .. math::
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.
    Here we sample "logit", s_{ij} by Monte-Carlo sampling to get the expected soft contrastive loss.
    .. math::
        s_{ij}^k = -a \| z_i^k - z_j^k \|_2 + b, z_i^k ~ \mathcal N (\mu_i, \Sigma_i), z_j^k ~ \mathcal N (\mu_j, \Sigma_j).
    Then we can compute NLL by logsumexp (here, we omit `k` in s_{ij}^k for the simplicity):
    .. math::
        NLL_{ij} = -\log \left[ \frac{1}{K^2} \sum_{s_{ij}} \left{ \frac{\exp(s_{ij} m_ij)}{\exp(s_{ij}) + \exp(-s_{ij})} \right} \right]
                 = (\log K^2) -\log \sum_{s_{ij}} \left[ \exp \left( s_{ij} m_ij - \log(\exp(s_{ij} + (-s_{ij}))) \right) \right]
                 = (\log K^2) -logsumexp( s_{ij} m_{ij} - logsumexp(s_{ij}, -s_{ij}) ).
    Parameters
    ----------
    logit: torch.Tensor (shape: N x K ** 2)
    matched: torch.Tensor (shape: N), an element should be either 1 (matched) or -1 (mismatched)
    Returns
    -------
    NLL loss: torch.Tensor (shape: N), should apply `reduction` operator for the backward operation.
    """
    if len(matched.size()) == 1:
        matched = matched[:, None]
    return -(
        (logit * matched - torch.stack(
            (logit, -logit), dim=2).logsumexp(dim=2, keepdim=False)
         ).logsumexp(dim=1)) + np.log(logit.size(1))


class MCSoftContrastiveLoss(nn.Module):
    r"""Creates a criterion that measures the pairwise soft contrastive loss given
    input tensor pairs :math:`X`, :math:`Y` where each tensor is already sampled from a distribution.
    .. math::
        \log p(m = \hat m | x, y)
        p(m = 1 | x, y) = \sigma(-a \| x - y \|_2 + b)
        p(m = 0 | x, y) = 1 - \sigma(-a \| x - y \|_2 + b)
        \sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(-x)}, \text{ i.e., }
        1 - \sigma(x) = \frac{\exp(-x)}{\exp(x) + \exp(-x)}.
    This code assumes that :math:`x_i` and :math:`y_j` are in same class if i = j,
    and in different class otherwise.
    The division by :math:`n` can be avoided if sets ``reduction = 'sum'``.
    Parameters
    ----------
    TBD
    Shape
    -----
    Input1 : torch.Tensor
        :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
    Input2: torch.Tensor
        :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
    Output: torch.Tensor
        If :attr:`reduction` is ``'none'``, then :math:`(N)`.
    """
    def __init__(self, config, reduction='sum'):
        super().__init__()
        if reduction not in {'mean', 'sum', None}:
            raise ValueError('unknown reduction {}'.format(reduction))
        self.reduction = reduction


        shift = int(config.init_shift) * torch.ones(1).cuda()
        negative_scale = int(config.init_negative_scale) * torch.ones(1).cuda()

        shift = nn.Parameter(shift)
        negative_scale = nn.Parameter(negative_scale)

        self.register_parameter('shift', shift)
        self.register_parameter('negative_scale', negative_scale)

        self.num_samples = config.num_samples

        self.uniform_lambda = int(config.uniform_lambda)
        self.vib_beta = float(config.vib_beta)

    def uniform_loss(self, x, max_samples=16384, t=2):
        if len(x) ** 2 > max_samples:
            # prevent CUDA error: https://github.com/pytorch/pytorch/issues/22313
            indices = np.random.choice(len(x), int(np.sqrt(max_samples)))
            x = x[indices]
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def kl_divergence(self, mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum()

    def pairwise_sampling(self, anchors, candidates):
        N = len(anchors)
        if len(anchors) != len(candidates):
            raise RuntimeError('# anchors ({}) != # candidates ({})'.format(anchors.shape, candidates.shape))
        anchor_idx, selected_idx, matched = self.full_sampling(N)

        anchor_idx = torch.from_numpy(np.array(anchor_idx)).long()
        selected_idx = torch.from_numpy(np.array(selected_idx)).long()
        matched = torch.from_numpy(np.array(matched)).float()

        anchor_idx = anchor_idx.to(anchors.device)
        selected_idx = selected_idx.to(anchors.device)
        matched = matched.to(anchors.device)

        anchors = anchors[anchor_idx]
        selected = candidates[selected_idx]

        cdist = batchwise_cdist(anchors, selected)

        return cdist, matched

    def full_sampling(self, N):
        candidates = []
        selected = []
        matched = []
        for i in range(N):
            for j in range(N):
                candidates.append(i)
                selected.append(j)
                if i == j:
                    matched.append(1)
                else:
                    matched.append(-1)
        return candidates, selected, matched

    def _compute_loss(self, input1, input2):
        """
        Shape
        -----
        Input1 : torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Input2: torch.Tensor
            :math:`(N, K, D)` shape, `N` is the batch size, `K` is the number of samples and `D` is the size of a sample.
        Output: torch.Tensor
            If :attr:`reduction` is ``'none'``, then :math:`(N)`.
        """
        distance, matched = self.pairwise_sampling(input1, input2)
        logits = -self.negative_scale * distance + self.shift

        idx = matched == 1
        loss_pos = soft_contrastive_nll(logits[idx], matched[idx]).sum()
        idx = matched != 1
        loss_neg = soft_contrastive_nll(logits[idx], matched[idx]).sum()

        return {
            'loss': loss_pos + loss_neg,
            'pos_loss': loss_pos,
            'neg_loss': loss_neg,
        }

    def match_prob(self, image_features, caption_features, image_logsigma, caption_logsigma, use_batchwise_cdist=True):
        sampled_image_features, sampled_caption_features = image_features, caption_features
        distance = batchwise_cdist(sampled_image_features, sampled_caption_features)

        distance = distance.to(self.negative_scale.device)
        distance = distance.float()
        logits = -self.negative_scale * distance + self.shift
        prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))

        return prob.mean(axis=1)

    def forward(self, image_features, caption_features, image_logsigma, caption_logsigma):
        uniform_loss = 0
        uniform_loss_val = 0
        vib_loss = 0
        vib_loss_val = 0

        if self.uniform_lambda != 0:
            dim = image_features.size()[-1]
            uniform_loss = self.uniform_loss(torch.cat([image_features.view(-1, dim), caption_features.view(-1, dim)]))
            uniform_loss_val = uniform_loss.item()
        sampled_image_features, sampled_caption_features = image_features, caption_features

        if self.vib_beta != 0:
            vib_loss =\
                self.kl_divergence(image_features.mean(dim=1), image_logsigma) + self.kl_divergence(caption_features.mean(dim=1), caption_logsigma)
            vib_loss_val = vib_loss.item()

        i2t_loss = self._compute_loss(sampled_image_features, sampled_caption_features)
        t2i_loss = self._compute_loss(sampled_caption_features, sampled_image_features)
        loss = i2t_loss['loss'] + t2i_loss['loss'] + self.uniform_lambda * uniform_loss + self.vib_beta * vib_loss

        loss_dict = {'i2t_loss': i2t_loss['loss'].item(),
                     't2i_loss': t2i_loss['loss'].item(),
                     'i2t_pos_loss': i2t_loss['pos_loss'].item(),
                     'i2t_neg_loss': i2t_loss['neg_loss'].item(),
                     't2i_pos_loss': t2i_loss['pos_loss'].item(),
                     't2i_neg_loss': t2i_loss['neg_loss'].item(),
                     'uniform_loss': uniform_loss_val,
                     'vib_loss': vib_loss_val,
                     'shift': self.shift.item(),
                     'negative_scale': self.negative_scale.item(),
                     'loss': loss.item()}
        return loss, loss_dict

class OursLoss(nn.Module):

  def __init__(self, opt, reduction='mean'):
    super().__init__()

    self.margin = opt.margin if hasattr(opt, 'margin') else 1.0
    self.num_embeds = opt.num_embeds if hasattr(opt, 'num_embeds') else 1
    self.mmd_weight = opt.mmd_weight if hasattr(opt, 'mmd_weight') else 0.
    self.div_weight = opt.div_weight if hasattr(opt, 'div_weight') else 0.
    self.sim_fn = order_sim if hasattr(opt, 'order') and opt.order else cosine_sim
    self.max_violation = opt.max_violation if hasattr(opt, 'max_violation') else False
    self.reduction = reduction

    if self.num_embeds > 1:
      self.max_pool = torch.nn.MaxPool2d((1,self.num_embeds))


  def diversity_loss(self, x):
    x = l2norm(x) # Columns of x MUST be l2-normalized
    gram_x = x.bmm(x.transpose(1,2))
    I = torch.autograd.Variable((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1))
    if torch.cuda.is_available():
      I = I.cuda()
    gram_x.masked_fill_(I, 0.0)
    loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (self.num_embeds**2)
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def mmd_rbf_loss(self, x, y, gamma=None):
    if gamma is None:
      gamma = 1./x.size(-1)
    loss = rbf(x, x, gamma) - 2 * rbf(x, y, gamma) + rbf(y, y, gamma)
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def triplet_ranking_loss(self, A, B, I, max_dim):
    loss = (self.margin + A - B).clamp(min=0.0)
    loss.masked_fill_(I, 0.0)
    if self.max_violation:
      loss = loss.max(max_dim)[0]
    return loss.mean() if self.reduction=='mean' else loss.sum()


  def forward(self, img, txt, txt_r):
    loss, losses = 0, dict()

    # compute image-sentence score matrix
    if self.num_embeds > 1:
      # 
      scores = self.sim_fn(img.view(-1, img.size(-1)), txt.view(-1, txt.size(-1)))
      scores = self.max_pool(scores.unsqueeze(0)).squeeze()
    
    else:
      scores = self.sim_fn(img, txt)
    try:
      diagonal = scores.diag().view(img.size(0), 1)
    except Exception:
      scores = scores.unsqueeze(0).unsqueeze(0)
      diagonal = scores.diag().view(img.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    mask = torch.eye(scores.size(0)) > .5
    I = torch.autograd.Variable(mask)
    if torch.cuda.is_available():
      I = I.cuda()

    # compare every diagonal score to scores in its column (image-to-text retrieval)
    i2t_loss = self.triplet_ranking_loss(scores, d1, I, 1)

    # compare every diagonal score to scores in its row (text-to-image retrieval)
    t2i_loss = self.triplet_ranking_loss(scores, d2, I, 0)

    ranking_loss = i2t_loss + t2i_loss
    loss += ranking_loss
    losses['ranking_loss'] = ranking_loss

    # diversity loss
    if self.num_embeds > 1 and self.div_weight > 0.:
      
      div_loss =  self.diversity_loss(txt_r)
      loss += self.div_weight * div_loss
      losses['div_loss'] = div_loss

    #domain discrepancy loss
    if self.num_embeds > 1 and self.mmd_weight > 0.:
      mmd_loss = self.mmd_rbf_loss(img.repeat(txt.size(1), 1), txt.view(-1, txt.size(-1)), gamma=0.5)
      loss += self.mmd_weight * mmd_loss
      losses['mmd_loss'] = mmd_loss

    return loss, losses