from __future__ import print_function
import os, sys
import pickle
import time
import glob

import numpy as np
import torch
import pdb
from model_lp import PVSE
from loss_lp import cosine_sim, order_sim
from vocab import Vocabulary
# from data import get_test_loader
from logger import AverageMeter
from option import parser, verify_input_args
import itertools

ORDER_BATCH_SIZE = 100

def encode_data(model, data_loader, use_gpu=False, ours = False):
  """Encode all images and sentences loadable by data_loader"""
  # switch to evaluate mode
  model.eval()
  
  use_mil = model.module.mil if hasattr(model, 'module') else model.mil
  # numpy array to keep all the embeddings
  img_embs, txt_embs = None, None
  all_img_ids = []
  for i, data in enumerate(data_loader):
    
    if ours:
      fig_ocr, spoken_output, pointer_target, cap_lengths, ids, img_ids = data
      if torch.cuda.is_available():
        for k,v in fig_ocr.items():
          fig_ocr[k] = v.cuda()
        spoken_output, pointer_target, cap_lengths =  spoken_output.cuda(), pointer_target.cuda(), cap_lengths.cuda()
      # Forward pass and compute loss; _a: attention map, _r: residuals
      if spoken_output.shape[0] == 1:
        for k,v in fig_ocr.items():
          fig_ocr[k] = v.unsqueeze(0)
      
      img_emb, txt_emb, txt_a, txt_r = model.forward(fig_ocr, spoken_output, pointer_target, cap_lengths)
      del fig_ocr, spoken_output, pointer_target, cap_lengths 
    
    else:
      images, spoken_output, ocr_target, cap_lengths, ids, img_ids = data
      if torch.cuda.is_available():
        images, spoken_output, cap_lengths = images.cuda(), spoken_output.cuda(), cap_lengths.cuda()
      # Forward pass and compute loss; _a: attention map, _r: residuals
      img_emb, txt_emb, _, _, _, _ = model.forward(images, spoken_output, cap_lengths)

      if not use_mil:
        img_emb = img_emb.squeeze()
        txt_emb = txt_emb.squeeze()
      

    # compute the embeddings
      del images, spoken_output, cap_lengths

    # initialize the output embeddings
    if img_embs is None:
      if use_gpu:
        emb_sz = [len(data_loader.dataset), img_emb.size(1), img_emb.size(2)] \
                if use_mil else [len(data_loader.dataset), img_emb.size(1)]
        img_embs = torch.zeros(emb_sz, dtype=img_emb.dtype, requires_grad=False).cuda()
        txt_embs = torch.zeros(emb_sz, dtype=txt_emb.dtype, requires_grad=False).cuda()
      else:
        if ours:
          img_emb_sz = (len(data_loader.dataset), img_emb.size(1))
          txt_emb_sz = (len(data_loader.dataset), txt_emb.size(1), txt_emb.size(2)) \
                  if use_mil else (len(data_loader.dataset), txt_emb.size(1))

        else:
          img_emb_sz = (len(data_loader.dataset), img_emb.size(1), img_emb.size(2)) \
                  if use_mil else (len(data_loader.dataset), img_emb.size(1))
          txt_emb_sz = (len(data_loader.dataset), txt_emb.size(1), txt_emb.size(2)) \
                  if use_mil else (len(data_loader.dataset), txt_emb.size(1))
          
        img_embs = np.zeros(img_emb_sz)
        txt_embs = np.zeros(txt_emb_sz)

    # preserve the embeddings by copying from gpu and converting to numpy
    img_embs[ids,...] = img_emb if use_gpu else img_emb.data.cpu().numpy().copy()
    txt_embs[ids,...] = txt_emb if use_gpu else txt_emb.data.cpu().numpy().copy()
    all_img_ids.append(img_ids)

  all_img_ids = list(itertools.chain(*all_img_ids))
  all_img_ids = [int(x) for x in all_img_ids]
  

  return img_embs, txt_embs, all_img_ids

def encode_data_pcme(model, data_loader, use_gpu=False):
  """Encode all images and sentences loadable by data_loader"""
  # switch to evaluate mode
  model.eval()
  

  # numpy array to keep all the embeddings
  img_embs, txt_embs = None, None
  all_img_ids = []
  for i, data in enumerate(data_loader):
    images, spoken_output, ocr_target, cap_lengths, ids, img_ids = data
    if torch.cuda.is_available():
      images, spoken_output, cap_lengths = images.cuda(), spoken_output.cuda(), cap_lengths.cuda()
    # Forward pass and compute loss; _a: attention map, _r: residuals
    output = model.forward(images, spoken_output, cap_lengths)
    
    img_emb, txt_emb = output['image_features'], output['caption_features']
    del images, spoken_output, cap_lengths

    # initialize the output embeddings
    if img_embs is None:
      if use_gpu:
        emb_sz = [len(data_loader.dataset), img_emb.size(1), img_emb.size(2)]
        img_embs = torch.zeros(emb_sz, dtype=img_emb.dtype, requires_grad=False).cuda()
        txt_embs = torch.zeros(emb_sz, dtype=txt_emb.dtype, requires_grad=False).cuda()
      else:
        emb_sz = (len(data_loader.dataset), img_emb.size(1), img_emb.size(2))
        img_embs = np.zeros(emb_sz)
        txt_embs = np.zeros(emb_sz)

    # preserve the embeddings by copying from gpu and converting to numpy
    img_embs[ids,...] = img_emb if use_gpu else img_emb.data.cpu().numpy().copy()
    txt_embs[ids,...] = txt_emb if use_gpu else txt_emb.data.cpu().numpy().copy()
    all_img_ids.append(img_ids)

  all_img_ids = list(itertools.chain(*all_img_ids))
  all_img_ids = [int(x) for x in all_img_ids]
  return img_embs, txt_embs, all_img_ids



def i2t(connect_json, all_img_ids, images, sentences, nreps=1, npts=None, return_ranks=False, order=False, use_gpu=False, random = False):
  """
  Images->Text (Image Annotation)
  Images: (nreps*N, K) matrix of images
  Captions: (nreps*N, K) matrix of sentences
  """
  if use_gpu:
    assert not order, 'Order embedding not supported in GPU mode'

  if npts is None:
    npts = int(images.shape[0] / nreps)

  index_list = []
  ranks, top1, top10 = np.zeros(npts), np.zeros(npts), np.zeros((npts, 10))
  for index in range(npts):
    # Get query image
    im = images[nreps * index]
    im = im.reshape((1,) + im.shape)
    if len(im.shape) == 2:
        im  = im[:,None,:]

    # Compute scores
    if use_gpu:
      if len(sentences.shape) == 2:
        sim = im.mm(sentences.t()).view(-1)
      else:
        _, K, D = im.shape
        sim_kk = im.view(-1, D).mm(sentences.view(-1, D).t())
        sim_kk = sim_kk.view(im.size(0), K, sentences.size(0), K)
        sim_kk = sim_kk.permute(0,1,3,2).contiguous()
        sim_kk = sim_kk.view(im.size(0), -1, sentences.size(0))
        sim, _ = sim_kk.max(dim=1)
        sim = sim.flatten()
    else:
      if order:
        if index % ORDER_BATCH_SIZE == 0:
          mx = min(images.shape[0], nreps * (index + ORDER_BATCH_SIZE))
          im2 = images[nreps * index:mx:nreps]
          sim_batch = order_sim(torch.Tensor(im2).cuda(), torch.Tensor(sentences).cuda())
          sim_batch = sim_batch.cpu().numpy()
        sim = sim_batch[index % ORDER_BATCH_SIZE]
      else:
        sim = np.tensordot(im, sentences, axes=[2, 2]).max(axis=(0,1,3)).flatten() \
            if len(sentences.shape) == 3 else np.dot(im, sentences.T).flatten()

    if use_gpu:
      _, inds_gpu = sim.sort()
      inds = inds_gpu.cpu().numpy().copy()[::-1]
    else:
      inds = np.argsort(sim)[::-1]
    index_list.append(inds[0])

    
    #from here
    if random:
      inds = np.random.permutation(inds) #random

    img_idx = all_img_ids[index]

    curr_best_ind = sim.shape[0]

    for k,v in connect_json.items(): #for all images in the same slide
      if str(img_idx) in v: #if
        slide_imgs_ids = v
        break
      else:
        continue


    for slide_id in slide_imgs_ids: #for each object
      index_of_int = all_img_ids.index(int(slide_id)) #find index
      curr_rank = np.where(inds == index_of_int)[0][0] #find rank
      if curr_rank < curr_best_ind:
        ranks[index] = curr_rank #assign rank
    top1[index] = inds[0]
    top10[index, :] = inds[:10]

    #to here

    # Score


    # rank = 1e20
    # for i in range(nreps * index, nreps * (index + 1), 1):
    #   tmp = np.where(inds == i)[0][0]
    #   if tmp < rank:
    #     rank = tmp
    # ranks[index] = rank
    # top1[index] = inds[0]

  # Compute metrics

  r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
  r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
  r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  medr = np.floor(np.median(ranks)) + 1
  meanr = ranks.mean() + 1
  if return_ranks:
    return (r1, r5, r10, medr, meanr), (ranks, top1, top10)
  else:
    return (r1, r5, r10, medr, meanr)



def t2i(connect_json, all_img_ids, images, sentences, nreps=1, npts=None, return_ranks=False, order=False, use_gpu=False, random = False):
  """
  Text->Images (Image Search)
  Images: (nreps*N, K) matrix of images
  Captions: (nreps*N, K) matrix of sentences
  """
  if use_gpu:
    assert not order, 'Order embedding not supported in GPU mode'

  if npts is None:
    npts = int(images.shape[0] / nreps)

  if use_gpu:
    ims = torch.stack([images[i] for i in range(0, len(images), nreps)])
  else:
    ims = np.array([images[i] for i in range(0, len(images), nreps)])
  
  if len(ims.shape) == 2:
        ims  = ims[:,None,:]

  ranks, top1, top10 = np.zeros(nreps * npts), np.zeros(nreps * npts), np.zeros((nreps*npts, 10))
  for index in range(npts):
    # Get query sentences
    queries = sentences[nreps * index:nreps * (index + 1)]


    # Compute scores
    if use_gpu:
      if len(sentences.shape) == 2:
        sim = queries.mm(ims.t())
      else:
        sim_kk = queries.view(-1, queries.size(-1)).mm(ims.view(-1, ims.size(-1)).t())
        sim_kk = sim_kk.view(queries.size(0), queries.size(1), ims.size(0), ims.size(1))
        sim_kk = sim_kk.permute(0,1,3,2).contiguous()
        sim_kk = sim_kk.view(queries.size(0), -1, ims.size(0))
        sim, _ = sim_kk.max(dim=1)
    else:
      if order:
        if nreps * index % ORDER_BATCH_SIZE == 0:
          mx = min(sentences.shape[0], nreps * index + ORDER_BATCH_SIZE)
          sentences_batch = sentences[nreps * index:mx]
          sim_batch = order_sim(torch.Tensor(images).cuda(),
                                torch.Tensor(sentences_batch).cuda())
          sim_batch = sim_batch.cpu().numpy()
        sim = sim_batch[:, (nreps * index) % ORDER_BATCH_SIZE:(nreps * index) % ORDER_BATCH_SIZE + nreps].T
      else:
        
        if len(sentences.shape) == 3:
          sim = np.tensordot(queries, ims, axes=[2, 2]).max(axis=(1,3)) 
        else:
          sim = np.dot(queries, ims.T.squeeze())

    inds = np.zeros(sim.shape)
    for i in range(len(inds)):
      if use_gpu:
        _, inds_gpu = sim[i].sort()
        inds[i] = inds_gpu.cpu().numpy().copy()[::-1]
      else:
        inds[i] = np.argsort(sim[i])[::-1]

        if random:
          inds[i] = np.random.permutation(inds[i]) #random


      img_idx = all_img_ids[index]

      curr_best_ind = sim.shape[1]
      for k,v in connect_json.items(): #for all images in the same slide
        if str(img_idx) in v:
          slide_imgs_ids = v

      for slide_id in slide_imgs_ids: #for each object
        index_of_int = all_img_ids.index(int(slide_id)) #find index
        curr_rank = np.where(inds[i] == index_of_int)[0][0] #find rank
        if curr_rank < curr_best_ind:
          ranks[nreps * index + i] = curr_rank #assign rank

      top1[nreps * index + i] = inds[i][0]
      top10[nreps * index + i, :] = inds[i][:10]

  # Compute metrics
  r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
  r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
  r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  medr = np.floor(np.median(ranks)) + 1
  meanr = ranks.mean() + 1
  if return_ranks:
    return (r1, r5, r10, medr, meanr), (ranks, top1, top10)
  else:
    return (r1, r5, r10, medr, meanr)


def convert_old_state_dict(x, model, multi_gpu=False):
  params = model.state_dict()
  prefix = ['module.img_enc.', 'module.txt_enc.'] \
      if multi_gpu else ['img_enc.', 'txt_enc.']
  for i, old_params in enumerate(x):
    for key, val in old_params.items():
      key = prefix[i] + key.replace('module.','').replace('our_model', 'pie_net')
      assert key in params, '{} not found in model state_dict'.format(key)
      params[key] = val
  return params



def evalrank(model, args, split='test'):
  print('Loading dataset')
  data_loader = get_test_loader(args, vocab)

  print('Computing results... (eval_on_gpu={})'.format(args.eval_on_gpu))
  img_embs, txt_embs = encode_data(model, data_loader, args.eval_on_gpu)
  n_samples = img_embs.shape[0]

  nreps = 5 if args.data_name == 'coco' else 1
  print('Images: %d, Sentences: %d' % (img_embs.shape[0] / nreps, txt_embs.shape[0]))

  # 5fold cross-validation, only for MSCOCO
  mean_metrics = None
  if args.data_name == 'coco':
    results = []
    for i in range(5):
      r, rt0 = i2t(img_embs[i*5000:(i + 1)*5000], txt_embs[i*5000:(i + 1)*5000],
                   nreps=nreps, return_ranks=True, order=args.order, use_gpu=args.eval_on_gpu)
      r = (r[0], r[1], r[2], r[3], r[3] / n_samples, r[4], r[4] / n_samples)
      print("Image to text: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % r)

      ri, rti0 = t2i(img_embs[i*5000:(i + 1)*5000], txt_embs[i*5000:(i + 1)*5000],
                     nreps=nreps, return_ranks=True, order=args.order, use_gpu=args.eval_on_gpu)
      if i == 0:
        rt, rti = rt0, rti0
      ri = (ri[0], ri[1], ri[2], ri[3], ri[3] / n_samples, ri[4], ri[4] / n_samples)
      print("Text to image: %.2f, %.2f, %.2f, %.2f (%.2f), %.2f (%.2f)" % ri)

      ar = (r[0] + r[1] + r[2]) / 3
      ari = (ri[0] + ri[1] + ri[2]) / 3
      rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
      print("rsum: %.2f ar: %.2f ari: %.2f" % (rsum, ar, ari))
      results += [list(r) + list(ri) + [ar, ari, rsum]]

    mean_metrics = tuple(np.array(results).mean(axis=0).flatten())

    print("-----------------------------------")
    print("Mean metrics from 5-fold evaluation: ")
    print("rsum: %.2f" % (mean_metrics[-1] * 6))
    print("Average i2t Recall: %.2f" % mean_metrics[-3])
    print("Image to text: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % mean_metrics[:7])
    print("Average t2i Recall: %.2f" % mean_metrics[-2])
    print("Text to image: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % mean_metrics[7:14])

  # no cross-validation, full evaluation
  r, rt = i2t(img_embs, txt_embs, nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
  ri, rti = t2i(img_embs, txt_embs, nreps=nreps, return_ranks=True, use_gpu=args.eval_on_gpu)
  ar = (r[0] + r[1] + r[2]) / 3
  ari = (ri[0] + ri[1] + ri[2]) / 3
  rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
  r = (r[0], r[1], r[2], r[3], r[3] / n_samples, r[4], r[4] / n_samples)
  ri = (ri[0], ri[1], ri[2], ri[3], ri[3] / n_samples, ri[4], ri[4] / n_samples)
  print("rsum: %.2f" % rsum)
  print("Average i2t Recall: %.2f" % ar)
  print("Image to text: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % r)
  print("Average t2i Recall: %.2f" % ari)
  print("Text to image: %.2f %.2f %.2f %.2f (%.2f) %.2f (%.2f)" % ri)

  return mean_metrics


if __name__ == '__main__':
  multi_gpu = torch.cuda.device_count() > 1

  args = verify_input_args(parser.parse_args())
  opt = verify_input_args(parser.parse_args())

  # load vocabulary used by the model
  with open('./vocab/%s_vocab.pkl' % args.data_name, 'rb') as f:
    vocab = pickle.load(f)
  args.vocab_size = len(vocab)

  

  # load model and options
  assert os.path.isfile(args.ckpt)
  model = PVSE(vocab.word2idx, args)
  if torch.cuda.is_available():
    model = torch.nn.DataParallel(model).cuda() if multi_gpu else model
    torch.backends.cudnn.benchmark = True
  model.load_state_dict(torch.load(args.ckpt))

  # evaluate
  metrics = evalrank(model, args, split='test')
