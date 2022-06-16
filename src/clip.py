import torch
import clip
from PIL import Image
import pdb
import random
import data_lp as data
import json
import numpy as np
import itertools
import argparse 
import os
##Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



def i2t(connect_json, all_img_ids, images, sentences, nreps=1, npts=None, return_ranks=False, order=False, use_gpu=False):
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

def lock_and_write_to_file(filename, text):
  # with LockFile(filename) as lock:
  with open(filename, 'a') as fid:
    fid.write('{}\n'.format(text))

def t2i(connect_json, all_img_ids, images, sentences, nreps=1, npts=None, return_ranks=False, order=False, use_gpu=False):
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
        sim = np.tensordot(queries, ims, axes=[2, 2]).max(axis=(1,3)) \
            if len(sentences.shape) == 3 else np.dot(queries, ims.T)

    inds = np.zeros(sim.shape)
    for i in range(len(inds)):
      if use_gpu:
        _, inds_gpu = sim[i].sort()
        inds[i] = inds_gpu.cpu().numpy().copy()[::-1]
      else:
        inds[i] = np.argsort(sim[i])[::-1]
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

##Load Eval



def CLIP_collate_fn(data, caption_lim = 500):
  """Build mini-batch tensors from a list of (image, sentence) tuples.
  Args:
    data: list of (image, sentence) tuple.
      - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
      - sentence: torch tensor of shape (?); variable length.

  Returns:
    images: torch tensor of shape (batch_size, 3, 256, 256) or
            (batch_size, padded_length, 3, 256, 256).
    targets: torch tensor of shape (batch_size, padded_length).
    lengths: list; valid length for each padded sentence.
  """
  # Sort a data list by sentence length
  # data.sort(key=lambda x: len(x[1]), reverse=True)
  # images, sentences, ids, img_ids = zip(*data)

  # # Merge images (convert tuple of 3D tensor to 4D tensor)
  # images = torch.stack(images, 0)

  # # Merge sentences (convert tuple of 1D tensor to 2D tensor)
  # cap_lengths = torch.tensor([len(cap) for cap in sentences])
  # targets = torch.zeros(len(sentences), max(cap_lengths)).long()
  # for i, cap in enumerate(sentences):
  #   end = cap_lengths[i]
  #   targets[i, :end] = cap[:end]

  # Sort a data list by sentence length
  data.sort(key=lambda x: len(x[1]), reverse=True)
  images, sentences, ids, img_ids = zip(*data)

  # Merge images (convert tuple of 3D tensor to 4D tensor)
  images = torch.stack(images, 0)

  # Merge sentences (convert tuple of 1D tensor to 2D tensor)
  cap_lengths = torch.tensor([len(cap) for cap in sentences])
  #targets = torch.zeros(len(sentences), max(cap_lengths)).long()
  targets = torch.zeros(len(sentences), 77).long()

  return images, targets, cap_lengths, ids, img_ids





def encode_data(model, data_loader, use_gpu=False):
  """Encode all images and sentences loadable by data_loader"""
  # switch to evaluate mode
  
  model.eval()

  # numpy array to keep all the embeddings
  img_embs, txt_embs = None, None
  all_img_ids = []
  for i, data in enumerate(data_loader):
    img, txt, txt_len, ids, img_ids = data
    if torch.cuda.is_available():
      img, txt, txt_len = img.cuda(), txt.cuda(), txt_len.cuda()

    # compute the embeddings
    img_emb = model.encode_image(img)
    txt_emb = model.encode_text(txt)
    del img, txt, txt_len

    # initialize the output embeddings
    if img_embs is None:
      if use_gpu:
        emb_sz = [len(data_loader.dataset), img_emb.size(1), img_emb.size(2)] \
                if use_mil else [len(data_loader.dataset), img_emb.size(1)]
        img_embs = torch.zeros(emb_sz, dtype=img_emb.dtype, requires_grad=False).cuda()
        txt_embs = torch.zeros(emb_sz, dtype=txt_emb.dtype, requires_grad=False).cuda()
      else:
        emb_sz = (len(data_loader.dataset), img_emb.size(1))
        img_embs = np.zeros(emb_sz)
        txt_embs = np.zeros(emb_sz)

    # preserve the embeddings by copying from gpu and converting to numpy
    img_embs[ids,...] = img_emb if use_gpu else img_emb.data.cpu().numpy().copy()
    txt_embs[ids,...] = txt_emb if use_gpu else txt_emb.data.cpu().numpy().copy()
    all_img_ids.append(img_ids)

  all_img_ids = list(itertools.chain(*all_img_ids))
  all_img_ids = [int(x) for x in all_img_ids]

  return img_embs, txt_embs, all_img_ids


##Calculate Embeddings and Recall Scores

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='args')
  parser.add_argument('--data_name', required = True)
  parser.add_argument('--wemb_type', required = True)
  parser.add_argument('--seed', required = True)
  parser.add_argument('--log_file', required = True)
  parser.add_argument('--human_study', action='store_true', help='Eval Loop')
  args = parser.parse_args()

  seed = str(args.seed)
  int_seed = int(seed)
  args.log_file = args.log_file.replace(args.data_name, os.path.join(seed,args.data_name))
  os.makedirs(os.path.dirname(args.log_file), exist_ok = True)

  print(os.path.dirname(args.log_file))

  # if args.ckpt:
  #   ckpt = torch.load(args.ckpt)
  #   args = copy_input_args_from_ckpt(args, ckpt['args'])
  print(args)

  torch.manual_seed(int_seed)
  random.seed(int_seed)
  np.random.seed(int_seed)



  ##Load Dataset

  sp = args.data_name
  root_dir = '/projects/dataset_processed/dongwonl/data/{}'.format(sp)

  with open("/projects/dataset_processed/dongwonl/data/{}/{}_figs.json".format(sp,sp), 'r') as f:
     fig_json = json.loads(f.read())

  with open("/projects/dataset_processed/dongwonl/data/{}/{}.json".format(sp,sp), 'r') as j:
     cap_json = json.loads(j.read())

  with open("/projects/dataset_processed/dongwonl/data/{}/{}_capfig.json".format(sp,sp), 'r') as c:
     connect_json = json.loads(c.read())


  l = list(connect_json.items())
  random.shuffle(l)
  connect_json = dict(l)
  test_connect_json = dict(list(connect_json.items())[(len(connect_json)//10)*8:])

  if args.human_study:
    test_connect_json_list = list(connect_json.items())[(len(connect_json)//10)*8:]
    values_count = 0 
    human_study_connect_json = {}
    #get 20 figures 
    for k,v in test_connect_json.items():
      human_study_connect_json[k] = v 
      values_count += len(v) 
      if values_count == 10:
        break 
      if values_count > 10:
        human_study_connect_json[k] = v[:len(v) - (values_count - 20)] 
        break
    
      
    print(len(list(human_study_connect_json.values())))
    human_study_connect_json = dict(human_study_connect_json)

    dataset = data.LPEvalDataset_CLIP(cap_json, fig_json, human_study_connect_json, clip, root_dir, args, transform = preprocess)
  else:
    dataset = data.LPEvalDataset_CLIP(cap_json, fig_json, test_connect_json, clip, root_dir, args, transform = preprocess)
  loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=16,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=1,
                                          collate_fn = CLIP_collate_fn
                                          )


  img_embs, txt_embs, all_img_ids = encode_data(model, loader, use_gpu=False)


  (r1, r5, r10, medr, meanr), (ranks, top1, top10) = i2t(test_connect_json, all_img_ids, img_embs, txt_embs, nreps=1, return_ranks=True, order=False, use_gpu=False)

  (r1i, r5i, r10i, medri, meanri), (ranksi, top1i, top10i) = t2i(test_connect_json, all_img_ids, img_embs, txt_embs, nreps=1, return_ranks=True, order=False, use_gpu=False)
  rsum = r1 + r5 + r10 + r1i + r5i + r10i
  med_rsum, mean_rsum = medr + medri, meanr + meanri

  log_str1 = "Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" \
              %(r1, r5, r10, medr, meanr)
  log_str2 = "Text to Image: %.2f, %.2f, %.2f, %.2f, %.2f" \
              %(r1i, r5i, r10i, medri, meanri)
  log_str3 = 'rsum: %.2f, med_rsum: %.2f, mean_rsum: %.2f' \
              %(rsum, med_rsum, mean_rsum)

  log_msg = '{}\n{}\n{}'.format(log_str1, log_str2, log_str3)

  # lock_and_write_to_file(args.log_file, log_msg)
  if not args.human_study:
    msg = "{}\n{}\n{}\n{}\n{}\n{}".format(args.data_name, all_img_ids, ranks.tolist(), top1.tolist(), ranksi.tolist(),  top1i.tolist())
    lock_and_write_to_file(args.log_file.replace(args.data_name, "ranks_top2"), msg)
  if args.human_study:
    msg = "{}\n{}\n{}".format(args.data_name, r1, r1i)
    lock_and_write_to_file(args.log_file.replace(args.data_name, "human_study"), msg)

  # sum of recalls to be used for early stopping
  rsum = r1 + r5 + r10 + r1i + r5i + r10i
  med_rsum, mean_rsum = medr + medri, meanr + meanri
  print(sp)
  print("i2t: recall@1 = {}".format(r1))
  print("i2t: recall@5 = {}".format(r5))
  print("i2t: recall@10 = {}".format(r10))

  print("t2i: recall@1 = {}".format(r1i))
  print("t2i: recall@5 = {}".format(r5i))
  print("t2i: recall@10 = {}".format(r10i))


  print("rsum: {}".format(rsum))

