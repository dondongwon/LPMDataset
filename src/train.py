import os
import sys
import math
import time
import shutil
import pickle5 as pickle
from lockfile import LockFile
import json
import pdb
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import numpy as np
import time
import random
from tqdm import tqdm

from vocab import Vocabulary
from eval import i2t, t2i, encode_data, encode_data_pcme
from logger import AverageMeter
from option import parser, verify_input_args
import data_lp as data


# import data_lp as data
from loss_lp import PVSELoss, MCSoftContrastiveLoss, OursLoss

import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


    


def lock_and_write_to_file(filename, text):
  # with LockFile(filename) as lock:
  with open(filename, 'a') as fid:
    fid.write('{}\n'.format(text))


def copy_input_args_from_ckpt(args, ckpt_args):
  args_to_copy = ['word_dim','crop_size','cnn_type','embed_size', 'num_embeds',
                  'img_attention','txt_attention','max_video_length']
  for arg in args_to_copy:
    val1, val2 = getattr(args, arg), getattr(ckpt_args, arg)
    if val1 != val2:
      logging.warning('Updating argument from checkpoint [{}]: [{}] --> [{}]'.format(arg, val1, val2))
      setattr(args, arg, val2)
  return args

def save_ckpt(state, is_best, filename='ckpt.pth.tar', prefix=''):
  torch.save(state, prefix + filename)
  if is_best:
    shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    logging.info('Updating the best model checkpoint: {}'.format(prefix + 'model_best.pth.tar'))


def get_description(args, epoch=-1):
  return ('[{}][epoch:{}] {}'.format(args.logger_name.split('/')[-1], epoch, args))


def train(epoch, data_loader, model, criterion, optimizer, args):
  # switch to train mode
  model.train()

  # average meters to record the training statistics
  losses = AverageMeter()
  losses_dict = dict()
  losses_dict['ranking_loss'] = AverageMeter()
  if args.div_weight > 0:
    losses_dict['div_loss'] = AverageMeter()
  if args.mmd_weight > 0:
    losses_dict['mmd_loss'] = AverageMeter()

  for itr, data in enumerate(tqdm(data_loader)):
    
    if 'vilt' in args.wemb_type:
      fig_ocr, spoken_output, pointer_target, cap_lengths, index, img_ids = data
      if torch.cuda.is_available():
        for k,v in fig_ocr.items():
          fig_ocr[k] = v.cuda()
        spoken_output, pointer_target, cap_lengths =  spoken_output.cuda(), pointer_target.cuda(), cap_lengths.cuda()
    else:
      images, spoken_output, ocr_target, cap_lengths, index, img_ids = data
      if torch.cuda.is_available():
        
        images, spoken_output, cap_lengths = images.cuda(), spoken_output.cuda(), cap_lengths.cuda()
    # Forward pass and compute loss; _a: attention map, _r: residuals
    if args.model == 'PVSE':
      img_emb, txt_emb, img_a, txt_a, img_r, txt_r = model.forward(images, spoken_output, cap_lengths)
      loss, loss_dict = criterion(img_emb, txt_emb, img_r, txt_r)
    if args.model == 'Ours_VILT':
      img_emb, txt_emb, txt_a, txt_r = model.forward(fig_ocr, spoken_output, pointer_target, cap_lengths)
      loss, loss_dict = criterion(img_emb, txt_emb, txt_r)
    if args.model == 'Ours_VILT_Trace':
      img_emb, txt_emb, txt_a, txt_r = model.forward(fig_ocr, spoken_output, pointer_target, cap_lengths)
      loss, loss_dict = criterion(img_emb, txt_emb, txt_r)
      
    # Compute loss and update statstics
   
    losses.update(loss.item())
    for key, val in loss_dict.items():
      losses_dict[key].update(val.item())

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    if args.grad_clip > 0:
      nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    # Print log info
    if itr > 0 and (itr % args.log_step == 0 or itr + 1 == len(data_loader)):
      log_msg = 'loss: %.4f (%.4f)' %(losses.val, losses.avg)
      for key, val in losses_dict.items():
        log_msg += ', %s: %.4f, (%.4f)' %(key.replace('_loss',''), val.val, val.avg)
      n = int(math.ceil(math.log(len(data_loader) + 1, 10)))
      logging.info('[%d][%*d/%d] %s' %(epoch, n, itr, len(data_loader), log_msg))

  log_msg = 'loss: %.4f' %(losses.avg)
  for key, val in losses_dict.items():
    log_msg += ', %s: %.4f' %(key.replace('_loss',''), val.avg)
  exp_name = args.logger_name.split('/')[-1]

  lock_and_write_to_file(args.log_file, '[%s][%d] %s' %(exp_name, epoch, log_msg))

  del img_emb, txt_emb, txt_a, txt_r, loss
  return losses.avg

def train_PCME(epoch, data_loader, model, criterion, optimizer, args):
  # switch to train mode
  model.train()

  losses = AverageMeter()
  losses_dict = dict()
  losses_dict['i2t_loss'] = AverageMeter()
  losses_dict['t2i_loss'] = AverageMeter()
  losses_dict['i2t_pos_loss'] = AverageMeter()
  losses_dict['i2t_neg_loss'] = AverageMeter()
  losses_dict['t2i_pos_loss'] = AverageMeter()
  losses_dict['t2i_neg_loss'] = AverageMeter()
  losses_dict['uniform_loss'] = AverageMeter()
  losses_dict['vib_loss'] = AverageMeter()
  losses_dict['loss'] = AverageMeter()

  for itr, data in enumerate(tqdm(data_loader)):

    images, spoken_output, ocr_target, cap_lengths, index, img_ids = data
    if torch.cuda.is_available():
      images, spoken_output, cap_lengths = images.cuda(), spoken_output.cuda(), cap_lengths.cuda()
    # Forward pass and compute loss; _a: attention map, _r: residuals
    output = model.forward(images, spoken_output, cap_lengths)

    loss, loss_dict = criterion(output["image_features"], output["caption_features"], output["image_logsigma"], output["caption_logsigma"])

    
    losses.update(loss.item())
    for key, val in loss_dict.items():
      if key in ['shift', 'negative_scale']:
        continue
      losses_dict[key].update(val)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    if args.grad_clip > 0:
      nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    # Print log info
    if itr > 0 and (itr % args.log_step == 0 or itr + 1 == len(data_loader)):
      log_msg = 'loss: %.4f (%.4f)' %(losses.val, losses.avg)
      for key, val in losses_dict.items():
        log_msg += ', %s: %.4f, (%.4f)' %(key.replace('_loss',''), val.val, val.avg)
      n = int(math.ceil(math.log(len(data_loader) + 1, 10)))
      logging.info('[%d][%*d/%d] %s' %(epoch, n, itr, len(data_loader), log_msg))

  log_msg = 'loss: %.4f' %(losses.avg)
  for key, val in losses_dict.items():
    log_msg += ', %s: %.4f' %(key.replace('_loss',''), val.avg)
  exp_name = args.logger_name.split('/')[-1]
  lock_and_write_to_file(args.log_file, '[%s][%d] %s' %(exp_name, epoch, log_msg))

  #del img_emb, txt_emb, img_a, txt_a, img_r, txt_r, loss
  return losses.avg


def validate(test_connect_json, data_loader, model, args, epoch=-1, best_score=None, evaluation = False, human_study = False, all_speakers = False):
  # switch to eval mode
  model.eval()

  nreps = 5 if 'coco' in args.data_name else 1
  order = args.order if hasattr(args, 'order') and args.order else False

  if args.model == 'PVSE':
   
    img_embs, txt_embs, all_img_ids = encode_data(model, data_loader, args.eval_on_gpu)
  if args.model == 'PCME':
    img_embs, txt_embs, all_img_ids = encode_data_pcme(model, data_loader, args.eval_on_gpu)
  
  if args.model == 'VILT':
    img_embs, txt_embs, all_img_ids = encode_data(model, data_loader, args.eval_on_gpu,  ours = True)
  
  if 'Ours' in args.model:
    img_embs, txt_embs, all_img_ids = encode_data(model, data_loader, args.eval_on_gpu,  ours = True)

  


  # for validation
  if args.model == 'Random':
    img_embs, txt_embs, all_img_ids = encode_data(model, data_loader, args.eval_on_gpu)
    (r1, r5, r10, medr, meanr), (ranks, top1, top10) = i2t(test_connect_json, all_img_ids, img_embs, txt_embs, 
        nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu, random = True)

    (r1i, r5i, r10i, medri, meanri), (ranksi, top1i, top10i) = t2i(test_connect_json, all_img_ids, img_embs, txt_embs,
        nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu, random = True)

  else:
    (r1, r5, r10, medr, meanr), (ranks, top1, top10) = i2t(test_connect_json, all_img_ids, img_embs, txt_embs, 
        nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu, sp = args.data_name)

    (r1i, r5i, r10i, medri, meanri), (ranksi, top1i, top10i) = t2i(test_connect_json, all_img_ids, img_embs, txt_embs,
        nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu)

  # sum of recalls to be used for early stopping
  rsum = r1 + r5 + r10 + r1i + r5i + r10i
  med_rsum, mean_rsum = medr + medri, meanr + meanri

  # log
  exp_name = args.logger_name.split('/')[-1]
  vname = 'Video' if args.max_video_length>1 else 'Image'

  log_str1 = "[%s][%d] %s to text: %.2f, %.2f, %.2f, %.2f, %.2f" \
              %(exp_name, epoch, vname, r1, r5, r10, medr, meanr)
  log_str2 = "[%s][%d] Text to %s: %.2f, %.2f, %.2f, %.2f, %.2f" \
              %(exp_name, epoch, vname, r1i, r5i, r10i, medri, meanri)
  log_str3 = '[%s][%d] rsum: %.2f, med_rsum: %.2f, mean_rsum: %.2f' \
              %(exp_name, epoch, rsum, med_rsum, mean_rsum)
  if best_score:
    log_str3 += ' (best %s: %.2f)' %(args.val_metric, best_score)

  logging.info(log_str1)
  logging.info(log_str2)
  logging.info(log_str3)

  dscr = get_description(args, epoch)
  log_msg = '{}\n{}\n{}'.format(log_str1, log_str2, log_str3)

  if not evaluation: 
    lock_and_write_to_file(args.log_file, log_msg)
  if evaluation:
#     msg = "{}\n{}\n{}\n{}".format(args.data_name, all_img_ids, ranks.tolist(), ranksi.tolist())
#     lock_and_write_to_file(args.log_file.replace(args.data_name, "ranks"), msg)
    if not human_study:
      msg = "{}\n{}\n{}\n{}\n{}\n{}".format(args.data_name, all_img_ids, ranks.tolist(), top1.tolist(), ranksi.tolist(),  top1i.tolist())
      lock_and_write_to_file(args.log_file.replace(args.data_name, "ranks_top2"), msg)
    
    if human_study:
      msg = "{}\n{}\n{}".format(args.data_name, r1, r1i)
      lock_and_write_to_file(args.log_file.replace(args.data_name, "human_study"), msg)


  if args.val_metric == 'rsum':
    return rsum
  elif args.val_metric == 'med_rsum':
    return med_rsum
  else:
    return mean_rsum



def validate_all_speakers(test_connect_json, data_loader, model, args, epoch=-1, best_score=None, evaluation = False, human_study = False):
  # switch to eval mode
  model.eval()

  nreps = 5 if 'coco' in args.data_name else 1
  order = args.order if hasattr(args, 'order') and args.order else False

  if args.model == 'PVSE':
   
    img_embs, txt_embs, all_img_ids = encode_data(model, data_loader, args.eval_on_gpu)
  if args.model == 'PCME':
    img_embs, txt_embs, all_img_ids = encode_data_pcme(model, data_loader, args.eval_on_gpu)
  
  if args.model == 'VILT':
    img_embs, txt_embs, all_img_ids = encode_data(model, data_loader, args.eval_on_gpu,  ours = True)
  
  if 'Ours' in args.model:
    img_embs, txt_embs, all_img_ids = encode_data(model, data_loader, args.eval_on_gpu,  ours = True)
  


  # for validation
  if args.model == 'Random':
    img_embs, txt_embs, all_img_ids = encode_data(model, data_loader, args.eval_on_gpu)
    (r1, r5, r10, medr, meanr), (ranks, top1, top10) = i2t(test_connect_json, all_img_ids, img_embs, txt_embs, 
        nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu, random = True)

    (r1i, r5i, r10i, medri, meanri), (ranksi, top1i, top10i) = t2i(test_connect_json, all_img_ids, img_embs, txt_embs,
        nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu, random = True)

  else:
    (r1, r5, r10, medr, meanr), (ranks, top1, top10) = i2t(test_connect_json, all_img_ids, img_embs, txt_embs, 
        nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu)

    (r1i, r5i, r10i, medri, meanri), (ranksi, top1i, top10i) = t2i(test_connect_json, all_img_ids, img_embs, txt_embs,
        nreps=nreps, return_ranks=True, order=order, use_gpu=args.eval_on_gpu)

  # sum of recalls to be used for early stopping
  rsum = r1 + r5 + r10 + r1i + r5i + r10i
  med_rsum, mean_rsum = medr + medri, meanr + meanri


  return rsum, r1, r5, r10, r1i, r5i, r10i

def update_best_score(new_score, old_score, is_higher_better):
  if not old_score:
    score, updated = new_score, True
  else:
    if is_higher_better:
      score = max(new_score, old_score)
      updated = new_score > old_score
    else:
      score = min(new_score, old_score)
      updated = new_score < old_score
  return score, updated

def itr_merge(*itrs):
    for itr in itrs:
        for v in itr:
            yield v


if __name__ == '__main__':
  multi_gpu = torch.cuda.device_count() > 1

  args = verify_input_args(parser.parse_args())

  seed = str(args.seed)
  
  args.log_file = args.log_file.replace(args.data_name, os.path.join(seed,args.data_name))
  args.logger_name = args.logger_name.replace(args.data_name, os.path.join(seed,args.data_name))
  print(args.logger_name)
  os.makedirs(os.path.dirname(args.log_file), exist_ok = True)
  os.makedirs(args.logger_name, exist_ok = True)

  print(os.path.dirname(args.log_file))
  print(os.path.dirname(args.logger_name))

  # if args.ckpt:
  #   ckpt = torch.load(args.ckpt)
  #   args = copy_input_args_from_ckpt(args, ckpt['args'])
  print(args)
    

  torch.manual_seed(args.seed)
  random.seed(args.seed)
  np.random.seed(args.seed)





  if args.data_name == "all":

    from model_ours import Ours_VILT
    model = Ours_VILT(args)


    if torch.cuda.is_available():
      model = nn.DataParallel(model).cuda() #model.cuda() 
      cudnn.benchmark = True


    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-10, verbose=True)

    # Train the Model
    if args.ckpt and 'best_score' in ckpt and ckpt['args'].val_metric == args.val_metric:
      best_score = ckpt['best_score']
    else:
      best_score = None


    trn_datasets = []
    val_datasets = []
    val_jsons = []
    speakers = ["anat-1", "bio-1", "bio-3", "dental", "ml-1", "psy-2", "anat-2", "bio-4", "psy-1", "speaking"]

    for speaker in speakers:
      sp = speaker

      vocab_path = os.path.join(args.vocab_path, '%s_vocab.pkl' % sp)
      vocab = pickle.load(open(vocab_path, 'rb'))


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
      train_connect_json = dict(list(connect_json.items())[:(len(connect_json)//10)*8])

      val_dataset, val_collate_fn = data.get_loaders(cap_json, fig_json, test_connect_json, vocab, root_dir, args.batch_size, args.wemb_type,  args.model, 'dataset')
      trn_dataset, trn_collate_fn = data.get_loaders(cap_json, fig_json, train_connect_json, vocab, root_dir, args.batch_size, args.wemb_type, args.model, 'dataset')

      val_jsons.append(test_connect_json)
      val_datasets.append(val_dataset)
      trn_datasets.append(trn_dataset)

    trn_datasets = torch.utils.data.ConcatDataset(trn_datasets)



    trn_loader = torch.utils.data.DataLoader(dataset=trn_datasets,
                                        batch_size=8,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=4,
                                        collate_fn = trn_collate_fn
                                        )

    



     
    for epoch in range(args.num_epochs):

      criterion = OursLoss(args)
      loss = train(epoch, trn_loader, model, criterion, optimizer, args)


      rsum_all, r1_all, r5_all, r10_all, r1i_all, r5i_all, r10i_all = 0,0,0,0,0,0,0

      for index, speaker in enumerate(speakers):
        test_connect_json = val_jsons[index]
        val_loader = torch.utils.data.DataLoader(dataset=val_datasets[index],
                                            batch_size=8,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4,
                                            collate_fn = val_collate_fn
                                            )
        rsum, r1, r5, r10, r1i, r5i, r10i  = validate_all_speakers(test_connect_json, val_loader, model, args, epoch, best_score)

        rsum_all += rsum
        r1_all += r1
        r5_all += r5
        r10_all += r10
        r1i_all += r1i
        r5i_all += r5i
        r10i_all += r10i
      
      rsum_all /= rsum
      r1_all /= 10
      r5_all /= 10
      r10_all /= 10
      r1i_all /= 10
      r5i_all /= 10
      r10i_all /= 10


        # log
      exp_name = args.logger_name.split('/')[-1]
      vname = 'Video' if args.max_video_length>1 else 'Image'

      log_str1 = "[%s][%d] %s to text: %.2f, %.2f, %.2f" \
                  %(exp_name, epoch, vname, r1_all, r5_all, r10_all)
      log_str2 = "[%s][%d] Text to %s: %.2f, %.2f, %.2f" \
                  %(exp_name, epoch, vname, r1i_all, r5i_all, r10i_all)
      log_str3 = '[%s][%d] rsum: %.2f' \
                  %(exp_name, epoch, rsum_all)
      if best_score:
        log_str3 += ' (best %s: %.2f)' %(args.val_metric, best_score)

      logging.info(log_str1)
      logging.info(log_str2)
      logging.info(log_str3)

      dscr = get_description(args, epoch)
      log_msg = '{}\n{}\n{}'.format(log_str1, log_str2, log_str3)

      lock_and_write_to_file(args.log_file, log_msg)

      val_score = rsum_all
      

      # adjust learning rate if rsum stagnates
      lr_scheduler.step(val_score)

      # remember best rsum and save ckpt
      best_score, updated = update_best_score(val_score, best_score,
                                              args.val_metric=='rsum')
      save_ckpt({
        'args': args,
        'epoch': epoch,
        'best_score': best_score,
        'model': model.state_dict(),
      }, updated, prefix=args.logger_name + '/')

    


  else:
    vocab_path = os.path.join(args.vocab_path, '%s_vocab.pkl' % args.data_name)
    vocab = pickle.load(open(vocab_path, 'rb'))


    if args.model == 'PCME':
      from model_pcme import *
      model = PCME(vocab.word2idx, args)
    if args.model == 'PVSE':
      from model_lp import PVSE
      model = PVSE(vocab.word2idx, args)
    if args.model == 'Ours_VILT':
      from model_ours import Ours_VILT
      model = Ours_VILT(args)
    if args.model == 'Ours_VILT_Trace':
      from model_ours import Ours_VILT_Trace
      model = Ours_VILT_Trace(args)
    if args.model == 'VILT':
      from model_ours import VILT
      model = VILT(args)
    if args.model == 'Random': 
      from model_lp import PVSE
      model = PVSE(vocab.word2idx, args)
    
    print('Model {} Loaded'.format(args.model))

    # Load Vocabulary Wrapper


    #load data

    sp = args.data_name
    root_dir = '/projects/dataset_processed/dongwonl/data/{}'.format(sp)

    with open("/projects/dataset_processed/dongwonl/data/{}/{}_figs.json".format(sp,sp), 'r') as f:
      fig_json = json.loads(f.read())

    with open("/projects/dataset_processed/dongwonl/data/{}/{}.json".format(sp,sp), 'r') as j:
      cap_json = json.loads(j.read())

    with open("/projects/dataset_processed/dongwonl/data/{}/{}_capfig.json".format(sp,sp), 'r') as c:
      connect_json = json.loads(c.read())


    print('Json Loaded')


    #shuffle this

    
    if torch.cuda.is_available():
      model = model.cuda() #model.cuda() 
      cudnn.benchmark = True
    


    if args.ckpt:
      target_vocab_path = './vocab/%s_vocab.pkl' % args.data_name
      src_vocab_path = './vocab/%s_vocab.pkl' % ckpt['args'].data_name
      if target_vocab_path != src_vocab_path:
        print('Vocab mismatch!')
        sys.exit(-1)
      model.load_state_dict(ckpt['model'])
      #validate(val_loader, model, args)

    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, min_lr=1e-10, verbose=True)

    # Train the Model
    if args.ckpt and 'best_score' in ckpt and ckpt['args'].val_metric == args.val_metric:
      best_score = ckpt['best_score']
    else:
      best_score = None


    l = list(connect_json.items())
    random.shuffle(l)
    connect_json = dict(l)
    test_connect_json = dict(list(connect_json.items())[(len(connect_json)//10)*8:])
    train_connect_json = dict(list(connect_json.items())[:(len(connect_json)//10)*8])

    # if args.k_analysis:
    #   #get a pair of same samples
    #   new_test_connect_json = {}
    #   for k,v in test_connect_json.items():
        
    #     if len(v) == 3:
    #       new_test_connect_json[k] = v
    #       print(new_test_connect_json)
    #   test_connect_json = new_test_connect_json

      # data_dir = "./k_analysis/{}.json".format(args.data_name)
      # with open(data_dir, 'w') as fp:
      #   json.dump(test_connect_json, fp)



    val_loader = data.get_loaders(cap_json, fig_json, test_connect_json, vocab, root_dir, args.batch_size, args.wemb_type,  args.model, 'eval')

    if args.eval:
      ckpt = torch.load(os.path.join(args.logger_name, "model_best.pth.tar"))
      epoch = ckpt['epoch']
      best_score = ckpt['best_score']
      model.load_state_dict(ckpt['model'])
      
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
          
      
        human_study_connect_json = dict(human_study_connect_json)

        data_dir = "./human_study/{}".format(args.seed)
        
        os.makedirs(data_dir, exist_ok = True)
        dict_path = '{}/{}.json'.format(data_dir, args.data_name)
        
        if not os.path.exists(dict_path):
          with open(dict_path, 'w') as fp:
            json.dump(human_study_connect_json, fp)
      
        val_loader = data.get_loaders(cap_json, fig_json, human_study_connect_json, vocab, root_dir, args.batch_size, args.wemb_type,  args.model, 'eval')
        val_score = validate(human_study_connect_json, val_loader, model, args, epoch, best_score, evaluation = True, human_study = True)
      
      val_score = validate(test_connect_json, val_loader, model, args, epoch, best_score, evaluation = True, human_study = False)

    else: 
      for epoch in range(args.num_epochs):
        
        no_rep_connect_json = {}
        for k,v in train_connect_json.items():
          if len(v) > 0:
            no_rep_connect_json[k] = [random.choice(v)]

      # Dataloaders
        #trn_loader = data.get_loaders(cap_json, fig_json, train_connect_json, vocab, 'train')

        if args.slide_level:
          trn_loader = data.get_loaders(cap_json, fig_json, no_rep_connect_json, vocab, root_dir, args.batch_size,  args.wemb_type, args.model, 'slide_train')
        else:
          trn_loader = data.get_loaders(cap_json, fig_json, train_connect_json, vocab, root_dir, args.batch_size, args.wemb_type, args.model, 'train')
      
        # train for one epoch
        # val_score = validate(test_connect_json, val_loader, model, args, epoch, best_score)
        if args.model == 'PCME':
          criterion = MCSoftContrastiveLoss(args)
          loss = train_PCME(epoch, trn_loader, model, criterion, optimizer, args)
          val_score = validate(test_connect_json, val_loader, model, args, epoch, best_score)
        
        if args.model == 'PVSE':
          criterion = PVSELoss(args)
          loss = train(epoch, trn_loader, model, criterion, optimizer, args)
          val_score = validate(test_connect_json, val_loader, model, args, epoch, best_score)
        
        if args.model == 'Ours_VILT':
          criterion = OursLoss(args)
          loss = train(epoch, trn_loader, model, criterion, optimizer, args)
          val_score = validate(test_connect_json, val_loader, model, args, epoch, best_score)

        if args.model == 'Ours_VILT_Trace':
          criterion = OursLoss(args)
          loss = train(epoch, trn_loader, model, criterion, optimizer, args)
          val_score = validate(test_connect_json, val_loader, model, args, epoch, best_score)
      
        if args.model == 'VILT':
          criterion = OursLoss(args)
          val_score = validate(test_connect_json, val_loader, model, args, epoch, best_score)

        if args.model == 'Random':
          criterion = PVSELoss(args)
          val_score = validate(test_connect_json, val_loader, model, args, epoch, best_score)
          break
        

        # adjust learning rate if rsum stagnates
        lr_scheduler.step(val_score)

        # remember best rsum and save ckpt
        best_score, updated = update_best_score(val_score, best_score,
                                                args.val_metric=='rsum')
        save_ckpt({
          'args': args,
          'epoch': epoch,
          'best_score': best_score,
          'model': model.state_dict(),
        }, updated, prefix=args.logger_name + '/')
