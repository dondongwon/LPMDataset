import os
import sys
import math
import random
import glob
import pdb
import scipy
import numpy as np
import json as json
import pickle5 as pickle
import ast
import video_transforms
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from vocab import Vocabulary

from transformers import AutoTokenizer, BertModel, AutoModelForQuestionAnswering, BertTokenizerFast, ViltProcessor, ViltModel

from PIL import Image
#from gulpio import GulpDirectory
from operator import itemgetter

import io, threading
_lock = threading.Lock()


class LPDataset(data.Dataset):

  def __init__(self, cap_json, fig_json, connect_json, vocab, rootdir, wemb_type, transform=None, ids=None):
    """
    Args:
      json: full_dataset.
      vocab: vocabulary wrapper.
      transform: transformer for image.
    """

    # if ids provided by get_paths, use split-specific ids
    self.ids = [item for sublist in list(connect_json.values()) for item in sublist] # all ids
    self.vocab = vocab
    self.transform = transform
    self.cap_json = cap_json
    self.fig_json = fig_json
    self.connect_json = connect_json
    self.root = rootdir

    self.wemb_type = wemb_type

    if 'bert' in self.wemb_type:
      self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    if 'vilt' in self.wemb_type:
      self.vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    
  def __len__(self):
    return len(self.ids)


  def __getitem__(self, index):
    vocab = self.vocab
    sent, img_id, path, image, ocr, pointers = self.get_raw_item(index)
    ocr_text = " ".join([o['text'] for o in ocr])

    #sent = ast.literal_eval(sent)
    if self.transform is not None:
      try: 
        image = self.transform(image)
      except Exception:
        print(image)
        pdb.post_mortem()

    if self.wemb_type == 'glove' or self.wemb_type == 'fasttext' :
      tokens = sent
      sentence = []
      sentence.append(vocab('<start>'))
      sentence.extend([vocab(token.replace(" ", "")) for token in tokens])
      sentence.append(vocab('<end>'))
      spoken_target = torch.Tensor(sentence)

      ocr_tokens = ocr_text
      ocr_sentence= []
      ocr_sentence.extend([vocab(token.replace(" ", "")) for token in tokens])
      ocr_target = torch.Tensor(ocr_sentence)

      
    if 'bert' in self.wemb_type:
      spoken_text = self.bert_tokenizer(sent,max_length=512, padding='max_length',truncation=True, return_tensors='pt')
      spoken_target = spoken_text['input_ids'].squeeze()
      ocr_target = self.bert_tokenizer(ocr_text,max_length=40, padding='max_length',truncation=True, add_special_tokens = False, return_tensors='pt')['input_ids']
      #pointers, same size as language
    
    # elif 'only-vilt' in self.wemb_type:
    #   fig_ocr = self.vilt_processor(image, sent, max_length=40, padding='max_length',truncation=True, return_tensors="pt")
    #   spoken_target = torch.zeros(1)
    #   pointer_target =  torch.zeros(1)
    #   return fig_ocr, spoken_target, pointer_target, index, img_id

    if 'vilt' in self.wemb_type:
      fig_ocr = self.vilt_processor(image, ocr_text, max_length=40, padding='max_length',truncation=True, return_tensors="pt")
      
      pointer_target =  torch.zeros_like(spoken_target)
      for point in pointers:
        inds = torch.Tensor([i for i, x in enumerate(spoken_text.word_ids()) if x == point]).long()
        pointer_target[inds] = 1 
      return fig_ocr, spoken_target, pointer_target, index, img_id
      
    
    return  image, spoken_target, ocr_target, index, img_id

  def get_raw_item(self, index):

    img_id = self.ids[index]

    words = [] 
    captions = self.fig_json[img_id]['captions']
    for word in captions:
      words.append(str(word['Word']))

    # try:
    sentence = " ".join(words)
    # except Exception:
    #   print(words[265])
    #   print(words)

    scene_id = self.fig_json[img_id]['scene_id']
    
    if scene_id[0] == os.path.basename(self.root):
      _ = scene_id.pop(0)
    path = os.path.join(self.root, "/".join(scene_id)+".jpg")
    
    image = Image.open(os.path.join(self.root, path)).convert('RGB')

    #crop image here
    cr = self.fig_json[img_id]['slide_figure']
    x = cr["left"]
    y = cr["top"]
    w = cr["width"]
    h = cr["height"]

    # print(cr)
    image = image.crop((x,y,x + w, y + h))
    ocr =  self.fig_json[img_id]['slide_text']
    pointers = self.fig_json[img_id]['pointers']
    
    return sentence, img_id, path, image, ocr, pointers



class LPDataset_SlideLevel(LPDataset):

  def __init__(self, cap_json, fig_json, connect_json, vocab, rootdir, transform=None, ids=None):
    """
    Args:
      json: full_dataset.
      vocab: vocabulary wrapper.
      transform: transformer for image.
    """

    # if ids provided by get_paths, use split-specific ids
    self.ids = list(connect_json.values()) #[item for sublist in list(connect_json.values()) for item in sublist] # all ids
    self.vocab = vocab
    self.transform = transform
    self.cap_json = cap_json
    self.fig_json = fig_json
    self.connect_json = connect_json
    self.root = rootdir

  def get_raw_item(self, index):

    slide_id = self.ids[index]

    words = [] 
    captions = self.cap_json[slide_id]['captions']
    for word in captions:
      words.append(word['Word'])
    sentence = " ".join(words)

    scene_id = self.cap_json[slide_id]['scene_id']
    
    if scene_id[0] == os.path.basename(self.root):
      _ = scene_id.pop(0)
    path = os.path.join(self.root, "/".join(scene_id)+".jpg")
    
    image = Image.open(os.path.join(self.root, path)).convert('RGB')

    #crop image here
    cr = self.fig_json[slide_id]['slide_figure']
    x = cr["left"]
    y = cr["top"]
    w = cr["width"]
    h = cr["height"]
    ocr =  self.fig_json[slide_id]['slide_text']
    pointers = []
    
    return sentence, img_id, path, image, ocr, pointers



class LPEvalDataset(LPDataset):

  def __init__(self, cap_json, fig_json, connect_json, vocab,rootdir, transform=None, ids=None):
    """
    Args:
      json: full_dataset.
      vocab: vocabulary wrapper.
      transform: transformer for image.
    """

    # if ids provided by get_paths, use split-specific ids
    self.ids = [item for sublist in list(connect_json.values()) for item in sublist] # all ids
    self.vocab = vocab
    self.transform = transform
    self.cap_json = cap_json
    self.fig_json = fig_json
    self.connect_json = connect_json
    self.root = rootdir


  def __len__(self):
    return len(self.ids)


  def __getitem__(self, index):
    vocab = self.vocab
    sent, img_id, path, image = self.get_raw_item(index)
    #sent = ast.literal_eval(sent)
    if self.transform is not None:
      image = self.transform(image)

    tokens = sent
    sentence = []
    sentence.append(vocab('<start>'))
    sentence.extend([vocab(token.replace(" ", "")) for token in tokens])
    sentence.append(vocab('<end>'))
    target = torch.Tensor(sentence)

    
    return image, target, index, img_id

def get_image_transform():
  normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  t_list = []
  # if split_name == 'train':
  #   t_list = [transforms.RandomResizedCrop(opt.crop_size)]
  #   if not (data_name == 'mrw' or data_name == 'tgif'):
  #     t_list += [transforms.RandomHorizontalFlip()]
  # elif split_name == 'val':
  #   t_list = [transforms.Resize(256), transforms.CenterCrop(opt.crop_size)]
  # elif split_name == 'test':
  #   t_list = [transforms.Resize(256), transforms.CenterCrop(opt.crop_size)]

  t_list = [transforms.Resize(256), transforms.CenterCrop(224)]
  t_end = [transforms.ToTensor(), normalizer]
  transform = transforms.Compose(t_list + t_end)
  return transform


class LPEvalDataset_CLIP(LPDataset):

  def __init__(self, cap_json, fig_json, connect_json, clip_t, rootdir, wemb_type, transform=None, ids=None):
    """
    Args:
      json: full_dataset.
      vocab: vocabulary wrapper.
      transform: transformer for image.
    """

    # if ids provided by get_paths, use split-specific ids
    self.vocab = clip_t
    self.ids = [item for sublist in list(connect_json.values()) for item in sublist] # all ids
    self.transform = transform
    self.cap_json = cap_json
    self.fig_json = fig_json
    self.connect_json = connect_json
    self.root = rootdir

  def __getitem__(self, index):
    vocab = self.vocab
    sent, img_id, path, image, ocr, pointers = self.get_raw_item(index)
    
    #sent = ast.literal_eval(sent)
   
    if self.transform is not None:
      image = self.transform(image)

    tokens = sent
    
    sentence = [sent]
    target = self.vocab.tokenize(sentence, truncate = True)
    return image, target, index, img_id


def get_loaders(cap_json, fig_json, connect_json, vocab, root_dir, bs, wemb_type, model_type, split = 'train'):
  num_w = 4
  bs = bs
  transform = get_image_transform()

  # def collate_fn(data, caption_lim = 512):
  # # Sort a data list by sentence length
  #   data.sort(key=lambda x: len(x[1]), reverse=True)
  #   images, spoken_target, ocr_target, index, img_ids = zip(*data)
  #   # Merge images (convert tuple of 3D tensor to 4D tensor)
  #   images = torch.stack(images, 0)

  #   print(spoken_target)
  #   cap_lengths = torch.tensor([len(cap) for cap in spoken_target])
  #   print(cap_lengths)
    

  #   spoken_output = torch.zeros(len(spoken_target), caption_lim).long()

  #   for i, cap in enumerate(spoken_target):
  #     end = cap_lengths[i]
  #     if end <= caption_lim:
  #       spoken_output[i, :end] = cap[:end]
  #     else:
  #       print("GLOVE PCME SHOULD BE HERE")
  #       cap_lengths[i] = caption_lim
  #       spoken_output[i, :end] = cap[:caption_lim]
  #       print(spoken_target)
  #       print(i)
  #       print(cap_lengths)
  #     return images, spoken_output, ocr_target, cap_lengths, index, img_ids


  def collate_fn_og(data, caption_lim = 512):
  # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, spoken_target, ocr_target, index, img_ids = zip(*data)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    cap_lengths = torch.tensor([len(cap) if len(cap) <= caption_lim else caption_lim for cap in spoken_target])
    spoken_output = torch.zeros(len(spoken_target), caption_lim).long()
    

    for i, cap in enumerate(spoken_target):
      end = cap_lengths[i]
      if end <= caption_lim:
        spoken_output[i, :end] = cap[:end]
      else:
        cap_lengths[i] = caption_lim
        spoken_output[i, :end] = cap[:caption_lim]
      return images, spoken_output, ocr_target, cap_lengths, index, img_ids

  def collate_fn_vilt(data, caption_lim = 512):
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    fig_ocr, spoken_target, pointer_target, index, img_id = zip(*data)
    spoken_output = torch.stack(spoken_target)
    pointer_target = torch.stack(pointer_target)
    cap_lengths = (spoken_output != 0).sum(1)
    #list of dict to dict of lists: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    fig_ocr= {k: torch.stack([dic[k] for dic in fig_ocr]).squeeze() for k in fig_ocr[0]}

      
    return  fig_ocr, spoken_output, pointer_target, cap_lengths, index, img_id

  
  if 'PCME' in model_type:
    collate_fn = collate_fn_og
  elif 'PVSE' in model_type:
    collate_fn = collate_fn_og
  elif 'Random' in model_type:
    collate_fn = collate_fn_og
  elif 'VILT' in model_type:
    collate_fn = collate_fn_vilt
  

  if split == 'train':
    dataset = LPDataset(cap_json, fig_json, connect_json, vocab, root_dir, wemb_type, transform)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=bs,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=num_w,
                                            collate_fn = collate_fn
                                            )
  elif split == 'slide_train':
    dataset = LPDataset_SlideLevel(cap_json, fig_json, connect_json, vocab, root_dir, wemb_type, transform)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=bs,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=num_w,
                                            collate_fn = collate_fn
                                            )

  elif split == 'CLIP_eval':
    dataset = LPEvalDataset_CLIP(cap_json, fig_json, connect_json, vocab, root_dir, wemb_type, transform)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=bs,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=num_w,
                                            collate_fn = collate_fn
                                            )

  else:
    dataset = LPDataset(cap_json, fig_json, connect_json, vocab, root_dir, wemb_type, transform)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=bs,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=num_w,
                                            collate_fn = collate_fn
                                            )



  return loader



if __name__ == '__main__':

  #load data

  sp = 'psy-1'
  target_vocab_path = './vocab/%s_vocab.pkl' % sp
  vocab = pickle.load(open(target_vocab_path, 'rb'))
  root_dir = '/projects/dataset_processed/dongwonl/data/{}'.format(sp)

  with open("/projects/dataset_processed/dongwonl/data/{}/{}_figs.json".format(sp,sp), 'r') as f:
     fig_json = json.loads(f.read())

  with open("/projects/dataset_processed/dongwonl/data/{}/{}.json".format(sp,sp), 'r') as j:
     cap_json = json.loads(j.read())

  with open("/projects/dataset_processed/dongwonl/data/{}/{}_capfig.json".format(sp,sp), 'r') as c:
     connect_json = json.loads(c.read())


  transform = get_image_transform()
  wemb_type = 'bert'
  dataset = LPDataset(cap_json, fig_json, connect_json, vocab, root_dir, wemb_type, transform)
  loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=1,
                                            collate_fn = collate_fn
                                            )

#Press s to enter and you can find that you just entered __getitem__, and then you can set other breakpoints in n or in getitem and then debug with c                           
  pdb.set_trace()
  dataset[4]

# #
#   for idx, batch in enumerate(loader):
    
#     img, txt, ocr_target, cap_lengths, index, img_ids = batch
#     pdb.set_trace()