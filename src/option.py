import os, sys, pdb

import argparse
parser = argparse.ArgumentParser(description='Parameters for training PVSE')

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# Names, paths, logging, etc
parser.add_argument('--data_name', default='ml-1', choices=("anat-1", "bio-1", "bio-3", "dental", "ml-1", "psy-2", "anat-2",  "bio-2", "bio-4", "psy-1", "speaking"), help='Dataset name (coco|tgif|mrw)')
parser.add_argument('--data_path', default=CUR_DIR+'/data/', help='path to datasets')
parser.add_argument('--vocab_path', default=CUR_DIR+'/vocab/', help='Path to saved vocabulary pickle files')
parser.add_argument('--logger_name', default=CUR_DIR+'/runs/runX', help='Path to save the model and logs')
parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log')
parser.add_argument('--log_file', default=CUR_DIR+'/logs/logX.log', help='Path to save result logs') 
parser.add_argument('--debug', action='store_true', help='Debug mode: use 1/10th of training data for fast iteration')
parser.add_argument('--slide_level', action='store_true', help='using slide level image features')


# Data parameters
parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding')
parser.add_argument('--workers', default=16, type=int, help='Number of data loader workers')
parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input')
parser.add_argument('--max_video_length', default=1, type=int, help='Maximum length of video sequence')

# Model parameters
parser.add_argument('--cnn_type', default='resnet152', help='The CNN used for image encoder')
parser.add_argument('--embed_size', default=1024, type=int, help='Dimensionality of the joint embedding')
parser.add_argument('--wemb_type', default=None, choices=('only-vilt', 'bert+vilt', 'bert', 'glove','fasttext'), type=str, help='Word embedding (glove|fasttext)')
parser.add_argument('--margin', default=0.1, type=float, help='Rank loss margin')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout rate')
parser.add_argument('--max_violation', action='store_true', help='Use max instead of sum in the rank loss')
parser.add_argument('--order', action='store_true', help='Enable order embedding')

# Attention parameters
parser.add_argument('--img_attention', action='store_true', help='Use self attention on images/videos')
parser.add_argument('--txt_attention', action='store_true', help='Use self attention on text')
parser.add_argument('--num_embeds', default=1, type=int, help='Number of embeddings for MIL formulation')

# Loss weights
parser.add_argument('--mmd_weight', default=.0, type=float, help='Weight term for the MMD loss')
parser.add_argument('--div_weight', default=.0, type=float, help='Weight term for the log-determinant divergence loss')

# Training / optimizer setting
parser.add_argument('--img_finetune', action='store_true', help='Fine-tune CNN image embedding')
parser.add_argument('--txt_finetune', action='store_true', help='Fine-tune the word embedding')
parser.add_argument('--val_metric', default='rsum', choices=('rsum','med_rsum','mean_rsum'), help='Validation metric to use (rsum|med_rsum|mean_rsum)')
parser.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs')
parser.add_argument('--batch_size', default=8, type=int, help='Size of a training mini-batch')
parser.add_argument('--batch_size_eval', default=16, type=int, help='Size of a evaluation mini-batch')
parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold')
parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay (l2 norm) for optimizer')
parser.add_argument('--lr', default=.0002, type=float, help='Initial learning rate') #default 0.0002
parser.add_argument('--ckpt', default='', type=str, metavar='PATH', help='path to latest ckpt (default: none)')
parser.add_argument('--eval_on_gpu', action='store_true', help='Evaluate on GPU (default: CPU)')
parser.add_argument('--legacy', action='store_true', help='Turn this on to reproduce results in CVPR2018 paper')
parser.add_argument('--seed', default=0, type=int, help='Random Seed')
parser.add_argument('--eval', action='store_true', help='Eval Loop')
parser.add_argument('--human_study', action='store_true', help='Eval Loop')



## PCME args 
parser.add_argument('--embed_dim', default='1024', help='Embedding dimensions')
parser.add_argument('--n_samples_inference', default='1', help='n_samples inference')
parser.add_argument('--model', required=True, help='["PCME", "PVSE"]')
parser.add_argument('--init_negative_scale', default='15', help='n_samples inference')
parser.add_argument('--init_shift', default='15', help='n_samples inference')
parser.add_argument('--num_samples', default='7', help='n_samples inference')
parser.add_argument('--vib_beta', default='0.00001', help='n_samples inference')
parser.add_argument('--uniform_lambda', default='10', help='n_samples inference')
parser.add_argument('--criterion__negative_sampling_ratio', default='-1', help='n_samples inference')
parser.add_argument('--img_probemb', action='store_true', help='using img prob embeddings')
parser.add_argument('--txt_probemb', action='store_true', help='using txt prob embeddings')




def verify_input_args(args):
  # Process input arguments
  if args.ckpt:
    assert os.path.isfile(args.ckpt), 'File not found: {}'.format(args.ckpt)
  # if args.num_embeds > 1 and (args.img_attention and args.txt_attention) is False:
  #   print('When num_embeds > 1, both img_attention and txt_attention must be True')
  #   sys.exit(-1)
  if not (args.val_metric=='rsum' or args.val_metric=='med_rsum' or args.val_metric=='mean_rsum'):
    print('Unknown validation metric {}'.format(args.val_metric))
    sys.exit(-1)
  if not os.path.isdir(args.logger_name):
    print('Creading a directory: {}'.format(args.logger_name))
    os.makedirs(args.logger_name)
  if args.log_file and not os.path.isdir(os.path.dirname(args.log_file)):
    print('Creating a directory: {}'.format(os.path.dirname(args.log_file)))
    os.makedirs(os.path.dirname(args.log_file))
  return args
