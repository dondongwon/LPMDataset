# PolyVILT and Other Baselines


This repository contains an PyTorch implementation of Figure-to-Text and Text-to-Figure Retrieval for Lecture Presentations Dataset:  


Baselines:  
1. PolyViLT (Ours)
2. PolyViLT w/ Trace (Ours)
3. PVSE [*Polysemous Visual-Semantic Embedding for Cross-Modal Retrieval* (CVPR 2019)](https://arxiv.org/abs/1906.04402).
4. PVSE w/ BERT 
5. PCME [*Probabilistic Cross-Modal Embedding* (CVPR 2021)](https://arxiv.org/abs/2101.05068).
6. PCME w/ BERT
7. Pre-trained CLIP [*CLIP: Connecting Text and Images* (OpenAI)](https://openai.com/blog/clip/).
8. Random Baseline


```
#PVSE 
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type glove --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/pvse/glove/anat-1 --logger_name ./runs/pvse/glove/anat-1 --model PVSE  

#PVSE w/ BERT
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/pvse/bert/anat-1 --logger_name ./runs/pvse/bert/anat-1 --model PVSE --word_dim 768

#PCME
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type glove --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/pcme/glove/anat-1 --logger_name ./runs/pcme/glove/anat-1  --embed_dim 1024 --n_samples_inference 7 --model PCME --img_probemb --txt_probemb  

#PCME w/ BERT
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/pcme/bert/anat-1 --logger_name ./runs/pcme/bert/anat-1  --embed_dim 1024 --n_samples_inference 7 --model PCME --img_probemb --txt_probemb --word_dim 768  

#Ours
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert+vilt --margin 0.1 --max_violation --num_embeds 5 --txt_attention --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/polyvilt/no_trace/anat-1 --logger_name ./runs/polyvilt/no_trace/anat-1 --model Ours_VILT --word_dim 768 --embed_size 768  

#Ours w/ Mouse Trace
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert+vilt --margin 0.1 --max_violation --num_embeds 5 --txt_attention --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/polyvilt/trace/anat-1 --logger_name ./runs/polyvilt/trace/anat-1 --model Ours_VILT_Trace --word_dim 768 --embed_size 768  

#CLIP
python clip.py --seed 3 --log_file ./logs/clip/anat-1 --data_name anat-1 --wemb_type glove

#Random
python3 train.py --seed 3 --data_name anat-1  --wemb_type bert --max_violation --batch_size 8 --log_file ./logs/random/bert/anat-1 --logger_name ./runs/random/bert/anat-1 --model Random 



```
