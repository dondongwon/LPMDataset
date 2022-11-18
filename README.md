# Multimodal Lecture Presentations (MLP) Dataset

This is the official repository for the *Multimodal Lecture Presentations (MLP) Dataset* 

The dataset can be downloaded here:
[MLP Dataset Download](https://drive.google.com/file/d/1amyxxy4reuHGQ3FaKLE5bYzVxD7DOl7K/view?usp=sharing)

Our Arxiv Paper can be found here: 
[Multimodal Lecture Presentations Dataset: Understanding Multimodality in Educational Slides](https://arxiv.org/abs/2208.08080)

The quickstart on colab can be found here: 
[Colab Quickstart](https://colab.research.google.com/drive/1htS8UltlGOw3XZiNFBx74z0of-4HJQ7l?usp=sharing)


 [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
 
[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# Overview

This repo is divided into the following sections:

* `dataset` -- this directory contains more information on the dataset structure 
* `examples` -- quickstart to use our data (dataloading, visualization), and other tools
* `src` -- contains our full experimental setup, including our proposed model `PolyViLT` and a dataloader that can be used to load the data
* `preprocessing` -- scripts that can be used to automatically process the data 
* `human_study` -- results of our human study, raw images and aligned captions can be downloaded here for quick investigation

![](/images/overview.png)

Lecture slide presentations, a sequence of pages that contain text and figures accompanied by speech, are constructed and presented carefully in order to optimally transfer knowledge to students. Previous studies in multimedia and psychology attribute the effectiveness of lecture presentations to their multimodal nature. As a step toward developing AI to aid in student learning as intelligent teacher assistants, we introduce the Multimodal Lecture Presentations dataset as a large-scale benchmark testing the capabilities of machine learning models in multimodal understanding of educational content. To benchmark the understanding of multimodal information in lecture slides, we introduce two research tasks which are designed to be a first step towards developing AI that can explain and illustrate lecture slides: automatic retrieval of (1) spoken explanations for an educational figure (Figure-to-Text) and (2) illustrations to accompany a spoken explanation (Text-to-Figure)

![](/images/datapipeline.png)

As a step towards this direction, MLP dataset contains over 9000 slides with natural images, diagrams, equations, tables and written text, aligned with the speaker's spoken language. These lecture slides are sourced from over 180 hours worth of educational videos in various disciplines such as anatomy, biology, psychology, speaking, dentistry, and machine learning. To enable the above mentioned tasks, we manually annotated the slide segments to accurately capture alignment between spoken language, slides, and figures (diagrams, natural images, table, equations).


MLP Dataset and its tasks bring new research opportunities through the following technical challenges: (1) addressing weak crossmodal alignment between figures and spoken language (a figure on the slide is often related to only a portion of spoken language), (2) representing novel visual mediums of man-made figures (e.g., diagrams, tables, and equations), (3) understanding technical language, and (4) capturing interactions in long-range sequences. Furthermore, it offers novel challenges that will spark future research in educational content modeling, multimodal reasoning, and question answering.


## Dataset Structure

In the dataset, you will find:
```
data
│   raw_video_links.csv - contains the slide segmentations, number of seconds, video_id, youtube_url, slide annotation containing learning objectives)
│   figure_annotations.csv - the save_directory path, ocr output, the bounding boxes and labels) 
│
└───{speaker}
    │   {speaker}.json - contains comprehensive data dictionary, where each datapoint corresponds to a slide 
    │   {speaker}_figs.json - contains dictionary where each datapoint corresponds to a figure
    │   {speaker}_capfig.json - a dictionary that maps the keys of {speaker}.json to {speaker}_figs.json, such that we can map the captions to the multiple figures
│   │
│   └───{lecture name or "unordered"} - a folder for each leacture series or 'unordered' (if videos are not from a consecutive series)
│       │   slide_{number}.jpg - image of the slide (**not shared directly, needs preprocessing - please refer to /preprocessing**)
│       │   slide_{number}_ocr.jpg - ocr of the slide
│       │   slide_{number}_spoken.jpg - spoken language of the slide
│       │   slide_{number}_trace.jpg - extracted mouse traces for slide
│       │   ...
│       │   segments.txt - the annotated slide transition segments
│       │   {video_id}_transcripts.csv - the google ASR transcriptions of spoken language
│       │   slide_{number}_ocr.jpg - ocr of the slide
└─── ...

```

## Setting up the Environment

### To use the dataset only:

```
python3 -m venv venv
pip install -r requirements_dataset_only.txt
```

### To use the full codebase with all baselines:

```
conda create -y --name mlp_env python=3.7
conda install --force-reinstall -y -q --name mlp_env -c conda-forge --file requirements_full.txt
```
You will have the necessary environment to run our scrips and easily use our dataset at `mlp_env`.

For quickstart, we recommend the user to take a look at our `quickstart.ipynb`


## Extend Dataset

For those interested in extending the dataset, please check our `/preprocessing` directory. We have provided details steps to do so. 

## Train  model

You can train your own model using [train.py](train.py); check [option.py](option.py) for all available options. We provide example scripts below, for the speaker 'anat-1' and seed 3. We used the train and test split using the seeds 0, 2, 3. This will reconstitute the exact splits of the dataset that could be used to reproduce the results in our paper. 


### Baselines 


Here are the scripts used to test baselines:

```
#Random
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/random/bert/anat-1 --logger_name ./runs/random/bert/anat-1 --model Random --word_dim 768  

#PVSE (Glove)
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/pvse/bert/anat-1 --logger_name ./runs/pvse/bert/anat-1 --model PVSE --word_dim 768

#PVSE (Bert)
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type glove --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/pvse/glove/anat-1 --logger_name ./runs/pvse/glove/anat-1 --model PVSE  

#PCME (Glove)
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/pcme/bert/anat-1 --logger_name ./runs/pcme/bert/anat-1  --embed_dim 1024 --n_samples_inference 7 --model PCME --img_probemb --txt_probemb --word_dim 768  

#PCME (BERT) 
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type glove --margin 0.1 --max_violation --num_embeds 5 --txt_attention   --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/pcme/glove/anat-1 --logger_name ./runs/pcme/glove/anat-1  --embed_dim 1024 --n_samples_inference 7 --model PCME --img_probemb --txt_probemb  

#CLIP
python clip.py --seed 3 --log_file ./logs/clip/anat-1 --data_name anat-1

```

### Our Model

We provide scripts to train our model, with all the ablations reported in the main paper.

```
#Ours 
python3 train.py --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert+vilt --margin 0.1 --max_violation --num_embeds 5 --txt_attention --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/vilt/no_trace/anat-1 --logger_name ./runs/vilt/no_trace/anat-1 --model Ours_VILT --word_dim 768 --embed_size 768  

#Ours (Using Mouse Trace)
python3 train.py --model Ours_VILT_Trace  --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert+vilt --margin 0.1 --max_violation --num_embeds 5 --txt_attention --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/vilt/trace/anat-1 --logger_name ./runs/vilt/trace/anat-1 --word_dim 768 --embed_size 768  

#Ours (Across all speakers)
python3 train.py --data_name all --seed 3 --cnn_type resnet152 --wemb_type bert+vilt --margin 0.1 --max_violation --num_embeds 5 --txt_attention --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/vilt/trace/anat-1 --logger_name ./runs/vilt/trace/anat-1 --model Ours_VILT_Trace --word_dim 768 --embed_size 768  

#Ours (w/o figure language)
python3 train.py --lang_mask --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert+vilt --margin 0.1 --max_violation --num_embeds 5 --txt_attention --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/vilt/no_lang2/anat-1 --logger_name ./runs/vilt/no_lang2/anat-1 --model Ours_VILT --word_dim 768 --embed_size 768   

#Ours (w/o figure image)
python3 train.py --img_mask   --seed 3 --data_name anat-1 --cnn_type resnet152 --wemb_type bert+vilt --margin 0.1 --max_violation --num_embeds 5 --txt_attention --img_finetune --mmd_weight 0.01 --div_weight 0.01 --batch_size 8 --log_file ./logs/vilt/no_lang2/anat-1 --logger_name ./runs/vilt/no_lang2/anat-1 --model Ours_VILT --word_dim 768 --embed_size 768 

```



## License
At the time of release, all videos included in this dataset were being made available by the original content providers under the standard [YouTube License](https://www.youtube.com/static?template=terms).

Unless noted otherwise, we are providing the data of this repository under under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

