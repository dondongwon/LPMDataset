# Multimodal Lecture Presentations (MLP) Dataset

This is the official repository for the *Multimodal Lecture Presentations (MLP) Dataset* 

The dataset can be downloaded here: [Download](https://drive.google.com/file/d/13aDrmStlaSDFpacSXMOH1M5gaTo0i8c-/view?usp=sharing)


 [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
 
[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# Overview

![](/images/overview.png)

Lecture slide presentations, a sequence of pages that contain text and figures accompanied by speech, are constructed and presented carefully in order to optimally transfer knowledge to students. Previous studies in multimedia and psychology attribute the effectiveness of lecture presentations to their multimodal nature. As a step toward developing AI to aid in student learning as intelligent teacher assistants, we introduce the Multimodal Lecture Presentations dataset as a large-scale benchmark testing the capabilities of machine learning models in multimodal understanding of educational content. To benchmark the understanding of multimodal information in lecture slides, we introduce two research tasks which are designed to be a first step towards developing AI that can explain and illustrate lecture slides: automatic retrieval of (1) spoken explanations for an educational figure (Figure-to-Text) and (2) illustrations to accompany a spoken explanation (Text-to-Figure)

![](/images/datapipeline.png)

As a step towards this direction, MLP dataset contains over 9000 slides with natural images, diagrams, equations, tables and written text, aligned with the speaker's spoken language. These lecture slides are sourced from over 180 hours worth of educational videos in various disciplines such as anatomy, biology, psychology, speaking, dentistry, and machine learning. To enable the above mentioned tasks, we manually annotated the slide segments to accurately capture alignment between spoken language, slides, and figures (diagrams, natural images, table, equations).


MLP Dataset and its tasks bring new research opportunities through the following technical challenges: (1) addressing weak crossmodal alignment between figures and spoken language (a figure on the slide is often related to only a portion of spoken language), (2) representing novel visual mediums of man-made figures (e.g., diagrams, tables, and equations), (3) understanding technical language, and (4) capturing interactions in long-range sequences. Furthermore, it offers novel challenges that will spark future research in educational content modeling, multimodal reasoning, and question answering.

## Paper

Coming Soon... 

## License
At the time of release, all videos included in this dataset were being made available by the original content providers under the standard [YouTube License](https://www.youtube.com/static?template=terms).

Unless noted otherwise, we are providing the data of this repository under under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

## Contributors

Correspondence to: 
  - [Dong Won Lee](http://dongwonl.com) (dongwonl@cs.cmu.edu)
  - [Chaitanya Ahuja](http://chahuja.com) (cahuja@andrew.cmu.edu)
  - [Paul Pu Liang](https://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)
  - [Sanika Natu]() (snatu@andrew.cmu.edu)
  - [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/) (morency@cs.cmu.edu)


