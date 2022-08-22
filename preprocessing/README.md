## Preprocessing

![](/images/preproc_mturk_1.png)

The above image represents how the dataset was collected and processed. The grey boxes indicate forms of data (CSV consisting of Youtube Video Links or Playlist Link, Raw Video, Audio, and Cropped Videos). The red indicates a processing script or step. The green indicates the resulting, final data form used in our dataset. We list additional information for each processing step below.


### Download Videos 

We use the `download_raw_videos.py` to download youtube videos from a csv. For each row, we have a column which has the `link`, `scrape_scope`, which indicates if we are going to download the entire channel or a specific playlist. 

### Collecting Automatic Transcriptions
We use `ASR.py` to collect Google ASR transcriptons to be used for our dataset. Please note that this is a paid service and you can find more information regarding the pricing [here](https://cloud.google.com/speech-to-text/pricing). After downloading the videos, you need to extract the FLAC audio from the vidoe, which can be done directly from the youtube-dl script or post-hoc by using FFMPEG. Our script takes in a directory path, and a bucket_name from your Google Account. It uploads the audio file on-the-fly transcribes it, saves it into your directory and deletes it afterwards. 

### Scene Detect (aka Slide Segmentation)
We use the provided MTurk `mturk_slide_seg.html` to collect slide end segments. For a screenshot, please check the appendix of the main paper. Here you need to feed a csv with links to videos, which we recommend saving directly onto a AWS server. We provide detailed instructions, as well as links to our instruction video. You can also find it [here](https://youtu.be/LEKoAzU_kjM). We highly recommend the user to preview the content in MTurk Sandbox.

### Extract Trace
We extract the mouse traces in `get_traces.py` by looking at the segmented videos and calculating the pixel-wise difference between frame. For each segmented slide, the background is static and the only object that is moving is the pointer. If there is any movement, we consider that as the pointer location.

### Extract Slide Image
We extract the slide image via `extract_slide_imgs.py` from the given annotations, saved in a CSV outputted from the MTurk experiment in Scene Detect. We use the saved annotations in the dataframe with FFMPEG to save the slide image.

### Extract Slide Figures 
We use the provided MTurk `mturk_layout_parsing.html` to collect slide figures. Here you need to feed a csv with links to images, which we recommend saving directly onto a AWS server. We provide detailed instructions in our Instructions Tab. Our class labels are inspired from PRImA, a dataset that consists of layouts from scientific reports. We follow their taxonomy to find labels on figures, which consist of natural images,
diagrams, table, and equations. In Appendix D of our main paper we provide details on figure class labels and a screenshot of the MTurk experiment. Afterwards we use `ocr.py` to extract the written text on the slide, it takes in a in a directory path with the images of each slide and outputs a pickle file with dictionary containing the strings and location values.


![](/images/preproc_auto_2.png)

There are only *2* main processing steps that needs to be automated for easy expansion of our dataset. Instead of using manual MTurk annotations, we list the automatic alternatives and provide the scripts to enable them. 

### Scene Detect (via PySceneDetect)
Instead of using MTurk `mturk_slide_seg.html, to collect the slide changes, we can utilize [PySceneDetect](https://pypi.org/project/scenedetect/),  "a command-line tool and Python library which analyzes a video, looking for scene changes or cuts". The script to enable this is in `pyscenedetect.py`. The script takes as input a base-path of videos to be segmented. Pyscene detect allows for many different arguments, such as 'save-images' that saves the desired slide images as you process the data, 'split-video' that saves the cropped videos, {}.stats.csv' that saves the metadata from the processing. 

### Extract Slide Figures (via LayoutParser)
Instead of using MTurk  `mturk_layout_parsing.html` to collect slide figures. We utilize [LayoutParser](https://github.com/Layout-Parser/layout-parser) that allows you to perform deep learning based layout detection simply, and also run OCR for each detected layout region. Layout Parser also supports the PRImA class label format, which is identical to the class labels that we utilize. The script can be found in `layoutparser.py`, which takes in a path and extracts the layouts of all images it finds in its subdirectories and saves them to a pickle file.



