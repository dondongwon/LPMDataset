## Preprocessing

![](/images/preproc_mturk_1.png)

The above image represents how the dataset was collected and processed. The grey boxes indicate forms of data (CSV consisting of Youtube Video Links or Playlist Link, Raw Video, Audio, and Cropped Videos). The red indicates a processing script or step. The green indicates the resulting, final data form used in our dataset. We list additional information for each processing step below.


### Download Videos 

We use the `download_raw_videos.py` to download youtube videos from a csv. For each row, we have a column which has the `link`, `scrape_scope`, which indicates if we are going to download the entire channel or a specific playlist. 

### Collecting Automatic Transcriptions
We use `ASR.py` to collect Google ASR transcriptons to be used for our dataset. Please note that this is a paid service and you can find more information regarding the pricing [here](https://cloud.google.com/speech-to-text/pricing). After downloading the videos, you need to extract the FLAC audio from the vidoe, which can be done directly from the youtube-dl script or post-hoc by using FFMPEG. Our script takes in a directory path, and a bucket_name from your Google Account. It uploads the audio file on-the-fly transcribes it, saves it into your directory and deletes it afterwards. 

### Scene Detect (aka Slide Segmentation)
We use the provided MTurk `mturk_slide_seg.html` to collect slide end segments. For a screenshot, please check the appendix of the main paper. Here you need to feed a csv with links to images, which we recommend saving directly onto a AWS server. We provide detailed instructions, as well as links to our instruction video. You can also find it [here](https://youtu.be/LEKoAzU_kjM). We highly recommend the user to preview the content in MTurk Sandbox.

### Extract Trace
We extract the mouse traces in `get_traces.py` by looking at the segmented videos and calculating the pixel-wise difference between frame. For each segmented slide, the background is static and the only object that is moving is the pointer. If there is any movement, we consider that as the pointer location.

### Extract Slide Image
We extract the slide image via `extract_slide_imgs.py` from the given annotations, saved in a CSV outputted from the MTurk experiment in Scene Detect. We use the saved annotations in the dataframe with FFMPEG to save the slide image.




![](/images/preproc_auto_1.png)


- [x] (Reviewer FPNg, GHav) Dataset Extension: Include prepocessing tools, as well as automatic preprocessing steps (LayoutParser, PysceneDetect)
- [x] Download video script
- [x] Pyscene Detect 
- [x] Layout Parser
- [x] Google ASR
- [x] Extract Frames
- [x] Get Traces


- [x] (Reviewer FPNg, GHav) Dataset Extension: Include manual annotation scripts (MTurk javascript code)
- [x] MTurk Scripts

