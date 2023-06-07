# MLP Dataset

MLP Dataset can be downloaded here: [Link to Data Download](https://drive.google.com/file/d/1amyxxy4reuHGQ3FaKLE5bYzVxD7DOl7K/view?usp=sharing)

We offer the Youtube video links in `raw_video_links.csv`, and the full MTurk annotations for in `figure_annotations.csv` (also included in the zipfile in the above link).

## Dataset Structure

Given `{Speaker}` as a speaker of interest, we have 3 json files:
* `/{Speaker}/{Speaker}.json` -- a json dictionary that contains the full information for each *slide*. Each *slide* is given an ID and the corresponding slide image, slide spoken language, slide object character recognition (OCR), slide mouse traces, the bounding boxes and labels (diagram, equation, table, natural image) of the figure on slide
* `/{Speaker}/{Speaker}_figs.json` -- a json dictionary that contains the full information for each *figure*. Each *figure* is given an ID and contains the same data as as above. 
* `/{Speaker}/{Speaker}_capfig.json` -- a json dictionary that maps the figures to the slide. The keys of the dictionary are the slide IDs in `{Speaker}.json` and the values are the figure IDs in `{Speaker}_figs.json`.

Given a speaker's `{Course Name}` (defaults 'unordered' if it is not an full course, but a collection of individual lecture videos), and `{Lecture Number}`, we offer the following:

*  `/{Speaker}/{Course Name}/{Lecture Number}/{video_id}_transcripts.csv` -- spoken language transcript from Google ASR
*  `/{Speaker}/{Course Name}/{Lecture Number}/segments.txt` -- manually annotated slide transition/segmentation timeframe, we extract the {Slide Number} from this data
* `/{Speaker}/{Course Name}/{Lecture Number}/slide_{Slide Number}.jpg` -- the raw slide image
* `/{Speaker}/{Course Name}/{Lecture Number}/slide_{Slide Number}_ocr.csv` -- the OCR of the in-slide text from Tesseract
* `/{Speaker}/{Course Name}/{Lecture Number}/slide_{Slide Number}_spoken.csv` -- the segmented spoken language 
* `/{Speaker}/{Course Name}/{Lecture Number}/slide_{Slide Number}_trace.csv` -- the automatically extracted mouse traces 
