# Human Study Results

We share the results for human study here:

To measure human student performance, we randomly sampled at least 10 figures from the unseen test set for each speaker from 3 random seeds. 
We ask the human student annotator to mark the caption which describes the figure.
For fair comparison, we use the same experimental set-up for our models, where we measure the recall@1 for the same 10 figures and associated captions. 
The results are shown below (Fig 4 in original paper):

![](/images/human_study.png)


For easy investigation into where the human evaluation fails, we ware providing an organized directory where you can easily access the test set used for human student evaluation.

You can find the results, the exact test set (images and captions) used for human evaluation. 

* 1 (Seed Number)
  * anat-1 (Speaker Number)
    *  anat-1.json (Dictionary containing ground truth aligned pairs of image IDs (Key), caption IDs (Value))
    *  results.json (Human Annotation of aligned pairs of image IDs (Key), caption IDs (Value)) 
    *  texts.json (Dictionary containing caption ID (Key), and ground truth captions (Value))
    *  scores.json (Recall@1 scores)
    *  xx.jpg (Images in Test Set) 


