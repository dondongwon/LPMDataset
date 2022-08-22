import pandas as pd
import os
import IPython
import sys
import numpy as np
import cv2
import pdb
import pytesseract
from PIL import Image 
from pytesseract import Output
import matplotlib.pyplot as plt
%matplotlib inline
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset ', default = "/projects/dataset_processed/dongwonl/localized_presentations")
args = parser.parse_args()

BASE_PATH = args.base_path



def get_OCR(img_path):
    image = cv2.imread(img_path)
    results = pytesseract.image_to_data(image, output_type=Output.DICT)
    #get surrounding median color
    image_2 = image.copy()
    unique, counts = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    image_2[:,:,0], image_2[:,:,1], image_2[:,:,2] = unique[np.argmax(counts)]
    color = unique[np.argmax(counts)]
    color_tuple = (int(color[0]), int(color[1]), int(color[2]))
    
    kf_dict = OrderedDict()
    
    for i in range(0, len(results["text"])):
      x = results["left"][i]
      y = results["top"][i]
      w = results["width"][i]
      h = results["height"][i]

      text = results["text"][i]
      
      conf = int(results["conf"][i])
      text = text.strip()
      if conf > 70 and len(text) > 0:
        print(text)
        
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        cv2.rectangle(image, (x, y), (x + w, y + h),(255, 0, 0), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 200), 2)
        kf_dict[text] = [x,y,w,h]

    return kf_dict

for dirpath, subdirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith('.jpg'):
            img_path = os.path.join(dirpath, file)
            print(img_path)
            

            ocr_dict = get_OCR(img_path)

            dest_path = img_path.replace("localized_presentations_imgs", "localized_presentations_ocr")
            dest_path = dest_path.replace('.jpg', '.pickle')
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as file_:
                pickle.dump(ocr_dict, file_, -1)



