from __future__ import unicode_literals

import argparse
from subprocess import call
import os
from youtube_dl import YoutubeDL
import json
import pandas as pd 
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset and csv', default = "/projects/dataset_processed/dongwonl/localized_presentations")
args = parser.parse_args()


BASE_PATH = args.base_path
CSV_PATH = os.path.join(BASE_PATH, "lecture_vids.csv")

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df = df.dropna()

    for index, row in df.iterrows():
        print("\n We are at Row: {} \n".format(index))
        speaker = row['channel']
        scrape_scope = row['scrape_scope']
        link = row['link']

        save_dir = os.path.join(BASE_PATH, speaker)

        if scrape_scope == 'channel':
            run =  "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' -i -o {save_dir}'/%(id)s/%(id)s.%(ext)s'  -v {link}".format(link = link, save_dir = save_dir)

        if scrape_scope == 'playlist':
            run = "youtube-dl -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' -i -o {save_dir}'/%(playlist_title)s/%(playlist_index)s/%(id)s.%(ext)s' --write-auto-sub {link}".format(link = link, save_dir = save_dir)
        
        res1 = call(run, shell=True)
        


    
