import os 
import argparse
from subprocess import call
import os
import pdb
import json

#pip install --upgrade scenedetect[opencv]
#https://pyscenedetect.readthedocs.io/en/latest/download/

parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset and csv', default =
args = parser.parse_args()


BASE_PATH = args.base_path

for dirpath, subdirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith('.mp4'):
            name = file
            dir_path = os.path.dirname(name)
            try:
                run_scene = "scenedetect -i {video_path} -o {video_root}  -s {video_path_wo_suffix}.stats.csv list-scenes detect-content -t 2.0 save-images split-video".format(video_path = name, video_root = dir_path, video_path_wo_suffix = file[:-4])
                res1 = call(run_scene, shell=True)
            except Exception:
                with open("./log_{range}.txt".format(range = str(start) +"_" + str(end)), "a") as file_object:
                    file_object.write(str(name) + "\n")
                pass


