import pandas as pd
import pdb
import datetime
import subprocess
#read all filtered df a
from IPython.display import display
from PIL import Image
import os 
import shutil


def cp_transcript_extract_frames(fin_df):
    for index, row in fin_df.iterrows():
        
        video_path = row['saved_dir']
        
        video_split = row['Answer.startTimeList'].split("|")[1:]
        src_dir_path = os.path.dirname(video_path)
        tgt_dir_path = src_dir_path
        os.makedirs(tgt_dir_path, exist_ok=True)

        video_id = row['video_id'].replace(".mp4", "")

        
        #extract frames 
        if len(video_split) == 0:
            
            dir_path = os.path.dirname(video_path)
            output_save = os.path.join(tgt_dir_path, "slide_{}.png".format(str(0).zfill(3)))
            time = str(row["Input.seconds"] - 1)
            frame_capture ="ffmpeg -ss {time_string} -i {input} -q:v 1 -qmin 1 -qmax 1 {output_save}".format(input = video_path, time_string = time[:-1], output_save = output_save)
            subprocess.call(frame_capture, shell = True)
            print(video_path)
            display(Image.open(output_save))
            pdb.set_trace()
    
#         for idx, time in enumerate(video_split):
#             dir_path = os.path.dirname(video_path)
#             output_save = os.path.join(tgt_dir_path, "slide_{}.jpg".format(str(idx).zfill(3)))
#     #         time_string = str(datetime.timedelta(seconds=float(time)))[:-3]
#     #         if not os.path.exists(output_save):

#             frame_capture ="ffmpeg -ss {time_string} -i {input} -frames:v 1 -q:v 2 {output_save}".format(input = video_path, time_string = time[:-1], output_save = output_save)
#             subprocess.call(frame_capture, shell = True)
#     #         display(Image.open(output_save))




parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset and csv', default = "/projects/dataset_processed/dongwonl/localized_presentations")
args = parser.parse_args()


path_to_df_w_saved_times = args.base_path
df = pd.read_csv(path_to_df_w_saved_times)
cp_transcript_extract_frames(df)
        
