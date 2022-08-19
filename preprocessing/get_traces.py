import numpy as np
import cv2
import os
import pdb
import pandas as pd
# Capture video from file


parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset and csv', default =
args = parser.parse_args()
                    
                    
def pointer_dict_gen(video_path):
    cap = cv2.VideoCapture(video_path)
    old_frame = None
    pointer_dict = {}
    count = 0
    while True:
        ret, captured_frame = cap.read()

        if ret == True:


            output_frame = captured_frame.copy()        

            if old_frame is not None:
                frame = cv2.absdiff(output_frame, old_frame)

                grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)




                if np.sum(blackAndWhiteImage > 0):
                    y_coord = int(blackAndWhiteImage.nonzero()[0].mean())
                    x_coord = int(blackAndWhiteImage.nonzero()[1].mean())
                    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    pointer_dict[count] = (x_coord, y_coord)

                    blackAndWhiteImage = cv2.circle(output_frame.copy(), (x_coord,y_coord), radius=0, color=(255, 0, 0), thickness=20)
                else:
                    blackAndWhiteImage = output_frame.copy()


            old_frame = output_frame
            count += 1
        else:
            print('Done:' + video_path)
            break

    cap.release()
    cv2.destroyAllWindows()
    return pointer_dict





for dirpath, subdirs, files in os.walk(args.base_path):
    for file in files:
        if file.endswith('.mp4') and "-Scene-" in file:
            vid_path = os.path.join(dirpath, file)
            csv_path = os.path.join(vid_path.replace(".mp4", "-Trace.csv"))
            if not os.path.exists(csv_path):
                pointer_dict = pointer_dict_gen(vid_path)
                df = pd.DataFrame(list(pointer_dict.items()),columns = ['frame','coord'])
                df.to_csv(csv_path)