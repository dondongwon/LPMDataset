
import cv2
import layoutparser as lp
import matplotlib.pyplot as plt
import os
import pdb
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset ', default = "/projects/dataset_processed/dongwonl/localized_presentations")


args = parser.parse_args()


BASE_PATH = args.base_path


for dirpath, subdirs, files in os.walk(BASE_PATH):
    for file in files:
        if file.endswith('.jpg'):
            img_path = os.path.join(dirpath, file)
            print(img_path)
            
            model = lp.Detectron2LayoutModel('lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config',
                                extra_config= ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                
                                label_map={1:"TextRegion", 2:"ImageRegion", 3:"TableRegion", 4:"MathsRegion", 5:"SeparatorRegion", 6:"OtherRegion"}
                                )

        
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            layout = model.detect(image)
            
            ocr_agent = lp.TesseractAgent(languages='eng') 
            
            for block in layout:
                if block.type == 'TextRegion':
                    segment_image = (block
                                       .pad(left=10, right=10, top=10, bottom=10)
                                       .crop_image(image))
                        # add padding in each image segment can help
                        # improve robustness 

                    text = ocr_agent.detect(segment_image)
                    block.set(text=text, inplace=True)


            dest_path = img_path.replace("localized_presentations_imgs", "localized_presentations_layout")
            dest_path = dest_path.replace('.jpg', '.pickle')
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as file_:
                pickle.dump(layout, file_, -1)

            

