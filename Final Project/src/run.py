import argparse
import numpy as np
import cv2

from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import glob
from PIL import Image
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ganzin Iris Recognition Challenge')
    parser.add_argument('--input', type=str, metavar='PATH', required=True,
                        help='Input file to specify a list of sampled pairs')
    parser.add_argument('--output', type=str, metavar='PATH', required=True,
                        help='A list of sampled pairs and the testing results')

    args = parser.parse_args()

    cfg = get_cfg("cfg.yaml")
    irisRec = irisRecognition(cfg)

    vector_dict = {}
    with open(args.input, 'r') as in_file, open(args.output, 'w') as out_file:
        for line in in_file:
            lineparts = line.split(',')
            img1_path = lineparts[0].strip()
            img2_path = lineparts[1].strip()
            
            vector_pair = []
            for img_path in [img1_path, img2_path]:

                # img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                # img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

                file_name = os.path.basename(img_path)
                if file_name in vector_dict:
                    vector = vector_dict[file_name]
                    vector_pair.append(vector)
                    continue

                img = Image.fromarray(np.array(Image.open(img_path).convert("RGB"))[:, :, 0], "L")

                # Iris Detection Pipeline
                # convert to ISO-compliant aspect ratio (4:3) and resize to ISO-compliant resolution: 640x480
                im = irisRec.fix_image(img)

                # segmentation mask and circular approximation:
                mask, pupil_xyr, iris_xyr = irisRec.segment_and_circApprox(im)
                im_mask = Image.fromarray(np.where(mask > 0.5, 255, 0).astype(np.uint8), 'L')

                # cartesian to polar transformation:
                im_polar, mask_polar = irisRec.cartToPol_torch(im, mask, pupil_xyr, iris_xyr)

                # human-driven BSIF encoding:
                vector = irisRec.extractVector(im_polar)
                
                vector_pair.append(vector)
                

            # Matching
            score = irisRec.matchVectors(vector_pair[0], vector_pair[1])
            output_line = f"{img1_path}, {img2_path}, {score}"
            print(output_line)
            out_file.write(output_line.rstrip('\n') + '\n')
