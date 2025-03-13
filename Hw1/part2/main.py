import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter

def load_setting(setting_path):
    with open(setting_path, 'r') as f:
        lines = f.readlines()

    grayscale_params = []
    sigma_s, sigma_r = None, None

    for line in lines:
        values = line.strip().split(',')
        if values[0] == 'R':
            continue
        elif "sigma_s" in line:
            sigma_s = int(values[1])
            sigma_r = float(values[3])
        else:
            grayscale_params.append(list(map(float, values)))  # 解析 R, G, B 權重
    
    return np.array(grayscale_params), sigma_s, sigma_r

def generate_grayscale_images(img_rgb, grayscale_params):
    gray_images = []
    for params in grayscale_params:
        gray_img = np.dot(img_rgb, params).astype(np.uint8)  # RGB 加權求和轉換
        gray_images.append(gray_img)
    return gray_images

def save_images(output_dir, gray_images, jbf_results, min_idx, max_idx):
    """ 儲存影像結果 """
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, "gray_lowest_cost.png"), gray_images[min_idx])

    cv2.imwrite(os.path.join(output_dir, "gray_highest_cost.png"), gray_images[max_idx])

    cv2.imwrite(os.path.join(output_dir, "filtered_rgb_lowest_cost.png"), cv2.cvtColor(jbf_results[min_idx], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "filtered_rgb_highest_cost.png"), cv2.cvtColor(jbf_results[max_idx], cv2.COLOR_RGB2BGR))

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    parser.add_argument('--output_dir', default='./testdata', help='path to saving img directory')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    grayscale_params, sigma_s, sigma_r = load_setting(args.setting_path)

    gray_images = generate_grayscale_images(img_rgb, grayscale_params)
    gray_images.append(img_gray)

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_result = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    jbf_results = [JBF.joint_bilateral_filter(img_rgb, gray).astype(np.uint8) for gray in gray_images]

    L1_costs = []
    for i, jbf_result in enumerate(jbf_results):
        L1_norm  = np.sum(np.abs(bf_result.astype('int32') - jbf_result.astype('int32')))
        L1_costs.append(L1_norm)
        print(f'Gray Image {i+1} - L1 Similarity: {L1_norm}')
    
    min_idx = np.argmin(L1_costs)
    max_idx = np.argmax(L1_costs)
    save_images(args.output_dir, gray_images, jbf_results, min_idx, max_idx)
    

if __name__ == '__main__':
    main()