
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        # Spatial kernel
        GaussianSpatial = np.zeros((self.wndw_size, self.wndw_size))
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                GaussianSpatial[i, j] = np.exp(np.divide(np.square(i - self.pad_w) + np.square(j - self.pad_w), -2 * np.square(self.sigma_s)))
        
        # Normalize guidance(gray)
        padded_guidance = padded_guidance.astype('float64') / 255

        # Range kernel
        padded_img = padded_img.astype('float64')
        output = np.zeros(img.shape)

        for i in range(self.pad_w, padded_guidance.shape[0] - self.pad_w):
            for j in range(self.pad_w, padded_guidance.shape[1] - self.pad_w):
                Tp = padded_guidance[i, j]
                Tq = padded_guidance[i - self.pad_w: i + self.pad_w + 1, j - self.pad_w: j + self.pad_w + 1]
                power = np.divide(np.square(Tq - Tp), -2 * np.square(self.sigma_r))
                if len(power.shape) == 3:
                    power = power.sum(axis=2)
                GaussianRange = np.exp(power)

                G = np.multiply(GaussianSpatial, GaussianRange)
                W = G.sum()

                Iq = padded_img[i - self.pad_w: i + self.pad_w + 1, j - self.pad_w: j + self.pad_w + 1] # (19, 19, 3)
                for c in range(img.shape[2]):
                    output[i - self.pad_w, j - self.pad_w, c] = np.multiply(G, Iq[:,:,c]).sum() / W
        
        return np.clip(output, 0, 255).astype(np.uint8)