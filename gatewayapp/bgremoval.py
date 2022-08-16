# Copyright (C) 2020 - 2022 APC, Inc.

from PIL import Image
import os, sys
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], './backgroundremoval'))
from libs.networks import U2NET
import numpy as np
import cv2

g_model = None

class myU2NET(U2NET):
    def generate_mask(self, pil_img):
        image, org_image = self.__load_image__(pil_img)  # Load image

        image = image.type(self.torch.FloatTensor)
        if self.torch.cuda.is_available():
            image = self.Variable(image.cuda())
        else:
            image = self.Variable(image)
        
        mask, d2, d3, d4, d5, d6, d7 = self.__net__(image)  # Predict mask

        # Normalization
        print("Mask normalization")
        mask = mask[:, 0, :, :]
        mask = self.__normalize__(mask)

        # Prepare mask
        print("Prepare mask")
        mask = self.__prepare_mask__(mask, org_image.size)
        # mask.save('mask.png')

        return mask

    # roi = (x_left, y_upper, x_right, y_bottom)
    def get_points(self, pil_image, roi):
        print("roi", roi)
        # get mask
        mask = self.generate_mask(pil_image.crop(roi))
        # convert to bw
        mask.convert("1")
        maskbw = np.array(mask, dtype=np.uint8)

        # create a binary thresholded image
        _, binary = cv2.threshold(maskbw, 225, 255, cv2.THRESH_BINARY)

        # find the contours from the thresholded image
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        print("contours", len(contours))
        prev_size = 0
        biggest_contour = None
        for contour in contours:
            print('countour', contour.shape, "sample", contour[0:2].shape)
            x,y,w,h = cv2.boundingRect(contour)
            print(x, y, w, h)
            if (prev_size < w * h):
                prev_size = w * h
                # biggest_contour = np.reshape(contour, [contour.shape[0], 2])
                biggest_contour = contour
        

        if (prev_size > 0 and biggest_contour.shape[0] > 4):
            eps = 0.0005 # hardcoded (smaller -> more points)
            peri = cv2.arcLength(biggest_contour, True)
            approx = cv2.approxPolyDP(biggest_contour, eps * peri, True)
            print("biggest contour", biggest_contour.shape, "sample", biggest_contour[0:2])
            print("approx contour", approx.shape, "sample", approx[0:2])
            approx = np.reshape(approx, [approx.shape[0], 2])
            points =  " ".join([str(int(a[0] + roi[0])) + ',' + str(int(a[1] + roi[1]))  for a in approx])
            # print("point", points)
            return points
        else:
            return None




def setup():
    print("setup background removal model")
    global g_model
    g_model = myU2NET()  # Load model

def get_points_str(pil_image, xyhw):
    xywh_arr = xyhw.split(",")
    xywh_arr = list(map(float, xywh_arr))
    xywh_arr = list(map(int, xywh_arr))
    print("xywh_arr", xywh_arr)
    roi = (xywh_arr[0], xywh_arr[1], xywh_arr[0] + xywh_arr[2], xywh_arr[1] + xywh_arr[3])
    points_str = g_model.get_points(pil_image, roi)    
    if (points_str is not None):
        return points_str
    else:
        # just return roi
        points_str = str(xywh_arr[0]) + ',' + str(xywh_arr[1])
        points_str += ' '  + str(xywh_arr[0] + xywh_arr[2]) + ',' + str(xywh_arr[1])
        points_str += ' ' + str(xywh_arr[0] + xywh_arr[2]) + ',' + str(xywh_arr[1] + xywh_arr[3])
        points_str += ' ' + str(xywh_arr[0]) + ',' + str(xywh_arr[1] + xywh_arr[3])
        return points_str

if __name__ == "__main__":
    working_directory = '/workspace-test-v1/mlapp/working_directory'
    setup()
    # input_img = Image.open(os.path.join(working_directory, 'images', 'IMG_0839.jpg'))
    # xywh = "509,162.21127319335938,1797.5657043457031,1681.7004699707031"    
    input_img = Image.open(os.path.join(working_directory, 'images', 'IMG_0603.jpg'))
    xywh = "241.64981079101562,195.31297302246094,384.0051574707031,294.62464904785156"
    print('file', input_img)
    
    print("input_img", input_img.size)       
    points_str = get_points_str(input_img, xywh)
    print("points_str", points_str)