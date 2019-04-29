import os
import utils
import numpy as np
import cv2
import os
import random

#path = '/home/pinlan/AiLi/Aili_classification/classification'
path = '/Users/peter/data/Aili_classification/classification'

def getpictures(mount):
    image = np.zeros((mount*25, 224, 224, 3))
    label = np.zeros((25*mount, 25))
    print(label.shape)
    class_flag = 0
    for class_path in os.listdir(path):
        image_list = getdirdata(mount,class_path)
        j = 0
        for img in image_list:
            full_path = os.path.join(path+'/'+class_path,img)
            Img = utils.ImageEncode(full_path)
            image[mount*class_flag+j,:,:,:] = Img[0,:,:,:]
            label[mount*class_flag+j,class_flag] = 1
            j = j+1
            if(j==mount):
                break
        class_flag = class_flag+1
    return image, label

def getdirdata(mount,pardir):
    i = 0
    image_list = []
    for img in os.listdir(os.path.join(path,pardir)):
        if(i<mount):
            image_list.append(img)
        i = i+1
    return image_list

if __name__ == '__main__':
    getpictures(200)