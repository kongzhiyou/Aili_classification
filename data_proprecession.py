import os
import utils
import numpy as np
import cv2
import os
import random
import PIL.Image as Image

path = '/home/pinlan/Aili_classification/classification'
#path = '/Users/peter/data/AI/classification'
class_dict = {}


def getpictures(mount):
    while True:
        image = np.zeros((mount, 224, 224, 3))
        label = np.zeros((mount, 25))
        class_flag = 0
        count = 0
        for class_path in os.listdir(path):
            image_list = getdirdata(mount,class_path)
            num = random.randint(1,3)
            if count < mount:
                for z in range(0,num):
                    cnt = len(image_list)
                    target = random.randint(0,cnt)
                    img = image_list[target]
                    full_path = os.path.join(path+'/'+class_path,img)

                    Img = utils.ImageEncode(full_path)

                    image[count,:,:,:] = Img[0,:,:,:]
                    label[count,class_flag] = class_flag
                    count += 1
                    print(count)
                    if count == mount:
                        break
                class_flag += 1
            else:
                count = 0
                yield (image,label)
def getdirdata(mount,pardir):
    i = 0
    image_list = []
    for img in os.listdir(os.path.join(path,pardir)):
        if(i<mount):
            image_list.append(img)
        i = i+1
    return image_list


getpictures(64)