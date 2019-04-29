import cv2
import glob
import math
stride = [0.97,0.96,0.94,0.93]

root_path = '/Users/peter/data/sinho/train'
image_path = glob.glob(root_path+'/*')
for path in image_path:
    image_list = glob.glob(path+'/*')
    if(len(image_list)<80):
        for image in image_list:
            img = cv2.imread(image)
            y,x = img.shape[0:2]
            for i in range(0,len(stride)):
                cropped = img[0:int(y*stride[i]),0:int(x*stride[i])]  # (left, upper, right, lower)
                image_name = image.split('/')[-1].split('.')[0]
                suffix_name = image.split('/')[-1].split('.')[1]
                img_path = path+"/"+image_name+'_aug_'+str(stride[i])+'.jpg'
                try:
                    cv2.imwrite(img_path,cropped)
                except Exception:
                    print('文件出错')
