import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import random
from pathlib import Path
def read_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
image_paths = list(Path('./testdata2/Normal').iterdir())  #load images from folder 
images = [read_image(p) for p in image_paths]
for i in range(len(images)): 
        print(images[i].shape, images[i].dtype)
fig = plt.figure(figsize=(20,20))
columns = 1
rows = 1
height = 256
width = 256
dim = (width, height)
res_img = []
for i in range(len(images)):
    res = cv2.resize(images[i], dim, interpolation=cv2.INTER_LINEAR)
    res_img.append(res)
    
for i in range(len(res_img)):
    cv2.imwrite('./test_pic/Normal_256/normal_resized256_'+str(i)+'.jpg', res_img[i]) #save new filename image to new folder
    
pics = []
for i in range(columns*rows):
    pics.append(fig.add_subplot(rows, columns, i+1,title=image_paths[i].parts[-1].split('.')[0]))
    plt.imshow(res_img[i])
    
plt.show()