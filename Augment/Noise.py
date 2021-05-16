import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import random
from pathlib import Path
def read_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
image_paths = list(Path('./glaucoma').iterdir())  #load images from folder 
images = [read_image(p) for p in image_paths]
for i in range(len(images)): 
        print(images[i].shape, images[i].dtype)
fig = plt.figure(figsize=(20,20))
columns = 1
rows = 1
noise_img = []
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

for i in range(len(images)):
    noise = sp_noise(images[i],0.001)
    noise_img.append(noise)
    
for i in range(len(noise_img)):
    cv2.imwrite('./result_sample/_noise_'+str(i)+'.jpg', noise_img[i]) #save new filename image to new folder
    
pics = []
for i in range(columns*rows):
    pics.append(fig.add_subplot(rows, columns, i+1,title=image_paths[i].parts[-1].split('.')[0]))
    plt.imshow(noise_img[i])
    
plt.show()