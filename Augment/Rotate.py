import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import imutils
from pathlib import Path
from scipy import ndimage
def read_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
image_paths = list(Path('./glaucoma').iterdir()) 
images = [read_image(p) for p in image_paths]
for i in range(len(images)):
        print(images[i].shape, images[i].dtype)
fig = plt.figure(figsize=(20,20))
columns = 4
rows = 4
angle = 20

rotated_img = []
for i in range(len(images)):
    rotated = imutils.rotate_bound(images[i], angle)
    rotated_img.append(rotated)

for i in range(len(rotated_img)):
    cv2.imwrite('./glaucoma_rotated_20/glaucoma_20_rotated_'+str(i)+'.jpg', rotated_img[i]) #save new filename image to new folder

pics = []
for i in range(columns*rows):
    pics.append(fig.add_subplot(rows, columns, i+1,title=image_paths[i].parts[-1].split('.')[0]))
    plt.imshow(rotated_img[i])
    
plt.show()