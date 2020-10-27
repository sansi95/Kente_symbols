# import image_slicer
import os
import cv2
import os
from PIL import Image
from image_slicer import slice

files = os.listdir('images/testing/')
j = ''
for i in files:
    if i.endswith('jpg'):
        j = i
    else:
        if i.endswith('png'):
            j = i
image = 'images/testing/' + str(j)
image = Image.open(image)
greyscale_img = image.convert(mode="1", dither=Image.NONE)
file = 'images/testing/' + str(j)
greyscale_img.save(file)
image = 'images/testing/' + str(j)
slice(image, 7)

os.remove(image)

files = os.listdir('images/testing/')
for i in files:
    if i.endswith('png') or i.endswith('jpg'):
        os.system("python3 recognize.py --training images/training --testing images/testing")
