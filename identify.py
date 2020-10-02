import image_slicer
import os
import cv2
import os
from PIL import Image

# files = os.listdir('images/testing/')
# for i in files:
#     if i.endswith('jpg'):
#         image = i
#     elif i.endswith('png'):
#         image = i
    # image = 'images/testing/' + str(image)
    # image = Image.open(image)
    # greyscale_img = image.convert(mode="1", dither=Image.NONE)
    # file = 'images/testing/' + str(image)
    # greyscale_img.save(file)
# image = 'images/testing/' + str(image)
# image_slicer.slice(image, 7)

files = os.listdir('images/testing/')
for i in files:
    if i.endswith('png') or i.endswith('jpg'):
        os.system("python3 recognize.py --training images/training --testing images/testing")
