#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import YoloHelper


# In[2]:


yolo_net, yolo_output_layers, yolo_classes = YoloHelper.load_yolo('yolov3.weights','yolov3.cfg', 'coco.names')


# In[3]:


img = cv2.imread('test-image.jpg')


# In[4]:


img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

#img_rgb = Image.open('test_img.jpg')
#plt.imshow(img_rgb)
#plt.show()


# In[5]:


outputs, img_box = YoloHelper.detect_obj(yolo_net, yolo_output_layers, yolo_classes, img)

#label,(x,y,w,h),confidences
for output in outputs:
    print(output)


# In[6]:


#plt.figure(figsize=(16,10))
img_rgb = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()


# In[7]:


img_crop = YoloHelper.crop_output(img, outputs[0], 10)

img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
plt.imshow(img_crop_rgb)
plt.show()


# In[8]:


img_crop = YoloHelper.crop_output(img, outputs[1], 10)

img_crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
plt.imshow(img_crop_rgb)
plt.show()


# In[9]:


outputs, img_box = YoloHelper.detect_obj(yolo_net, yolo_output_layers, yolo_classes, img, only=['person'])

#label,(x,y,w,h),confidences
for output in outputs:
    print(output)
    
#plt.figure(figsize=(16,10))
img_rgb = cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()


# In[10]:


plt.imshow(img_crop_rgb)
plt.show()


# In[11]:


img_crop_rgb_fill = YoloHelper.make_image_square(img_crop)

img_crop_rgb_fill_rgb = cv2.cvtColor(img_crop_rgb_fill, cv2.COLOR_BGR2RGB)
plt.imshow(img_crop_rgb_fill_rgb)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




