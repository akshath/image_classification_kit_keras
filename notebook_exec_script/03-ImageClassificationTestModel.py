#!/usr/bin/env python
# coding: utf-8

# # Image Classification Test Model

# #### Project Url: https://github.com/akshath/image_classification_kit_keras

# In[1]:


is_google_colab = False
is_azure_ml = False


# In[2]:


3 ## If you are using the data by mounting the google drive, use the following :
if is_google_colab:
    from google.colab import drive
    drive.mount('/content/gdrive')
##Ref:https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166


# In[3]:


#get path for pycode folder 
if is_google_colab:
    #google_colab_notebook_loc = '/content/gdrive/MyDrive/Colab Notebooks/'
    google_colab_notebook_loc = '/content/gdrive/MyDrive/ColabNotebooks/'
    get_ipython().system('ls $google_colab_notebook_loc')


# In[4]:


if is_google_colab:
    #!conda create -n img_class_ml_env python=3.7
    #!conda activate img_class_ml_env
    get_ipython().system("pip install -r $google_colab_notebook_loc'requirements.txt'")
    #!pip install ipykernel
    #!python -m ipykernel install --user --name img_class_ml_env --display-name "img_class_ml_env"


# In[5]:


#print system info
import sys
import numpy
from tensorflow import keras
import tensorflow as tf

print('Python: ',sys.version)
print('numpy: ', numpy.version.version)
print('tensorflow: ', tf.__version__)
print('keras: ', keras.__version__)


# In[6]:


#if .py files are in Google Drive
if is_google_colab:
    import sys
    sys.path.append(google_colab_notebook_loc)


# In[7]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import time
from datetime import datetime
import random
from glob import glob
import os


# In[8]:


import cv2
from PIL import Image


# In[9]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# ## Functions

# In[11]:


import pycode.FileIOUtil
import pycode.TTSUtil
from pycode.ImageClassificationCfg import ImageClassificationCfg
from pycode.LoadAndTestCNNUtil import LoadAndTestCNNUtil


# In[12]:


#function to take pic from webcam on browser for Google Collab kinda env
if is_google_colab:
    from IPython.display import display, Javascript
    from google.colab.output import eval_js
    from base64 import b64decode

    def take_photo_over_webpage(filename='photo.jpg', quality=0.8):
        js = Javascript('''
            async function takePhoto(quality) {
                const div = document.createElement('div');
                const capture = document.createElement('button');
                capture.textContent = 'Capture';
                div.appendChild(capture);

                const video = document.createElement('video');
                video.style.display = 'block';
                const stream = await navigator.mediaDevices.getUserMedia({video: true});

                document.body.appendChild(div);
                div.appendChild(video);
                video.srcObject = stream;
                await video.play();

                // Resize the output to fit the video element.
                google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

                // Wait for Capture to be clicked.
                await new Promise((resolve) => capture.onclick = resolve);

                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0);
                stream.getVideoTracks()[0].stop();
                div.remove();
                return canvas.toDataURL('image/jpeg', quality);
            }
          ''')
        display(js)
        data = eval_js('takePhoto({})'.format(quality))
        binary = b64decode(data.split(',')[1])
        with open(filename, 'wb') as f:
            f.write(binary)
        return filename


# # Read Cfg

# In[13]:


if is_google_colab:
  #locate cfg file in google drive
  get_ipython().system("ls $google_colab_notebook_loc'project/flowers-recognition/' #change me")


# In[14]:


cfg_file = './project/flowers-recognition/cfg-localsys.yml'
#cfg_file = './project/work_pose/cfg.yml'
#cfg_file = './project/home_presence/cfg.yml'

if is_google_colab:
    cfg_file = google_colab_notebook_loc+'project/flowers-recognition/cfg.yml' #change me

cfg_file = ImageClassificationCfg(cfg_file)
cfg_file.load()


# In[15]:


cfg_file.log_info()


# In[16]:


def get_capture_dim(src_video):
    cap = None
    try:
        cap = cv2.VideoCapture(src_video)
        if(cap.isOpened()):
            ret, frame = cap.read()
            return frame.shape
        else:
            return 0,0
    finally:
        if cap!=None:
            cap.release()
        cv2.destroyAllWindows()


# In[17]:


#org shape 1080, 1920
capture_dim = get_capture_dim(cfg_file.cfg['src_video'])
print('org dim: ',capture_dim)

img_h = capture_dim[0]
img_w = capture_dim[1]

img_h = img_h//cfg_file.cfg['reduce_image_wh_by']
img_w = img_w//cfg_file.cfg['reduce_image_wh_by']

print('new dim h:',img_h)
print('new dim w:',img_w)


# In[18]:


def plot_image(images, captions=None, cmap=None):    
    if captions!=None:
        print(captions)
    
    if len(images) > 1:
        f, axes = plt.subplots(1, len(images), sharey=True, figsize=(4,4))
        f.set_figwidth(15)
        for ax,image in zip(axes, images):
            ax.imshow(image, cmap)
    else:
        plt.figure(figsize=(4,4))
        plt.imshow(images[0])
    plt.show()


# In[19]:


def get_file_name():
    return 'img_'+datetime.now().strftime("%Y%m%d%H%M%S")
#get_file_name()


# In[20]:


def capture_frames(cfg_file, crop=True, count=1, delay_sec=60, save=True):
    cap = None
    if img_h==0 or img_w==0:
        raise Exception('img_h or img_w can not be 0')
    try:
        cap = cv2.VideoCapture(cfg_file.cfg['src_video'])
        #cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if(cap.isOpened()):
            for i in range(0, count):
                ret, frame = cap.read()
                if not ret:
                    print("failed to grab frame ")
                    #raise Exception("failed to grab frame ")
                    return None, None

                #print('org shape: ',frame.shape)
                frame = cv2.resize(frame, (img_w,img_h), 
                                   interpolation = cv2.INTER_AREA)

                if crop:
                    if cfg_file.cfg['crop_image_from_left']>0:
                        new_wl = int(img_w*cfg_file.cfg['crop_image_from_left'])
                    else:
                        new_wl = 0
                        
                    if cfg_file.cfg['crop_image_from_right']>0:
                        #crop 70% on width from right
                        new_wr = int(img_w*cfg_file.cfg['crop_image_from_right'])
                    else:
                        new_wr = img_w                    
                        
                    frame = frame[0:img_h,new_wl:new_wr]

                if save:
                    img_name = cfg_file.loc_unknown+"frame_{0}.{1}".format(get_file_name(), cfg_file.cfg['file_ext'][2:])
                    cv2.imwrite(img_name, frame)
                else:
                    img_name = None

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if count>1 and save:
                    print('file saved: ',img_name)
                    
                if (i+1)!=count:
                    #close and re-open else we will get old frame
                    if cap != None:
                        cap.release()
                    #it take about 2 sec to open cam again
                    if delay_sec>0:
                        time.sleep(delay_sec-2) 
                    cap = cv2.VideoCapture(cfg_file.cfg['src_video'])
                    if(cap.isOpened()==False):
                        print('Could not open camera!')
                        break            
            return frame, img_name
        else:
            print('Could not open camera!')
            return None, None
    finally:
        if cap != None:
            cap.release()
        cv2.destroyAllWindows()


# # Load Model

# In[21]:


#cfg_file.load() #re-load
cfg_file.cfg['model_file']


# In[22]:


cnn = LoadAndTestCNNUtil(cfg_file)
cnn.debug = True
cnn.load()


# # Load and Test Image

# In[23]:


print('project_data_dir: ',cfg_file.project_data_dir)


# In[24]:


all_images = glob(cfg_file.project_data_dir+'*/'+cfg_file.cfg['file_ext'])
print('all img count: ',len(all_images))


# In[25]:


frame = None
file = None

file = random.choice(all_images)
#frame, file = capture_frames(cfg_file=cfg_file,crop=True, count=1, delay_sec=0, save=True)
#file = google_colab_notebook_loc+'project/flowers-recognition/data/sunflower/6133988570_9dc778e622_m.jpg'
if is_google_colab==True and False:
    try:
        file = cfg_file.loc_unknown+"frame_{0}.{1}".format(get_file_name(), cfg_file.cfg['file_ext'][2:])
        take_photo_over_webpage(file)

        frame = cv2.imread(file)
        w = cfg_file.cfg['input_img_width']
        h = cfg_file.cfg['input_img_height']
        frame = cv2.resize(frame, (w,h), interpolation = cv2.INTER_AREA)
        cv2.imwrite(file, frame)
        frame = None
    except Exception as err:
        # Errors will be thrown if the user does not have a webcam or if they do not
        # grant the page permission to access it.
        print(str(err))

print('file: ',file)
if frame is None:
    frame = Image.open(file)

plot_image([frame])

(img_class, model_acc) = cnn.predict(file)


# In[26]:


#delete temp file
if file.startswith(cfg_file.loc_unknown):    
    if os.path.isfile(file):
        print('Deleted temp file')
        os.remove(file)
    else:
        #no file to delete
        pass


# In[ ]:





# In[ ]:




