#!/usr/bin/env python
# coding: utf-8

# # Image Classification Data Collector

# #### Project Url: https://github.com/akshath/image_classification_kit_keras

# In[1]:


is_google_colab = False
is_azure_ml = False


# In[2]:


## If you are using the data by mounting the google drive, use the following :
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


import os
import time
import random
import shutil

from glob import glob
from pathlib import Path
from datetime import datetime

from gtts import gTTS
from playsound import playsound


from IPython.display import clear_output


# In[8]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[10]:


import cv2
from PIL import Image


# ## Functions

# In[11]:


import pycode.FileIOUtil
import pycode.TTSUtil
from pycode.ImageClassificationCfg import ImageClassificationCfg
from pycode.LoadAndTestCNNUtil import LoadAndTestCNNUtil


# In[12]:


def get_capture_dim(src_video):
    try:
        cap = cv2.VideoCapture(src_video)
        if(cap.isOpened()):
            ret, frame = cap.read()
            return frame.shape
        else:
            return 0,0
    finally:
        cap.release()
        cv2.destroyAllWindows()


# In[13]:


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


# In[14]:


def get_file_name():
    return 'img_'+datetime.now().strftime("%Y%m%d%H%M%S")+str(random.randrange(10, 101, 2))
#get_file_name()


# In[15]:


def print_label_select(cfg_file):
    for i in range(len(cfg_file.labels)):
        print(f'[{labels_id[i]}] {cfg_file.labels[i]}')
    sel = input('Select ([d] for del): ')
    return sel


# In[16]:


def save_file_unknown(cfg_file, frame):
    file_name = cfg_file.loc_unknown+"frame_{0}.{1}".format(get_file_name(), cfg_file.cfg['file_ext'][2:])
    cv2.imwrite(file_name, frame)
    #print('save_file_unknown',file_name)
    return file_name


# In[17]:


def save_frames(cfg_file, h,w, crop=True, count=1, delay_sec=60):
    cap = None
    yolo_outputs, yolo_img_box = None, None
    if h==0 or w==0:
        raise Exception('h or w can not be 0')
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
                frame = cv2.resize(frame, (w,h), interpolation = cv2.INTER_AREA)

                if crop:
                    if cfg_file.cfg['crop_image_from_left']>0:
                        new_wl = int(w*cfg_file.cfg['crop_image_from_left'])
                    else:
                        new_wl = 0
                        
                    if cfg_file.cfg['crop_image_from_right']>0:
                        #crop 70% on width from right
                        new_wr = int(w*cfg_file.cfg['crop_image_from_right'])
                    else:
                        new_wr = w                    
                        
                    frame = frame[0:h,new_wl:new_wr]
                    
                img_name = save_file_unknown(cfg_file, frame)
                
                #do yolo
                if yolo_net!=None:
                    #print('doing yolo')
                    yolo_outputs, yolo_img_box = yolo_v3.YoloHelper.detect_obj(
                        yolo_net, yolo_output_layers, yolo_classes, frame, only=cfg_file.cfg['yolo_filter_only'].split())
                    
                    for yolo_output in yolo_outputs:
                        img_crop = yolo_v3.YoloHelper.crop_output(frame, yolo_output, cfg_file.cfg['yolo_object_padding'])
                        if cfg_file.cfg['yolo_resize_as_square']:
                            img_crop = yolo_v3.YoloHelper.make_image_square(img_crop)
                        crop_file = save_file_unknown(cfg_file, img_crop)
                        yolo_output.append(crop_file)
                    os.remove(img_name)
                    print('Saved',len(yolo_outputs),'objects',)
                else:
                    if count>1:
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
                        
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if yolo_img_box is not None:
                yolo_img_box = cv2.cvtColor(yolo_img_box, cv2.COLOR_BGR2RGB)
            
            return frame, img_name, yolo_outputs, yolo_img_box
        else:
            print('Could not open camera!')
            return None, None, None, None
    finally:
        if cap != None:
            cap.release()
        cv2.destroyAllWindows()


# In[18]:


def show_img_and_ask_label(cfg_file, frame, file):
    plt.imshow(frame)
    plt.show()
    
    if cnn is not None:
        (img_class, model_acc) = cnn.predict(file)
        print('Model: {0} [{1}%]\n'.format(img_class,model_acc))
    
    sel = print_label_select(cfg_file)
    sel_array = sel.split(',')
    if sel_array[0] == 'd':
        print('Selected: Delete')
        os.remove(file)
    else:
        for _sel in sel_array:
            sel = int(_sel)
            file_path = Path(file)                
            new_loc = cfg_file.project_data_dir+cfg_file.labels[sel]+'/'+file_path.parts[-1]
            print('Selected: ',cfg_file.labels[sel])
            #print(new_loc)
            shutil.copy2(file, new_loc)
        os.remove(file)


# In[19]:


def collect_image(cfg_file, count=1, delay_sec=60, collect_label=True):
    if count==0:
        print('min count is 1')
        return
    
    frame, file, yolo_outputs, yolo_img_box = save_frames(cfg_file, img_h, img_w, crop=True, count=count, delay_sec=delay_sec)
    if count==1 and collect_label==True:
        if yolo_outputs is not None:
            plot_image([yolo_img_box])
        elif file!=None:
            show_img_and_ask_label(cfg_file, frame, file)            

    return frame, file, yolo_outputs, yolo_img_box


# In[20]:


def label_non_labeled(cfg_file):
    all_images = glob(cfg_file.loc_unknown+cfg_file.cfg['file_ext'])
    if len(all_images)==0:
        print('Non labled folder is empty!')
    for file in all_images:
        clear_output(wait=True)

        print('file: ',file)    
        frame = Image.open(file)
        show_img_and_ask_label(cfg_file, frame, file)


# In[21]:


#take pic at time set
def take_pics_bw_time(trigger_from, trigger_to, sleep_time_bw_pic = 5, skip_if_size_diff_less_than=-1):
    print('Taking pics from: ',trigger_from,' to: ',trigger_to)
    last_file_size = 0
    while True:
        try:
            time_now = int(datetime.now().strftime("%H%M"))
            
            if time_now>=trigger_from and time_now<=trigger_to:
                last_frame, file, yolo_outputs, yolo_img_box = collect_image(cfg_file, count=1, delay_sec=0, collect_label=False)
                if yolo_outputs is not None:
                    if len(yolo_outputs)>0:
                        #last file
                        file = yolo_outputs[len(yolo_outputs)-1][3]
                        file_size = pycode.FileIOUtil.get_file_size(file)
                        #print('file saved: ',file,', size: ',file_size,'kb')
                        if skip_if_size_diff_less_than > -1:
                            diff = abs(last_file_size - file_size)
                            if diff < skip_if_size_diff_less_than:
                                print('deleting '+file,', size: ',file_size,'kb')
                                os.remove(file)
                        last_file_size = file_size
                elif file != None:
                    file_size = pycode.FileIOUtil.get_file_size(file)
                    #print('file saved: ',file,', size: ',file_size,'kb')
                    if skip_if_size_diff_less_than > -1:
                        diff = abs(last_file_size - file_size)
                        if diff < skip_if_size_diff_less_than:
                            print('deleting '+file,', size: ',file_size,'kb')
                            os.remove(file)
                        else:
                            print('file saved: ',file,', size: ',file_size,'kb')
                    else:
                        print('file saved: ',file,', size: ',file_size,'kb')
                    last_file_size = file_size

                time.sleep(sleep_time_bw_pic)
            else:
                time.sleep(60)
        except KeyboardInterrupt:
            print('Stopping..')
            break


# In[22]:


#img stats functions


# In[23]:


def get_label_stats(label, cfg_file, show_image=True):
    dir_loc = cfg_file.project_data_dir+label+'/'
    all_images = glob(dir_loc+cfg_file.cfg['file_ext'])
    int_len = len(all_images)
    
    str_len = str(len(all_images))
    caption = 'Class: '+label.ljust(20, ' ')+'Count: '+str_len.rjust(3,' ')
    
    if show_image:
        rand_img = []
        max_rand = 5 if int_len>=5 else int_len

        if int_len!=0:
            for i in range(max_rand):
                if(int_len>0):
                    frame = Image.open(random.choice(all_images))
                else:
                    frame = Image.open(random.choice(i))
                rand_img.append(frame)
            plot_image(rand_img, caption)
    else:
        pass
        #print(caption)
    return int_len


# In[24]:


#_ = get_label_stats(labels[0], show_image=True, file_ext='*.png')


# In[25]:


#print image count stats
import sys

def print_img_stats(cfg_file, show_image=True):
    total_img = 0
    class_mis_df = pd.DataFrame(columns=['Class','Count'])
    for i,label in enumerate(cfg_file.labels):
        count = get_label_stats(label, cfg_file, show_image=show_image)
        total_img += count
        
        row = {'Class':label, 'Count':count}
        class_mis_df = class_mis_df[class_mis_df['Class']!=row['Class']] #check to prevent duplicate on re-run
        class_mis_df = class_mis_df.append(row, ignore_index=True) 
    return class_mis_df


# In[26]:


#print 1 random images per class from all class
def show_1image_per_class(cfg_file):
    plt.figure(figsize=(14, 14))
    for i,label in enumerate(cfg_file.labels):
        ax = plt.subplot(3, 3, i + 1)

        dir_loc = cfg_file.project_data_dir+label+'/'
        all_images = glob(dir_loc+cfg_file.cfg['file_ext'])

        if len(all_images)!=0:
            frame = Image.open(random.choice(all_images))
            plt.imshow(frame)
        else:
            #plt.imshow(None)
            pass        
        plt.title(label)

        plt.axis("off")


# In[27]:


def plot_image_stats():
    print('Total Images: ',class_mis_df.Count.sum())
    print('')
    plt.figure(figsize=(8,4))
    sns.barplot(data=class_mis_df, y='Class', x='Count')
    plt.show()


# In[28]:


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

# In[29]:


import os

if os.name == 'nt':
    get_ipython().system('cd')
else:
    get_ipython().system('pwd')


# In[30]:


if is_google_colab:
    #data dir
    get_ipython().system("ls $google_colab_notebook_loc'project/flowers-recognition/'")
    
    #cfg file
    get_ipython().system("ls $google_colab_notebook_loc'project/flowers-recognition/cfg.yml'")


# In[31]:


path_to_zip_file = None
#path_to_zip_file = google_colab_notebook_loc + 'project/flowers-recognition/flowers-recognition.zip'
#path_to_zip_file = google_colab_notebook_loc + 'project/work_pose/work_pose.zip'
#path_to_zip_file = google_colab_notebook_loc + 'project/home_presence/home_presence.zip'

if path_to_zip_file is not None:
    get_ipython().system('ls $path_to_zip_file')


# In[32]:


if path_to_zip_file is not None:
    import zipfile
    #already_extracted = False
    already_extracted = False if len(pycode.FileIOUtil.get_dir(google_colab_notebook_loc+'project/flowers-recognition/'))==0 else True
    print('already_extracted: ',already_extracted)
    if already_extracted==False:  
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(google_colab_notebook_loc+'project/flowers-recognition/')


# In[33]:


if is_google_colab:
    #data dir
    get_ipython().system("ls $google_colab_notebook_loc'project/flowers-recognition/'")
    get_ipython().system("mv $google_colab_notebook_loc'project/flowers-recognition/flowers/' $google_colab_notebook_loc'project/flowers-recognition/data/'")
    get_ipython().system("ls $google_colab_notebook_loc'project/flowers-recognition/'")


# In[34]:


if is_google_colab:
    cfg_file = google_colab_notebook_loc+'project/flowers-recognition/cfg.yml'
elif is_azure_ml:
    print('todo')
else:
    #refer to config_template.yml for example of how it should be  
    cfg_file = './project/flowers-recognition/cfg-localsys.yml'
    #cfg_file = './project/work_pose/cfg.yml'
    #cfg_file = './project/home_presence/cfg.yml'

cfg_file = ImageClassificationCfg(cfg_file)
cfg_file.load()


# In[35]:


cfg_file.log_info()


# In[36]:


if is_google_colab==False and is_azure_ml==False:
    #--- do local camera capture
    #org shape 1080, 1920
    capture_dim = get_capture_dim(cfg_file.cfg['src_video'])
    print('org dim: ',capture_dim)

    img_h = capture_dim[0]
    img_w = capture_dim[1]

    img_h = img_h//cfg_file.cfg['reduce_image_wh_by']
    img_w = img_w//cfg_file.cfg['reduce_image_wh_by']

    print('new dim h:',img_h)
    print('new dim w:',img_w)
elif cfg_file.cfg['labels_from_dir']:
    print('Lets get a sample img from dir and get its size')
    all_images = glob(cfg_file.project_data_dir+'*/'+cfg_file.cfg['file_ext'])  
    sample_image = cv2.imread(random.choice(all_images), 1)
    #print(sample_image.shape)
    #plt.imshow(sample_image)
    #plt.show()
    img_h = sample_image.shape[0]
    img_w = sample_image.shape[1]

    print('new dim h:',img_h)
    print('new dim w:',img_w)
elif is_google_colab==True:
    #Lets get file from webpage
    from IPython.display import Image
    try:
        filename = take_photo_over_webpage()
        print('Saved to {}'.format(filename))

        sample_image = cv2.imread(filename, 1)
        #print(sample_image.shape)
        #plt.imshow(sample_image)
        #plt.show()
        img_h = sample_image.shape[0]
        img_w = sample_image.shape[1]

        print('new dim h:',img_h)
        print('new dim w:',img_w)
    except Exception as err:
        # Errors will be thrown if the user does not have a webcam or if they do not
        # grant the page permission to access it.
        print(str(err))


# In[37]:


if cfg_file.cfg['labels_from_dir']==False:
    if os.path.isdir(cfg_file.project_dir)==False:
        os.mkdir(cfg_file.project_dir)
    if os.path.isdir(cfg_file.project_data_dir)==False:
        os.mkdir(cfg_file.project_data_dir)
    #!ls $project_data_dir
    pycode.FileIOUtil.print_dir(cfg_file.project_data_dir, only_dir=True)

    if os.path.isdir(cfg_file.project_temp_dir)==False:
        os.mkdir(cfg_file.project_temp_dir)
    #!ls $project_temp_dir
    pycode.FileIOUtil.print_dir(cfg_file.project_temp_dir, only_dir=True)
else:
    print('We assume dir is alreay there.')


# In[38]:


if os.path.isdir(cfg_file.project_temp_dir)==False:
    os.mkdir(cfg_file.project_temp_dir)
    
if os.path.isdir(cfg_file.loc_unknown)==False:
    os.mkdir(cfg_file.loc_unknown)
    
for label in cfg_file.labels:
    if os.path.isdir(cfg_file.project_data_dir+label)==False:
        os.mkdir(cfg_file.project_data_dir+label)


# In[39]:


#!ls $project_data_dir
pycode.FileIOUtil.print_dir(cfg_file.project_data_dir, only_dir=True)


# In[40]:


labels_id = [x for x in range(0,len(cfg_file.labels))]
print(labels_id)
print(cfg_file.labels)


# In[41]:


#if model file is there.. lets load it
if cfg_file.cfg['model_file'] != '':
    if os.path.isfile(cfg_file.cfg['model_file']):
        try:
            cnn = LoadAndTestCNNUtil(cfg_file)
            cnn.debug = False
            cnn.load()
            print('modle file loaded')
        except Exception as e:
            print('modle load failed! ',e)
            cnn = None
    else:
        cnn = None
        print('modle file not present')
else:
    cnn = None
    print('modle file not set')


# In[42]:


#if Yolo is enabled.. load it
if cfg_file.cfg['yolo_for_capture']== True:
    import yolo_v3.YoloHelper
    if is_google_colab:
        yolo_net, yolo_output_layers, yolo_classes = yolo_v3.YoloHelper.load_yolo(google_colab_notebook_loc+'yolo_v3/yolov3.weights',
                                                                          google_colab_notebook_loc+'yolo_v3/yolov3.cfg', 
                                                                          google_colab_notebook_loc+'yolo_v3/coco.names')
    else:
        yolo_net, yolo_output_layers, yolo_classes = yolo_v3.YoloHelper.load_yolo('./yolo_v3/yolov3.weights',
                                                                          './yolo_v3/yolov3.cfg', 
                                                                          './yolo_v3/coco.names')


# # Collect Data

# In[43]:


collect_data_type = 5

if collect_data_type == 1:
    #pick 1 quick image and label it
    last_frame, file, yolo_outputs, yolo_img_box = collect_image(cfg_file, count=1, delay_sec=60)
    #print(last_frame.shape)
elif collect_data_type == 2:
    #pick 20 quick image every 
    #2m - 120, 30img = 60m = 1h, 
    #5m - 300, 20img = 1.6h
    #10m = 600, 20img = 3.3h
    #15m - 900, 20 img = 5h
    try:
        last_frame, file, yolo_outputs, yolo_img_box = collect_image(cfg_file, count=40, delay_sec=120)
    except KeyboardInterrupt: 
        print('Stopping..')
elif collect_data_type == 3:
    #pick 60 quick image every 2sec = 2m captured
    try:
        last_frame, file, yolo_outputs, yolo_img_box = collect_image(cfg_file, count=120, delay_sec=3)
    except KeyboardInterrupt:
        print('Stopping..')
elif collect_data_type == 4:
    #take pics b/w time windows, at 5sec interval. save only if size diff is over 2kb
    #take_pics_bw_time(trigger_from=1730, trigger_to=1830, sleep_time_bw_pic = 5, skip_if_size_diff_less_than=3)
    take_pics_bw_time(trigger_from=1113, trigger_to=1500, sleep_time_bw_pic = 5, skip_if_size_diff_less_than=3)
    #take_pics_bw_time(trigger_from=1700, trigger_to=1830, sleep_time_bw_pic = 5, skip_if_size_diff_less_than=3)
elif collect_data_type == 5:
    #label all non labeled images
    label_non_labeled(cfg_file)
else:
    print('Nothing to do!')


# # Data Stats

# In[44]:


class_mis_df = print_img_stats(cfg_file, show_image=False)
class_mis_df


# In[45]:


plot_image_stats()


# In[46]:


_ = print_img_stats(cfg_file, show_image=True)


# In[47]:


show_1image_per_class(cfg_file)


# In[ ]:





# In[ ]:




