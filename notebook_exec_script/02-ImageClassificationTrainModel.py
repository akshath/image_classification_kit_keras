#!/usr/bin/env python
# coding: utf-8

# # Image Classification Train Model

# #### Project Url: https://github.com/akshath/image_classification_kit_keras

# In[1]:


is_google_colab = False
is_azure_ml = False


# In[2]:


if is_google_colab or is_azure_ml:
    #!conda create -n img_class_ml_env python=3.7
    #!conda activate img_class_ml_env
    get_ipython().system('pip install -r requirements.txt')
    #!pip install ipykernel
    #!python -m ipykernel install --user --name img_class_ml_env --display-name "img_class_ml_env"


# In[3]:


#print system info
import sys
import numpy
from tensorflow import keras
import tensorflow as tf

print('Python: ',sys.version)
print('numpy: ', numpy.version.version)
print('tensorflow: ', tf.__version__)
print('keras: ', keras.__version__)


# In[4]:


## If you are using the data by mounting the google drive, use the following :
if is_google_colab:
    from google.colab import drive
    drive.mount('/content/gdrive')
##Ref:https://towardsdatascience.com/downloading-datasets-into-google-drive-via-google-colab-bcb1b30b0166


# In[5]:


#get path for pycode folder 
if is_google_colab:
    #!ls '/content/gdrive/MyDrive/Colab Notebooks/' #todo - change me
    get_ipython().system("ls '/content/gdrive/MyDrive/ColabNotebooks/' #todo - change me")


# In[6]:


#if .py files are in Google Drive
if is_google_colab:
    import sys
    #sys.path.append('/content/gdrive/MyDrive/Colab Notebooks/') #todo - change me
    sys.path.append('/content/gdrive/MyDrive/ColabNotebooks/') #todo - change me


# In[7]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
import PIL
import zipfile

from PIL import Image


# In[8]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[10]:


#check for GPU
#print('tf gpu: ',tf.test.is_gpu_available())
print('tf gpu: ',tf.config.list_physical_devices('GPU'))
for x in tf.config.list_physical_devices():
    print('device: ',x)


# In[11]:


import pycode.FileIOUtil
import pycode.TTSUtil
from pycode.ImageClassificationCfg import ImageClassificationCfg
from pycode.RandomColorDistortion import RandomColorDistortion


# # Read Cfg

# In[12]:


import os


# In[13]:


if os.name == 'nt':
    get_ipython().system('cd')
    get_ipython().system('dir')
else:
    get_ipython().system('pwd')
    get_ipython().system('ls')


# In[14]:


if is_google_colab:
    #locate cfg file in google drive
    get_ipython().system("ls '/content/gdrive/MyDrive/ColabNotebooks/project/flowers-recognition/' #change me")


# In[15]:


cfg_file = './project/flowers-recognition/cfg-localsys.yml'
#cfg_file = './project/work_pose/cfg.yml'
#cfg_file = './project/home_presence/cfg.yml'

if is_google_colab:
    cfg_file = '/content/gdrive/MyDrive/ColabNotebooks/project/flowers-recognition/cfg.yml' #change me

cfg_file = ImageClassificationCfg(cfg_file)
cfg_file.load()


# In[16]:


cfg_file.log_info()


# In[17]:


working_dir_str = cfg_file.project_data_dir


# In[18]:


#!ls $working_dir_str
pycode.FileIOUtil.print_dir(working_dir_str, only_dir=True)


# In[19]:


get_ipython().system('ls $working_dir_str')


# In[20]:


path_to_zip_file = None
#path_to_zip_file = '/content/gdrive/MyDrive/ColabNotebooks/project/flowers-recognition/flowers-recognition.zip'
#path_to_zip_file = '/content/gdrive/MyDrive/ColabNotebooks/work_pose/work_pose.zip'
#path_to_zip_file = '/content/gdrive/MyDrive/ColabNotebooks/home_presence/home_presence.zip'
if path_to_zip_file is not None:
    get_ipython().system('ls $path_to_zip_file')


# In[21]:


if path_to_zip_file is not None:
    #already_extracted = False
    already_extracted = False if len(pycode.FileIOUtil.get_dir(working_dir_str))==0 else True
    print('already_extracted: ',already_extracted)
    if already_extracted==False:  
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(working_dir_str)


# In[22]:


#run only for flowers data set
#!mv ./project/flowers-recognition/data/flowers/* ./project/flowers-recognition/data/
#!rmdir ./project/flowers-recognition/data/flowers


# In[23]:


#!ls $working_dir_str
pycode.FileIOUtil.print_dir(working_dir_str, only_dir=True)


# In[24]:


if os.name == 'nt':
    get_ipython().system("rmdir /Q /S \\'$working_dir_str'__MACOSX'\\'")
else:
    get_ipython().system("rm -r $working_dir_str'__MACOSX'")


# In[25]:


rand_seed = 30
np.random.seed(rand_seed)
import random as rn
rn.seed(rand_seed)
tf.random.set_seed(rand_seed)


# # Create Data Set

# In[26]:


batch_size = 256

color_mode="rgb"
#color_mode="grayscale"


# In[27]:


print('working_dir_str: ', working_dir_str)
#!ls $working_dir_str
pycode.FileIOUtil.print_dir(working_dir_str, only_dir=True)


# In[28]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    working_dir_str, 
    label_mode='categorical',
    batch_size=batch_size, image_size=(cfg_file.cfg['input_img_height'], cfg_file.cfg['input_img_width']), 
    shuffle=True, 
    seed=rand_seed, validation_split=0.2, subset='training', color_mode=color_mode
)


# In[29]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    working_dir_str, 
    label_mode='categorical',
    batch_size=batch_size, image_size=(cfg_file.cfg['input_img_height'], cfg_file.cfg['input_img_width']), 
    shuffle=True, 
    seed=rand_seed, validation_split=0.2, subset='validation', color_mode=color_mode
)


# In[30]:


class_names = train_ds.class_names
print(class_names)


# In[31]:


num_classes = len(train_ds.class_names)
num_classes


# ### Visualize the data

# In[35]:


#see few sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(3*3):
        if i >= len(images):
            continue
        ax = plt.subplot(3, 3, i + 1)
        if color_mode=="grayscale":      
            plt.imshow(images[i].numpy().astype("uint8")[:, :, 0], cmap='gray') #
        else:
            plt.imshow(images[i].numpy().astype("uint8"))
        score = tf.nn.softmax(labels[i])
        class_i = np.argmax(score)
        plt.title( class_names[class_i] )
        plt.axis("off")
plt.show()


# In[36]:


for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


# In[37]:


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ### Data Augumentation

# In[38]:


print('input_img_height:',cfg_file.cfg['input_img_height'],)
print('input_img_width:',cfg_file.cfg['input_img_width'])


# In[39]:


#reload cfg
#cfg_file.load()


# In[40]:


#https://towardsdatascience.com/writing-a-custom-data-augmentation-layer-in-keras-2b53e048a98
randomColorDistortion = RandomColorDistortion()
randomColorDistortion.update_cfg(cfg_file)

data_augmentation = Sequential(
    [
        layers.experimental.preprocessing.RandomRotation(
            tuple(cfg_file.cfg['train_augumentation']['random_rotation']),#(-0.03,0.03), #3% random rotation        
            input_shape=(cfg_file.cfg['input_img_height'], cfg_file.cfg['input_img_width'], 1 if color_mode=="grayscale" else 3)), 
        
        layers.experimental.preprocessing.RandomZoom(
            tuple(cfg_file.cfg['train_augumentation']['random_zoom']) #(-0.05,0)  #5% random zoom-in
        ),       
        
        randomColorDistortion,
    ])

if cfg_file.cfg['train_augumentation']['random_flip'] != 'none':
    if cfg_file.cfg['train_augumentation']['random_flip']=='horizontal':
      data_augmentation.add(tf.keras.layers.RandomFlip(mode='horizontal'))
    elif cfg_file.cfg['train_augumentation']['random_flip']=='vertical':
      data_augmentation.add(tf.keras.layers.RandomFlip(mode='vertical'))
    elif cfg_file.cfg['train_augumentation']['random_flip']=='horizontal_and_vertical':
      data_augmentation.add(tf.keras.layers.RandomFlip(mode='horizontal_and_vertical'))
    else:
      print('unknown random_flip value!',cfg_file.cfg['train_augumentation']['random_flip'])


# In[41]:


# visualize how your augmentation strategy works for one instance of training image.
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    img_index = 1
    for i in range(3*3):
        augmented_images = data_augmentation(images, training=True)
        ax = plt.subplot(3, 3, i + 1)
        if color_mode=="grayscale":      
            plt.imshow(augmented_images[img_index].numpy().astype("uint8")[:, :, 0], cmap='gray') #
        else:
            plt.imshow(augmented_images[img_index].numpy().astype("uint8"))
        plt.axis("off")
plt.show()


# # Model

# In[42]:


#reload cfg
#cfg_file.load()


# In[43]:


filepath = cfg_file.cfg['model_file']
print('filepath:',filepath)

print('model_to_try: ',cfg_file.cfg['model_to_try'])
print('model_base: ',cfg_file.cfg['model_base'])

print('train_freeze_base_layer: ', cfg_file.cfg['train_freeze_base_layer'])
print('train_freeze_skip_last_layers: ', cfg_file.cfg['train_freeze_skip_last_layers'])


# In[44]:


# Callbacks

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', save_freq="epoch")

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=4, cooldown=1) # write the REducelronplateau code here

ES = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode="auto")

callbacks_list = [checkpoint, LR, ES]


# In[45]:


#model_to_try = 1 #base cnn
#model_to_try = 2 #transfer learning cnn
model_to_try = cfg_file.cfg['model_to_try']

print('model_to_try: ', model_to_try)


# In[46]:


if model_to_try==1:
    #model - bare cnn
    cnn_model = Sequential([
        data_augmentation, 
        #tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.Normalization(), 

        Conv2D(16, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),

        Flatten(),

        Dense(64, activation='relu'),
        Dropout(0.50),
        BatchNormalization(),

        Dense(64, activation='relu'),
        Dropout(0.50),
        BatchNormalization(),

        Dense(num_classes, activation='softmax')])


# In[47]:


#model_base: ResNet50, ResNet50V2, VGG16, VGG19, MobileNet
try:
    model_base = cfg_file.cfg['model_base']
except KeyError:
    model_base = 'ResNet50V2'

if model_base == 'ResNet50':
    from tensorflow.keras.applications.resnet50 import ResNet50
elif model_base == 'ResNet50V2':
    from tensorflow.keras.applications.resnet_v2 import ResNet50V2
elif model_base == 'VGG16':
    from tensorflow.keras.applications.vgg16 import VGG16
elif model_base == 'VGG19':
    from tensorflow.keras.applications.vgg19 import VGG19
elif model_base == 'MobileNet':
    from tensorflow.keras.applications.mobilenet import MobileNet


# In[48]:


#model - transfer learning
if model_to_try==2:
    if model_base == 'ResNet50':
        conv_base = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(cfg_file.cfg['input_img_height'], cfg_file.cfg['input_img_width'], 1 if color_mode=="grayscale" else 3))
    elif model_base == 'ResNet50V2':
        conv_base = ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=(cfg_file.cfg['input_img_height'], cfg_file.cfg['input_img_width'], 1 if color_mode=="grayscale" else 3))
    elif model_base == 'VGG16':
        conv_base = VGG16(
            include_top=False,
            weights='imagenet',
            input_shape=(cfg_file.cfg['input_img_height'], cfg_file.cfg['input_img_width'], 1 if color_mode=="grayscale" else 3))
    elif model_base == 'VGG19':
        conv_base = VGG19(
            include_top=False,
            weights='imagenet',
            input_shape=(cfg_file.cfg['input_img_height'], cfg_file.cfg['input_img_width'], 1 if color_mode=="grayscale" else 3))
    elif model_base == 'MobileNet':
        conv_base = MobileNet(
            include_top=False,
            weights='imagenet',
            input_shape=(cfg_file.cfg['input_img_height'], cfg_file.cfg['input_img_width'], 1 if color_mode=="grayscale" else 3))
    
    print(model_base,'layer count',len(conv_base.layers))
    #print existing layers in base model
    #for i in range (len(conv_base.layers)):
    #    print (i,conv_base.layers[i])    

    if cfg_file.cfg['train_freeze_base_layer']:
        # freeze all the weights of the model except the last N layers
        for layer in conv_base.layers[:cfg_file.cfg['train_freeze_skip_last_layers']*-1]:
            layer.trainable = False

    cnn_model = Sequential([
        data_augmentation, 
        #tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
        tf.keras.layers.experimental.preprocessing.Normalization(), 
        conv_base, 

        #MaxPooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),
        Flatten(),
        BatchNormalization(),

        #--- vs --
        Dense(8, activation='relu'),
        Dropout(0.40),#50
        BatchNormalization(),
        #--- s --

        #--- s --
        #Dense(256, activation='relu'),
        #Dropout(0.50),#40
        #BatchNormalization(),

        #Dense(128, activation='relu'),
        #Dropout(0.50),#40
        #BatchNormalization(),

        #Dense(64, activation='relu'),
        #Dropout(0.50),#40
        #BatchNormalization(),
        #--- s --

        #--- l --
        #Dense(512, activation='relu'),
        #Dropout(0.60),#40
        #BatchNormalization(),

        #Dense(256, activation='relu'),
        #Dropout(0.60),
        #BatchNormalization(),

        #Dense(128, activation='relu'),
        #Dropout(0.60),
        #BatchNormalization(),

        #Dense(64, activation='relu'),
        #Dropout(0.10),
        #BatchNormalization(),
        #--- l --

        Dense(num_classes, activation='softmax')
        ])


# In[49]:


lr = 0.0001
optimiser = keras.optimizers.Adam(learning_rate=lr)
cnn_model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])
print (cnn_model.summary())


# In[50]:


### Train the model
epochs = 10
history = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks_list,
    initial_epoch = 0
)


# ### Visualizing training results

# In[51]:


# Visualizing training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

#epochs_range = range(num_epochs)
epochs_range = range(len(val_acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[52]:


pycode.TTSUtil.speak('We got validation accuracy of '+str(round(history.history['val_accuracy'][-1]*100,2)), ipython=True)


# In[53]:


get_ipython().system('ls $cfg_file.project_dir')


# In[54]:


#save json model file also
from keras.models import model_from_json

model_json = cnn_model.to_json()
#print('model_json: ',model_json)

with open(cfg_file.project_dir+cfg_file.cfg["project_name"]+".json", "w") as json_file:
    json_file.write(model_json)


# In[55]:


#todo add code to see what failed most
#run through val set and log case wise no and accuracy and log image of good and bad


# In[ ]:





# In[ ]:




