
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

from PIL import Image

from tensorflow.keras.utils import CustomObjectScope
from RandomColorDistortion import RandomColorDistortion
from ImageClassificationCfg import ImageClassificationCfg

class LoadAndTestCNNUtil:    

    def __init__(self, imageClassificationCfg):
        self.class_names = imageClassificationCfg.cfg['labels']
        self.model_file_h5 = imageClassificationCfg.cfg['model_file']
        self.num_classes = len(self.class_names)
        self.img_height = imageClassificationCfg.cfg['input_img_height']
        self.img_width = imageClassificationCfg.cfg['input_img_width']

        self.debug = True
        self.use_pil = False
            
    def load(self, file=None):        
        if file is None:
            file = self.model_file_h5
        print('Loading model file: ',file)
        
        with CustomObjectScope({'RandomColorDistortion': RandomColorDistortion}):
            self.model = tf.keras.models.load_model(file)

        if self.debug:
            print(self.model.summary())
        
    def predict(self, file):
        img_size = (self.img_height,self.img_width)
        
        if self.use_pil==False:
            frame = keras.preprocessing.image.load_img(file, target_size=img_size)
        else:
            frame = Image.open(file)
            frame = frame.resize(img_size)
            
        frame = np.asarray(frame)
        #frame = keras.preprocessing.image.img_to_array(frame)
        
        return self.predict_frame(frame)
        
    def predict_frame(self, frame):    
        if frame.shape[0]!=self.img_height or frame.shape[1]!=self.img_width:
            print('before resize :',frame.shape)
                
            img_size = (self.img_height,self.img_width)
            if False:
                print(type(frame))
                frame = tf.keras.preprocessing.image.array_to_img(frame, scale=False)
                #print(type(frame))
                frame = tf.image.resize(frame, size=img_size)
            else:
                #re-size
                im = Image.fromarray(frame)
                frame = im.resize(img_size)
            #frame = tf.keras.preprocessing.image.img_to_array(frame)
            frame = np.asarray(frame)
                
        if self.debug:
            print('type:',type(frame))
            print('shape:',frame.shape)
            print('-')
        
        img_array = tf.expand_dims(frame, 0) # Create a batch
        #img_array = frame.reshape(1, 360,448,3)
        #print('shape:',img_array.shape)
        #print('-')
        
        predictions = self.model.predict(img_array)        
        score = tf.nn.softmax(predictions[0])
        
        if self.debug:
            print('Predictions:',predictions)
            print('Score:',score)
            print('-')
            print('Class:', self.class_names[np.argmax(score)], '[', round(100*np.max(score),3), '%]')
            print('-'*25)
        
            #for l,s in zip(self.class_names,list(score*100)):
            #    print(l.rjust(10,' '),round(float(s),3))
                
            df = pd.DataFrame({'ClassName': self.class_names,
                             'Score': [round(float(s),3) for s in list(score*100)]})
            print(df.sort_values(by='Score', ascending=False))
            print('-')
            sns.barplot(x=df.Score, y=df.ClassName, ci=None)
                
        return (self.class_names[np.argmax(score)], round(100*np.max(score),3))

if __name__ == '__main__':
    print('Testing..')
    cfg_file = ImageClassificationCfg('./project/home_presence.yml')
    cfg_file.load()

    cnn = LoadTestCNNUtil(cfg_file)
    cnn.debug = False
    cnn.load()
    print('Prediction: ',cnn.predict('./project/home_presence/no-one/frame_img_20210713121302.png'))