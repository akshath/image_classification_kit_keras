import tensorflow as tf
import numpy as np

class RandomColorDistortion(tf.keras.layers.Layer):
    contrast_range=[-1.0, 1.0]
    brightness_delta=[-50, 50]

    def __init__(self, **kwargs):
        super(RandomColorDistortion, self).__init__(**kwargs)

    def update_cfg(self, cfg_file):
        contrast_range = cfg_file.cfg['train_augumentation']['contrast_range']
        brightness_delta = cfg_file.cfg['train_augumentation']['brightness_delta']

    def call(self, images, training=True):
        if not training:
            return images

        contrast = np.random.uniform(self.contrast_range[0], self.contrast_range[1])
        brightness = np.random.uniform(self.brightness_delta[0], self.brightness_delta[1])

        #print('brightness: ',brightness, ', contrast: ',contrast)

        #images = tf.image.adjust_contrast(images, contrast)
        images = tf.image.adjust_brightness(images, brightness)
        images = tf.clip_by_value(images, 0, 255)
        return images