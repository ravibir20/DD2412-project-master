from tensorflow.keras.preprocessing import image
import numpy as np
import keras.backend as K

def process_image(img, preprocessor):
    imgArray        =   image.img_to_array(img)
    originalImage   =   imgArray.astype(int)
    imgArray        =   np.expand_dims(imgArray, axis=0)
    imgArray        =   preprocessor(imgArray)
    return imgArray, originalImage

def analyse(x):
    print(np.min(x), np.max(x), x.shape)

def rescale(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x