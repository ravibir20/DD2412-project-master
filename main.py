from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_vgg
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg 
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import decode_predictions as decode_resnet
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet

from tensorflow.keras.preprocessing import image
from analysis import analysis
from utils import process_image, rescale
import matplotlib.pyplot as plt
from os.path import isdir
from os import mkdir
import glob

configs     =   [
                    ["vgg", VGG16(weights="imagenet"), preprocess_vgg, decode_vgg, "block5_conv3", 1000],
                    ["resnet", ResNet50(weights="imagenet"), preprocess_resnet, decode_resnet, "conv5_block3_3_conv", 1000],
                ]
filenames   =   glob.glob("Images/224Images/*.png")

p_dir = "processed_images"
if not isdir(p_dir):
    mkdir(p_dir)
for filename in filenames:
    name = filename.rsplit("/")[-1].split(".")[0]
    n_dir = "{}/{}".format(p_dir, name)
    if isdir(n_dir):
        continue
    else:
        mkdir(n_dir)
    img = image.load_img(filename, target_size=(224,224))
    _,ground_truth = process_image(img, preprocess_vgg)
    plt.imsave("{}/ground_truth.jpg".format(n_dir), rescale(ground_truth))
    
    for modelname, model, preprocessor, decoder, layer_name, n_classes in configs:
        backprop, gradcam, squadcam, data = analysis(img, model, preprocessor, decoder, layer_name, n_classes)    
        plt.imsave("{}/{}_{}_guided_backprop.jpg".format(n_dir, modelname, data[1]), backprop)
        plt.imsave("{}/{}_{}_guided_gradcam.jpg".format(n_dir, modelname, data[1]), gradcam)
        plt.imsave("{}/{}_{}_guided_squadcam.jpg".format(n_dir, modelname, data[1]), squadcam)
        print(data)
    plt.show()
    