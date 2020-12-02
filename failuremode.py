from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_vgg
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
import ssl
ssl._create_default_https_context = ssl._create_unverified_context  # Mac workaround
from tensorflow.keras.preprocessing import image
from analysis import analysis
from utils import process_image, rescale
import matplotlib.pyplot as plt
from os.path import isdir, isfile
from os import mkdir
import glob
import xml.etree.ElementTree as ET

id_to_class = {}
f = open("id_to_class.txt")
for l in f:
    id_to_class[l.split(" ", maxsplit=1)[0]] = l.split(" ", maxsplit=1)[1].rstrip()

configs = [
    ["vgg", VGG16(weights="imagenet"), preprocess_vgg, decode_vgg, "block5_conv3", 1000]]
imagenet_dir = "Images/ILSVRC2012_img_val"
bbox_dir = "Images/ILSVRC2012_bbox_val"
filenames = glob.glob("{}/*.JPEG".format(imagenet_dir))
p_dir = "failuremode_images"

if not isdir(p_dir):
    mkdir(p_dir)
for filename in filenames:
    name = filename.rsplit("/")[-1].split(".")[0]
    bbox_file = "{}/{}.xml".format(bbox_dir, name)
    bbox_info = ET.parse(bbox_file)
    names = bbox_info.findall(".//name")
    names = list(set([e.text for e in names]))
    names_h = [id_to_class[n] for n in names]
    names_h = "_".join(names_h)
    if not isfile(bbox_file):
        continue
    n_dir = "{}/{}".format(p_dir, name)
    if isdir(n_dir):
        continue
    img = image.load_img(filename, target_size=(224, 224))
    imgArray, ground_truth = process_image(img, preprocess_vgg)

    for modelname, model, preprocessor, decoder, layer_name, n_classes in configs:
        preds = model.predict(imgArray)
        decoded_preds = decode_vgg(preds)
        top5 = [decoded_preds[0][i][0] for i in range(5)]
        skip = False
        for c in names:
            if c in top5:
                skip = True
        if skip:
            continue

        mkdir(n_dir)
        plt.imsave("{}/ground_truth.jpg".format(n_dir), rescale(ground_truth))
        backprop, gradcam, squadcam, data = analysis(img, model, preprocessor, decoder, layer_name, n_classes)
        plt.imsave("{}/{}_{}_{}_guided_backprop.jpg".format(n_dir, modelname, data[1], names_h), backprop)
        plt.imsave("{}/{}_{}_{}_guided_gradcam.jpg".format(n_dir, modelname, data[1], names_h), gradcam)
        plt.imsave("{}/{}_{}_{}_guided_squadcam.jpg".format(n_dir, modelname, data[1], names_h), squadcam)
        print(data)
