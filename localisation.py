"""
Draw bbox with grad-cam and evaluate localization ability
"""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_vgg
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.resnet import decode_predictions as decode_resnet
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_resnet

from tensorflow.keras.preprocessing import image
from analysis import analysis_localize
from utils import process_image, rescale
import matplotlib.pyplot as plt
import numpy as np
from os.path import isdir, isfile
from os import listdir, mkdir
import glob
import xml.etree.ElementTree as ET
from matplotlib.patches import Rectangle


def iou(bb1, bb2):
    """
    Intersection over union of 2 bounding boxes.
    Adapted from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    """
    x_left = max(bb1[1], bb2[1])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[3], bb2[3])
    y_bottom = min(bb1[4], bb2[4])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    bb1_area = (bb1[3] - bb1[1] + 1) * (bb1[4] - bb1[2] + 1)
    bb2_area = (bb2[3] - bb2[1] + 1) * (bb2[4] - bb2[2] + 1)
    return intersection_area / float(bb1_area + bb2_area - intersection_area)


def plot(bbs1, bbs2, filename, error1, error2):
    """
    Draw bounding boxes on given image and save
    """
    fig, ax = plt.subplots()
    ax.imshow(image.load_img(filename))
    for bb in bbs1:
        rect = Rectangle((bb[1], bb[2]), (bb[3] - bb[1]), (bb[4] - bb[2]), linewidth=2, edgecolor="g", facecolor="None")
        ax.add_patch(rect)
    for bb in bbs2[0]:
        rect = Rectangle((bb[1], bb[2]), (bb[3] - bb[1]), (bb[4] - bb[2]), linewidth=2, edgecolor="m", facecolor="None")
        print("grad {} {} {} {}".format(bb[1], bb[2], bb[3], bb[4]))
        ax.add_patch(rect)
    for bb in bbs2[1]:
        rect = Rectangle((bb[1], bb[2]), (bb[3] - bb[1]), (bb[4] - bb[2]), linewidth=2, edgecolor="c", facecolor="None")
        print("squad {} {} {} {}".format(bb[1], bb[2], bb[3], bb[4]))
        ax.add_patch(rect)
    plt.savefig("{}/{}_g{}_s{}.png".format(p_dir, name, error1, error2))
    plt.close("all")


imagenet_dir = "Images/ILSVRC2012_img_val"
bbox_dir = "Images/ILSVRC2012_bbox_val"
filenames = glob.glob("{}/*.JPEG".format(imagenet_dir))
perm = np.random.permutation(len(filenames))
filenames = [filenames[p] for p in perm]

p_dir = "images_with_bbox"

configs = [
    ["vgg", VGG16(weights="imagenet"), preprocess_vgg, decode_vgg, "block5_conv3", 1000],
    ["resnet", ResNet50(weights="imagenet"), preprocess_resnet, decode_resnet, "conv5_block3_3_conv", 1000]
]

errors = [[[], []], [[], []]]  # vgg grad, vgg squad, resnet grad, resnet squad
if not isdir(p_dir):
    mkdir(p_dir)
for filename in filenames:
    name = filename.rsplit("/")[-1].split(".")[0]
    bbox_file = "{}/{}.xml".format(bbox_dir, name)
    if not isfile(bbox_file):
        continue

    # Get bbox info
    bbox_info = ET.parse(bbox_file)
    xmin, ymin, xmax, ymax = bbox_info.findall(".//xmin"), bbox_info.findall(".//ymin"), bbox_info.findall(
        ".//xmax"), bbox_info.findall(".//ymax")
    xmin = [int(e.text) for e in xmin]
    ymin = [int(e.text) for e in ymin]
    xmax = [int(e.text) for e in xmax]
    ymax = [int(e.text) for e in ymax]

    # Get object ids
    names = bbox_info.findall(".//name")
    names = [e.text for e in names]

    bboxes_true = []
    for i in range(len(names)):
        bboxes_true.append([names[i], xmin[i], ymin[i], xmax[i], ymax[i]])

    # Skip imgs with > 5 unique objects
    if len(set(names)) > 5:
        continue

    img = image.load_img(filename, target_size=(224, 224))
    full_img = image.img_to_array(image.load_img(filename))
    w, h = full_img.shape[1], full_img.shape[0]

    j = 0
    for modelname, model, preprocessor, decoder, layer_name, n_classes in configs:
        bboxes, bboxes_s = analysis_localize(img, model, preprocessor, decoder,
                                             layer_name, n_classes)
        # Check if success and rescale
        if not bboxes:
            continue
        for i in range(len(bboxes)):
            bboxes[i][1], bboxes[i][3] = int((w / 224) * bboxes[i][1]), int((w / 224) * bboxes[i][3])
            bboxes[i][2], bboxes[i][4] = int((w / 224) * bboxes[i][2]), int((w / 224) * bboxes[i][4])

            bboxes_s[i][1], bboxes_s[i][3] = int((w / 224) * bboxes_s[i][1]), int((w / 224) * bboxes_s[i][3])
            bboxes_s[i][2], bboxes_s[i][4] = int((w / 224) * bboxes_s[i][2]), int((w / 224) * bboxes_s[i][4])

        filtered_bboxes = []


        """
        bbox evaluation
        https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/evaluation , 
        http://image-net.org/challenges/LSVRC/2015/
        """
        i = 0
        for bbxs in [bboxes, bboxes_s]:
            error_per_class = []
            best_boxes0 = []
            for n in set(names):
                bboxes_true_i = [b for b in bboxes_true if b[0] == n]
                min_errors = []
                best_boxes = []
                for bbox_true in bboxes_true_i:
                    max_errors = []
                    for bbox in bbxs:
                        if bbox_true[0] == bbox[0]:
                            d = 0
                        else:
                            d = 1
                        if iou(bbox, bbox_true) > 0.5:
                            f = 0
                        else:
                            f = 1
                        max_errors.append(max(d, f))
                    min_errors.append(min(max_errors))
                    best_boxes.append(np.argmin(max_errors))
                error_per_class.append(min(min_errors))
                best_boxes0.append(best_boxes[int(np.argmin(min_errors))])
            error = np.average(error_per_class)
            print("{} {}: {}".format(name, modelname, error))
            errors[j][i].append(error)
            i += 1
            filtered_bboxes.append([b for i, b in enumerate(bbxs) if i in best_boxes0])
        j += 1

    plot(bboxes_true, filtered_bboxes, filename, errors[0][0][-1], errors[0][1][-1])
print("vgg grad {}\nvgg squad {}\nresnet grad {}\nresnet squad {}".format(np.average(errors[0][0]),
                                                                          np.average(errors[0][1]),
                                                                          np.average(errors[1][0]),
                                                                          np.average(errors[1][1])))
