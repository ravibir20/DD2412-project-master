from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Lambda
from tensorflow.keras.backend import sum, function
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
from tensorflow import one_hot
import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
import skimage.measure
from skimage.measure import regionprops
# Import from other files
from utils import process_image, analyse, rescale, deprocess_image
from activation_functions import relu as relu_activation
from combinations import linear, squared_weights


def guided_backprop(input_data, model, index, layer_name, n_classes):
    # Define the loss function
    @tf.custom_gradient
    def guidedRelu(x):
        def grad(dy):
            return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

        return tf.nn.relu(x), grad

    def category_loss(x):
        return categorical_crossentropy(tf.one_hot([index], n_classes), x)

    # Update with loss output
    loss_layer = Lambda(category_loss)(model.output)
    guidedModel = Model(inputs=model.input, outputs=loss_layer)

    # Replace relu activations with our custom activation function
    for layer in guidedModel.layers:
        if (hasattr(layer, "activation")):
            if layer.activation == relu:
                layer.activation = guidedRelu

    # Compute the gradient.
    with tf.GradientTape() as tape:
        indata = tf.cast(input_data, tf.float32)
        tape.watch(indata)
        loss = guidedModel(indata)

    gradients = tape.gradient(loss, indata)[0]

    # Don't know why this, but it seems to work :)
    gradients = np.flip(deprocess_image(np.array(gradients)), -1)

    return gradients


def grad_cam(input_data, model, index, originalImage, activation_function, combination, layer_name, n_classes):
    # Define the loss function
    def loss_function(x):
        return categorical_crossentropy(one_hot([index], n_classes), x)

    # Create loss layer
    loss_layer = Lambda(loss_function)(model.output)

    # Build a new model with the loss function.
    model = Model(inputs=model.input, outputs=loss_layer)

    # Extract layer and loss to compute gradient
    conv_layer = model.get_layer(layer_name).output
    loss = sum(model.output)

    # Snagged this code from someplace else
    grads = K.gradients(loss, conv_layer)
    gradient_function = function([model.inputs[0]], [conv_layer, grads])

    # Compute the desired values
    output_values, gradients = gradient_function([input_data])

    output_values = output_values[0]
    gradients = -gradients[0][0]

    # Compute weights according to equation 1
    alphas = np.mean(gradients, axis=(0, 1))

    # Apply combination
    combo = combination(alphas, output_values)

    # Apply activation function
    combo = activation_function(combo)

    # Reshape and rescale feature map
    combo = cv2.resize(combo, (224, 224))
    combo = rescale(combo)

    return combo


def analysis(img, model, preprocess_input, decode_predictions, layer_name, n_classes):
    # Preprocess data
    imgArray, originalImage = process_image(img, preprocess_input)
    preds = model.predict(imgArray)
    pred_class = np.argmax(preds)  # Change here to get view of something else
    decoded_preds = decode_predictions(preds)
    class_data = decoded_preds[0][0]

    # Compute methods
    localization_grad = grad_cam(imgArray, model, pred_class, originalImage, relu_activation, linear, layer_name,
                                 n_classes)
    localization_squad = grad_cam(imgArray, model, pred_class, originalImage, relu_activation, squared_weights,
                                  layer_name, n_classes)
    bprop = guided_backprop(imgArray, model, pred_class, layer_name, n_classes)

    # Make it three dimensions to allow for multiplication
    localization_grad = np.array([localization_grad, localization_grad, localization_grad])
    localization_grad = np.swapaxes(localization_grad, 0, 2)
    localization_grad = np.swapaxes(localization_grad, 0, 1)
    localization_squad = np.array([localization_squad, localization_squad, localization_squad])
    localization_squad = np.swapaxes(localization_squad, 0, 2)
    localization_squad = np.swapaxes(localization_squad, 0, 1)

    # Combine gradcam and backprop to get guided gradcam
    guided_gradcam = np.multiply(localization_grad, bprop)
    guided_squadcam = np.multiply(localization_squad, bprop)
    guided_gradcam = rescale(guided_gradcam)
    guided_squadcam = rescale(guided_squadcam)

    return bprop, guided_gradcam, guided_squadcam, class_data


def analysis_localize(img, model, preprocess_input, decode_predictions, layer_name, n_classes):
    # Preprocess data
    imgArray, originalImage = process_image(img, preprocess_input)
    preds = model.predict(imgArray)
    # pred_class = np.argmax(preds)  # Change here to get view of something else
    pred_classes = np.argsort(-preds[0])[:5]  # top 5 classes
    decoded_preds = decode_predictions(preds)

    bboxes = []
    bboxes_s = []

    for i, pred_class in enumerate(pred_classes):
        # Compute methods
        localization_grad = grad_cam(imgArray, model, pred_class, originalImage, relu_activation, linear, layer_name,
                                     n_classes)
        # Threshold and pick largest

        max_value = np.amax(localization_grad)
        masked = np.where(localization_grad > 0.15 * max_value, 1, 0)
        labels = skimage.measure.label(masked)
        if labels.any() > 0:
            largest_segment = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            largest_segment = np.where(largest_segment, 1, 0)
            for region in regionprops(largest_segment):
                ymin, xmin, ymax, xmax = region.bbox
            bboxes.append([decoded_preds[0][i][0], xmin, ymin, xmax, ymax])
        else:
            return False, False
    for i, pred_class in enumerate(pred_classes):
        # Compute methods
        localization_squad = grad_cam(imgArray, model, pred_class, originalImage, relu_activation, squared_weights,
                                      layer_name, n_classes)
        # Threshold and pick largest

        max_value = np.amax(localization_squad)
        masked = np.where(localization_squad > 0.15 * max_value, 1, 0)
        labels = skimage.measure.label(masked)
        if labels.any() > 0:
            largest_segment = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
            largest_segment = np.where(largest_segment, 1, 0)
            for region in regionprops(largest_segment):
                ymin, xmin, ymax, xmax = region.bbox
            bboxes_s.append([decoded_preds[0][i][0], xmin, ymin, xmax, ymax])
        else:
            return False, False
    del(imgArray)
    del(originalImage)
    del(localization_grad)
    del(localization_squad)


    return bboxes, bboxes_s


if __name__ == "__main__":
    from tensorflow.keras.preprocessing import image
    import matplotlib.pyplot as plt

    filename = "boxer.jpg"
    img = image.load_img(filename, target_size=(224, 224))
    a, b, c, d = analysis(img)

    plt.imshow(img)
    plt.figure()
    plt.imshow(a)
    plt.figure()
    plt.imshow(b)
    plt.figure()
    plt.imshow(c)
    plt.show()
    print(d)
