import numpy as np

def linear(alphas, output_values):
    combo = np.zeros((output_values.shape[0],output_values.shape[1]))
    for i, alpha in enumerate(alphas):
        combo += alpha*output_values[:,:,i]
    return combo

def squared_values(alphas, output_values):
    combo = np.zeros((output_values.shape[0],output_values.shape[1]))
    for i, alpha in enumerate(alphas):
        combo += alpha*np.multiply(output_values[:,:,i], output_values[:,:,i])
    return combo

def squared_weights(alphas, output_values):
    combo = np.zeros((output_values.shape[0],output_values.shape[1]))
    for i, alpha in enumerate(alphas):
        combo += alpha*alpha*output_values[:,:,i]
    return combo

def root_weights(alphas, output_values):
    combo = np.zeros((output_values.shape[0],output_values.shape[1]))
    for i, alpha in enumerate(alphas):
        combo += np.sqrt(np.abs(alpha))*output_values[:,:,i]
    return combo

def root_values(alphas, output_values):
    combo = np.zeros((output_values.shape[0],output_values.shape[1]))
    for i, alpha in enumerate(alphas):
        combo += alpha*np.sqrt(np.abs(output_values[:,:,i]))
    return combo
