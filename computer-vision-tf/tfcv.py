# Tensorflow Computer Vision Helper

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os

def plot_convolution(data,t,title=''):
    fig, ax = plt.subplots(2,len(data)+1,figsize=(8,3))
    fig.suptitle(title,fontsize=16)
    tt = np.expand_dims(np.expand_dims(t,2),2)
    for i,im in enumerate(data):
        ax[0][i].imshow(im)
        ximg = np.expand_dims(np.expand_dims(im,2),0)
        cim = tf.nn.conv2d(ximg,tt,1,'SAME')
        ax[1][i].imshow(cim[0][:,:,0])
        ax[0][i].axis('off')
        ax[1][i].axis('off')
    ax[0,-1].imshow(t)
    ax[0,-1].axis('off')
    ax[1,-1].axis('off')
    #plt.tight_layout()
    plt.show()

def plot_results(hist):
    fig,ax = plt.subplots(1,2,figsize=(15,3))
    ax[0].set_title('Accuracy')
    ax[1].set_title('Loss')
    for x in ['acc','val_acc']:
        ax[0].plot(hist.history[x])
    for x in ['loss','val_loss']:
        ax[1].plot(hist.history[x])
    plt.show()

def display_dataset(dataset, labels=None, n=10, classes=None):
    fig,ax = plt.subplots(1,n,figsize=(15,3))
    for i in range(n):
        ax[i].imshow(dataset[i])
        ax[i].axis('off')
        if classes is not None and labels is not None:
            ax[i].set_title(classes[labels[i][0]])

def check_image(fn):
    try:
        im = Image.open(fn)
        im.verify()
        return True
    except:
        return False
    
def check_image_dir(path):
    for fn in glob.glob(path):
        if not check_image(fn):
            print("Corrupt image: {}".format(fn))
            os.remove(fn)

