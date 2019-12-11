import torch
import numpy as np
import scipy
import pdb

def sampling_bernoulli(probs):
    #pdb.set_trace()
    return probs - torch.rand(probs.size()).cuda()


def sampling_gaussian(probs):
    return probs+ torch.rand(probs.size()).cuda()


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def norm_minmax(x):
    min_val = np.min(x)
    max_val = np.max(x)
    out = (x-min_val) / (max_val-min_val)
    return out


def show_multiple_images(images,size):
    template = np.zeros(((size[0]*8,size[1]*8)))
    cnt = 0
    for i in range(8):
        for j in range(8):
            template[i*size[0]:i*size[0]+size[0],j*size[1]:j*size[1]+size[1]] = np.reshape(images[cnt,:],(28,28))
            cnt +=1
    return template