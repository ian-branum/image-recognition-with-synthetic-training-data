from PIL import Image
import os
import random
import numpy as np

def make_grayscale(directory, target_sub='gray'):
    '''
    Convert all images in directory 'directory' to grayscale and write to directory 'target_sub'
    '''
    if not os.path.exists(os.path.join(directory, target_sub)):
        os.mkdir(os.path.join(directory, target_sub))
    for filename in os.listdir(directory):
        if(os.path.isdir(os.path.join(directory, filename))):
            continue
        img = Image.open(os.path.join(directory, filename)).convert('LA') 
        target = os.path.join(directory, target_sub, filename)
        img.save(target)

def flatten_mask(directory, target_sub='flattened'):
    '''
    Flattens mask from RGB, that happens to be bw, to one bit deep
    '''
    if not os.path.exists(os.path.join(directory, target_sub)):
        os.mkdir(os.path.join(directory, target_sub))
    for filename in os.listdir(directory):
        if(os.path.isdir(os.path.join(directory, filename))):
            continue
        img = Image.open(os.path.join(directory, filename)).convert('LA')
        flat = img.convert('1') 
        target = os.path.join(directory, target_sub, filename)
        flat.save(target)

def prepare_images(directory, target_sub='../prepared', prefix='img-', new_width=256, new_height=256, crop=False):
    '''
    Prepare real images for prediction. Make 256x256 and either resize or crop out the center. 
    '''
    if not os.path.exists(os.path.join(directory, target_sub)):
        os.mkdir(os.path.join(directory, target_sub))
    for i, filename in enumerate(os.listdir(directory)):
        if(os.path.isdir(os.path.join(directory, filename))):
            continue
        target = '{0}/{1}/{2}{3}.png'.format(directory, target_sub, prefix, i)
        img = Image.open(os.path.join(directory, filename))
        if(crop):
            width, height = img.size   # Get dimensions
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            img = img.crop((left, top, right, bottom))
        else:
            img = img.resize((new_width, new_height))
        img.save(target)


def rename_images(directory, prefix='car'):
    '''
    Rename images to 'img-1.png' or the such
    '''
    for i, filename in enumerate(os.listdir(directory)):
        if(os.path.isdir(os.path.join(directory, filename))):
            continue
        img = Image.open(directory + '\\' + filename).convert('LA')
        target = '{0}\\{1}{2}.png'.format(directory, prefix, i)
        img.save(target)

def train_val_split(directory, val_size=0.1):
    '''
    Split train and val images into separate directories
    '''
    if not os.path.exists(os.path.join(directory, 'train')):
        os.mkdir(os.path.join(directory, 'train'))
        os.mkdir(os.path.join(directory, 'val'))
    for filename in os.listdir(directory):
        if(os.path.isdir(os.path.join(directory, filename))):
            continue
        if(random.random() < val_size): 
            os.rename(os.path.join(directory, filename), os.path.join(directory, 'val', filename))
        else:
            os.rename(os.path.join(directory, filename), os.path.join(directory, 'train', filename))
        break