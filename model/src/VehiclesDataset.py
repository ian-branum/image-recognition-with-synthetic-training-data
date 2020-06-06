import os
import random
from PIL import Image
import numpy as np

from mrcnn import utils


class VehiclesDataset(utils.Dataset):
    '''
    This provides tools to load the training images/masks and encapsulates them for the model
    '''
    def load_images(self, dataset_dir, mask_dir, proportion=1.0):
        '''
        Load images from filesystem into object
        '''
        self.mask_dir = mask_dir
        self.add_class('vehicles', 1, 'car')
        self.add_class('vehicles', 2, 'pickup')
        for filename in os.listdir(dataset_dir):
            if(os.path.isdir(dataset_dir + '\\' + filename)):
                continue
            image_path = os.path.join(dataset_dir, filename)
            if(random.random() < proportion):            
                self.add_image(
                    "vehicles",
                    image_id=filename,  
                    path=image_path,
                    width=256, 
                    height=256
            )

    def load_mask(self, image_id):
        '''
        Return the mask for the id'd image
        '''
        id = self.image_info[image_id]['id']
        car_id = id.split('_')[2]
        cam_id = id.split('_')[4]
        shot_id = id.split('_')[5]
        mask_id = '{0}_{1}_{2}'.format(car_id, cam_id, shot_id)
        mask = Image.open(self.mask_dir + mask_id)
        if('car' in car_id):
            class_id = 1
        else:
            class_id = 2
        mask_as_array = np.expand_dims(np.array(mask), axis=2)
        return mask_as_array*1, np.array([class_id])
