import os
import sys
# import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

'''
        file fields:
        xmin,ymin,xmax,ymax,name,file,width,height,class,Xcent,Ycent,boxW,boxH

        output:
            x_train:    [img1.png, img2.png, img3.png, ...]
            y_train:    [
                            {   name:           img1.png,
                                annotations:    [
                                    {
                                        xmin:   ... ,
                                        ymin:   ... ,
                                        xman:   ... ,
                                        ymax:   ... ,
                                        height: ... ,
                                        width:  ... ,
                                        class_name: with_mask   /   mask_weared_incorrect   /   without_mask ,
                                        class_id:   0           /   1                       /   2 ,
                                    }
                                ]
                            },
                            {
                                name:   ... ,
                                annotations: [ { ... } ]
                            }
            ]

'''

class FaceMaskData:
    def __init__(self, images_path, masks_path=None):
        self.images_path = images_path
        self.masks_path = masks_path

    def split(self, train_size=.8, drop_rate=0, seed=42):
        if train_size > 1 or train_size < 0:
            raise ValueError('Split sizes can not be greater than 1 or less than 0')

        if drop_rate > 0:
            self.images = self.images[:int(len(self.images) * (1 - drop_rate))]
            self.masks = self.masks[:int(len(self.masks) * (1 - drop_rate))]

        if train_size == 1:
            return self.images, self.masks

        test_size = 1 - train_size
        x_train, x_test, y_train, y_test = train_test_split(self.images, self.masks, test_size=test_size, random_state=seed)

        return (x_train, y_train), (x_test, y_test)


    def load_data(self, train_size=.8, drop_rate=0, seed=42):

        imgs_path = os.path.join(sys.path[0], self.images_path)
        self.images = os.listdir(imgs_path)
        self.masks = []

        msks_path = self.masks_path
        if not self.masks_path:
            msks_path = self.images_path
        
        msks_path = os.path.join(sys.path[0], msks_path)

        masks_df = pd.read_csv(msks_path)

        for img in self.images:
            img_dict = dict()

            img_dict['name'] = img
            img_masks = []

            df_img_name = img.split('.')[0]         #   df file name format is: 'img123' while img variable is img123.jpg
            temp_df = masks_df[masks_df.file==df_img_name]

            for _, r in temp_df.iterrows():
                single_mask = dict()
                single_mask['xmin'] = r['xmin']
                single_mask['ymin'] = r['ymin']
                single_mask['xmax'] = r['xmax']
                single_mask['ymax'] = r['ymax']
                single_mask['width'] = r['width']
                single_mask['height'] = r['height']
                single_mask['class_name'] = r['name']
                single_mask['class_id'] = r['class']

                img_masks.append(single_mask)
            
            img_dict['annotations'] = img_masks
            self.masks.append(img_dict)

        self.images = np.array(self.images)
        self.masks = np.array(self.masks, dtype=object)

        return self.split(train_size, drop_rate, seed)


def test_data():
    imgs_path = './images'
    msks_path = './annotation.csv'

    faceMasksData = FaceMaskData(imgs_path, msks_path)
    (x_train, y_train), (x_test, y_test) = faceMasksData.load_data()

    for x_phase, y_phase in [(x_train, y_train), (x_test, y_test)]:
        for x, y in zip(x_phase, y_phase):
            print(x, y)

    print('Training contains {} samples which is {:g}% of the data'.format(len(x_train), len(x_train) * 100 / (len(x_train) + len(x_test))))
    print('Validation contains {} samples which is {:g}% of the data'.format(len(x_test), len(x_test) * 100 / (len(x_train) + len(x_test))))
    
    



if __name__ == '__main__':
    test_data()