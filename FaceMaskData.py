import os
import sys
import uuid
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
    def __init__(self, images_path, annotations_path):
        self.images_path = images_path
        self.annotations_path = annotations_path

    def split(self, train_size=.8, drop_rate=0, seed=42):
        if train_size > 1 or train_size < 0:
            raise ValueError('Split sizes can not be greater than 1 or less than 0')

        if drop_rate > 0:
            self.images = self.images[:int(len(self.images) * (1 - drop_rate))]
            self.annotates = self.annotates[:int(len(self.annotates) * (1 - drop_rate))]

        if train_size == 1:
            return self.images, self.annotates

        test_size = 1 - train_size
        x_train, x_test, y_train, y_test = train_test_split(self.images, self.annotates, test_size=test_size, random_state=seed)

        return (x_train, y_train), (x_test, y_test)


    def load_data(self, train_size=.8, drop_rate=0, seed=42):

        self.images = np.array( [img for img in sorted(os.listdir(self.images_path))] )
        self.annotates = np.array( [ant for ant in sorted(os.listdir(self.annotations_path))] )

        return self.split(train_size, drop_rate, seed)


def test_data():
    imgs_path = './images'
    msks_path = './annotations'

    faceMasksData = FaceMaskData(imgs_path, msks_path)
    (x_train, y_train), (x_test, y_test) = faceMasksData.load_data()

    for x_phase, y_phase in [(x_train, y_train), (x_test, y_test)]:
        for x, y in zip(x_phase, y_phase):
            print(x, y)

    print('Training contains {} samples which is {:g}% of the data'.format(len(x_train), len(x_train) * 100 / (len(x_train) + len(x_test))))
    print('Validation contains {} samples which is {:g}% of the data'.format(len(x_test), len(x_test) * 100 / (len(x_train) + len(x_test))))
    

if __name__ == '__main__':
    test_data()