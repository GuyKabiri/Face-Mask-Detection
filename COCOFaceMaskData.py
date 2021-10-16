import os
import sys
# import torch
import uuid
import numpy as np
import pandas as pd
from detectron2.structures import BoxMode

'''
        dataset = [{'file_name': '..//first_image.jpg',
                'image_id': 125361,
                'height': 1300,
                'width': 800,
                'annotations': [
                    {'iscrowd': 0,
                    'segmentation': [[x_0, y_0, x_1, y_1, ..., x_n, y_n]], 
                    'bbox': [x_0, y_0, width, height],
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'category_id': 0 },
                    {'iscrowd': 0,
                    'segmentation': [[x_0, y_0, x_1, y_1, ..., x_n, y_n]], 
                    'bbox': [x_0, y_0, width, height],
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'category_id': 1 }]},
                {'file_name': '..//second_image.jpg',
                'image_id': 1425361,
                'height': 1300,
                'width': 800,
                'annotations': [
                    {'iscrowd': 0,
                    'segmentation': [[x_0, y_0, x_1, y_1, ..., x_n, y_n]], 
                    'bbox': [x_0, y_0, width, height],
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'category_id': 0 },
                    {'iscrowd': 0,
                    'segmentation': [[x_0, y_0, x_1, y_1, ..., x_n, y_n]], 
                    'bbox': [x_0, y_0, width, height],
                    'bbox_mode': BoxMode.XYWH_ABS,
                    'category_id': 3 }]}, ...]


        file fields:
        xmin,ymin,xmax,ymax,name,file,width,height,class,Xcent,Ycent,boxW,boxH
'''

class FaceMaskData:
    def __init__(self, images_path, masks_path=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.classes = [ 'with_mask', 'mask_weared_incorrect', 'without_mask' ]

    # def split(self, train_size=.8, drop_rate=0, seed=42):
    #     if train_size > 1 or train_size < 0:
    #         raise ValueError('Split sizes can not be greater than 1 or less than 0')

    #     if drop_rate > 0:
    #         self.images = self.images[:int(len(self.images) * (1 - drop_rate))]
    #         self.masks = self.masks[:int(len(self.masks) * (1 - drop_rate))]

    #     if train_size == 1:
    #         return self.images, self.masks

    #     test_size = 1 - train_size
    #     x_train, x_test, y_train, y_test = train_test_split(self.images, self.masks, test_size=test_size, random_state=seed)

    #     return (x_train, y_train), (x_test, y_test)


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
            record = dict()

            record['file_name'] = '{}/{}'.format(imgs_path, img)
            record['image_id'] = str(uuid.uuid1())
            annotations = []

            df_img_name = img.split('.')[0]         #   df file name format is: 'img123' while img variable is img123.jpg
            temp_df = masks_df[masks_df.file==df_img_name]

            for _, r in temp_df.iterrows():
                record['width'] = r['width']
                record['height'] = r['height']

                single_mask = {
                    'iscrowd':      int(len(temp_df) > 1),
                    # single_mask['bbox'] = [ r['xmin'], r['ymin'], r['boxW'], r['boxH'] ]
                    # single_mask['bbox_mode']= BoxMode.XYWH_ABS,
                    'bbox':         [ r['xmin'], r['ymin'], r['xmax'], r['ymax'] ],
                    'bbox_mode':    BoxMode.XYXY_ABS,
                    'category_id': r['class']
                }

                annotations.append(single_mask)
            
            record['annotations'] = annotations
            self.masks.append(record)

        self.images = np.array(self.images)
        self.masks = np.array(self.masks, dtype=object)

        return self.masks


def test_data():
    imgs_path = './images'
    msks_path = './annotation.csv'

    faceMasksData = FaceMaskData(imgs_path, msks_path)

    from detectron2.data import DatasetCatalog, MetadataCatalog

    DatasetCatalog.register('mask_dataset', lambda: faceMasksData.load_data())
    MetadataCatalog.get('mask_dataset').set(thing_classes=faceMasksData.classes)
    chess_metadata = MetadataCatalog.get('mask_dataset')

    dataset_dicts = DatasetCatalog.get('mask_dataset')

    print(len(dataset_dicts))

    print(dataset_dicts[1])


    # (x_train, y_train), (x_test, y_test) = faceMasksData.load_data()

    # for x_phase, y_phase in [(x_train, y_train), (x_test, y_test)]:
    #     for x, y in zip(x_phase, y_phase):
    #         print(x, y)

    # print('Training contains {} samples which is {:g}% of the data'.format(len(x_train), len(x_train) * 100 / (len(x_train) + len(x_test))))
    # print('Validation contains {} samples which is {:g}% of the data'.format(len(x_test), len(x_test) * 100 / (len(x_train) + len(x_test))))
    
    



if __name__ == '__main__':
    test_data()