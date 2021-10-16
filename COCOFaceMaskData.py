import os
import sys
# import torch
import uuid
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

    def split(self, train_size=.8, drop_rate=0, seed=42):
        if train_size > 1 or train_size < 0:
            raise ValueError('Split sizes can not be greater than 1 or less than 0')

        if drop_rate > 0:
            self.records = self.records[:int(len(self.records) * (1 - drop_rate))]

        if train_size == 1:
            return self.records

        test_size = 1 - train_size
        train_records, test_records = train_test_split(self.records, test_size=test_size, random_state=seed)

        return train_records, test_records


    def load_data(self, train_size=.8, drop_rate=0, seed=42):

        imgs_path = os.path.join(sys.path[0], self.images_path)
        images = os.listdir(imgs_path)
        self.records = []

        msks_path = self.masks_path
        if not self.masks_path:
            msks_path = self.images_path
        
        msks_path = os.path.join(sys.path[0], msks_path)

        masks_df = pd.read_csv(msks_path)

        for img in images:
            record = dict()

            record['file_name'] = '{}/{}'.format(imgs_path, img)
            record['image_id'] = str(uuid.uuid1())
            annotations = []

            df_img_name = img.split('.')[0]         #   df file name format is: 'img123' while img variable is img123.jpg
            temp_df = masks_df[masks_df.file==df_img_name]

            for _, r in temp_df.iterrows():
                record['width'] = r['width']
                record['height'] = r['height']

                xmin, ymin, xmax, ymax = r['xmin'], r['ymin'], r['xmax'], r['ymax']

                poly = [
                        (xmin, ymin), (xmax, ymin),
                        (xmax, ymax), (xmin, ymax)
                ]
                poly = list(itertools.chain.from_iterable(poly))

                single_mask = {
                    'iscrowd':      int(len(temp_df) > 1),
                    # single_mask['bbox'] = [ r['xmin'], r['ymin'], r['boxW'], r['boxH'] ]
                    # single_mask['bbox_mode']= BoxMode.XYWH_ABS,
                    'bbox':         [ xmin, ymin, xmax, ymax ],
                    'bbox_mode':    BoxMode.XYXY_ABS,
                    'segmentation': [poly],
                    'category_id': r['class']
                }

                annotations.append(single_mask)
            
            record['annotations'] = annotations
            self.records.append(record)

        self.records = np.array(self.records, dtype=object)

        return self.split()


def test_data():
    imgs_path = './images'
    msks_path = './annotation.csv'

    faceMasksData = FaceMaskData(imgs_path, msks_path)

    from detectron2.data import DatasetCatalog, MetadataCatalog

    train_recs, valid_recs = faceMasksData.load_data()
    records = {
        'train':    train_recs,
        'valid':    valid_recs
    }

    for phase in ['train', 'valid']:
        DatasetCatalog.register('masks_{}_dataset'.format(phase), lambda: records[phase] )
        MetadataCatalog.get('masks_{}_dataset'.format(phase)).set(thing_classes=faceMasksData.classes)

    metadata_rec = {
        'train':    MetadataCatalog.get('masks_train_dataset'),
        'valid':    MetadataCatalog.get('masks_valid_dataset')
    }

    dataset_dicts = {
        'train':    DatasetCatalog.get('masks_train_dataset'),
        'valid':    DatasetCatalog.get('masks_valid_dataset')
    }

    print('Training contains {} samples which is {:g}% of the data'.format(len(dataset_dicts['train']), len(dataset_dicts['train']) * 100 / (len(dataset_dicts['train']) + len(dataset_dicts['valid']))))
    print('Validation contains {} samples which is {:g}% of the data'.format(len(dataset_dicts['valid']), len(dataset_dicts['valid']) * 100 / (len(dataset_dicts['train']) + len(dataset_dicts['valid']))))

    print(dataset_dicts['train'][1])


    # (x_train, y_train), (x_test, y_test) = faceMasksData.load_data()

    # for x_phase, y_phase in [(x_train, y_train), (x_test, y_test)]:
    #     for x, y in zip(x_phase, y_phase):
    #         print(x, y)

    # print('Training contains {} samples which is {:g}% of the data'.format(len(x_train), len(x_train) * 100 / (len(x_train) + len(x_test))))
    # print('Validation contains {} samples which is {:g}% of the data'.format(len(x_test), len(x_test) * 100 / (len(x_train) + len(x_test))))
    
    



if __name__ == '__main__':
    test_data()