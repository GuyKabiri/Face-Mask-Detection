import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from matplotlib import pyplot as plt
from FaceMaskData import FaceMaskData

class FaceMaskDataset(Dataset):
    def __init__(self, samples_name, annotations, samples_path, is_xy=True, transforms=None):
        self.x = samples_name
        self.y = annotations
        self.path = os.path.join(sys.path[0], samples_path)
        self.is_xy = is_xy
        self.transforms = transforms

        self.class_names = {
            0:  'mask',
            1:  'incorrect mask',
            2:  'no mask'
        }

    def __getitem__(self, idx, with_text=False):
        img_name = self.x[idx]
        y_dict = self.y[idx]

        img_path = os.path.join(self.path, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        names = []
        for entity in y_dict['annotations']:
            if self.is_xy:
                xmin, ymin, xmax, ymax = entity['xmin'], entity['ymin'], entity['xmax'], entity['ymax']
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                xmin, ymin, width, height = entity['xmin'], entity['ymin'], entity['width'], entity['height']
                boxes.append([xmin, ymin, width, height])
            labels.append(entity['class_id'])
            names.append(entity['class_name'])

       
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)

        target = {
            'bboxes':   boxes,
            'labels':   labels,
            'image_id': y_dict['image_id'],
        }

        # target = {
        #     'bboxes':    boxes,
        #     'labels':   labels,
        #     'image_id': y_dict['image_id'],
        #     'area':     (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        #     'iscrowd':  int(len(y_dict['annotations']) > 1)
        # }

        if self.transforms:
            sample = self.transforms(   image=img,
                                        bboxes=target['bboxes'],
                                        labels=target['labels'])

            img = sample['image']
            target['bboxes'] = sample['bboxes']
            target['labels'] = sample['labels']
        
        if with_text:
            return img, target, names

        return img, target

    def __len__(self):
        return len(self.x)

    def decode(self, value):
        value = np.array(value, dtype=np.int8).item()
        return self.class_names[value] if value < len(self.class_names) else 'None'
        # values = torch.as_tensor(values, dtype=torch.int32)
        # return [ names[val] if val < len(names) else 'None' for val in values ]





def test_dataset():
    imgs_path = './images'
    msks_path = './annotation.csv'

    faceMasksData = FaceMaskData(imgs_path, msks_path)
    (x_train, y_train), (x_test, y_test) = faceMasksData.load_data()

    trainset = FaceMaskDataset(x_train, y_train, imgs_path, transforms=None)
    validset = FaceMaskDataset(x_test, y_test, imgs_path, transforms=None)

    print('Training contains {} samples which is {:g}% of the data'.format(len(trainset), len(trainset) * 100 / (len(trainset) + len(validset))))
    print('Validation contains {} samples which is {:g}% of the data'.format(len(validset), len(validset) * 100 / (len(trainset) + len(validset))))
    
    data_iter = iter(validset)
    for img, target in data_iter:
        if len(target['bboxes']) > 1:
            example = (img, target)
            break

    print(example)

    img, target = example

    for box, lbl in zip(target['bboxes'], target['labels']):
        xmin, ymin, xmax, ymax = np.array(box, dtype=np.int32)
        start_point = (xmin, ymin)
        end_point = (xmax, ymax)

        color = (0, 0, 0)
        if lbl == 0:
            color = (0, 255, 0)
        elif lbl == 1:
            color = (0, 0, 255)
        elif lbl == 2:
            color = (255, 0, 0)
        thickness = 1
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1 / 3
        thickness = 1
        img = cv2.putText(img, '{} ({})'.format(validset.decode(lbl), lbl), start_point, font, fontScale, color, thickness, cv2.LINE_AA)

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test_dataset()