import os
import sys
from torch.utils.data import Dataset
import cv2
import numpy as np
from FaceMaskData import FaceMaskData

class FaceMaskDataset(Dataset):
    def __init__(self, samples_name, annotations, samples_path, transforms=None, is_width_height=False, width=None, height=None):
        self.x = samples_name
        self.y = annotations
        self.path = os.path.join(sys.path[0], samples_path)
        self.transforms = transforms
        self.is_width_height = is_width_height
        self.width = width
        self.height = height

    def __getitem__(self, idx):
        img_name = self.x[idx]
        y_dict = self.y[idx]
        rectangles = []
        clases_id = []
        classes_name = []

        print(img_name,y_dict)

        img_path = os.path.join(self.path, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.width and self.height:
            img = cv2.resize(img, (self.width, self.height), cv2.INTER_AREA)
        
        if self.transforms:
            x = self.transforms(image = img)
            x = x['image']
        else:
            x = np.transpose(img, (2, 0, 1))

        for anot in y_dict['annotations']:
            xmin, ymin, xmax, ymax = anot['xmin'], anot['ymin'], anot['xmax'], anot['ymax']
            width, height = anot['width'], anot['height']
            class_name, class_id = anot['class_name'], anot['class_id']

            if self.is_width_height:
                values = [xmin, ymin, width, height]
            else:
                values = [xmin, ymin, xmax, ymax]

            rectangles.append(values)
            clases_id = [class_id]
            classes_name = [class_name]

        return x, rectangles, clases_id, classes_name

    def __len__(self):
        return len(self.x)




def test_dataset():
    imgs_path = './images'
    msks_path = './annotation.csv'

    faceMasksData = FaceMaskData(imgs_path, msks_path)
    (x_train, y_train), (x_test, y_test) = faceMasksData.load_data()

    trainset = FaceMaskDataset(x_train, y_train, imgs_path, transforms=None)
    validset = FaceMaskDataset(x_test, y_test, imgs_path, transforms=None)

    print('Training contains {} samples which is {:g}% of the data'.format(len(trainset), len(trainset) * 100 / (len(trainset) + len(validset))))
    print('Validation contains {} samples which is {:g}% of the data'.format(len(validset), len(validset) * 100 / (len(trainset) + len(validset))))
    
    print(next(iter(validset)))

if __name__ == '__main__':
    test_dataset()