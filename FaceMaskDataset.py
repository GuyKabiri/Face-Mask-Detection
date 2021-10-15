import os
import sys
from torch.utils.data import Dataset
import cv2
import numpy as np
from matplotlib import pyplot as plt
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
        classes_id = []
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
            # x = np.transpose(img, (2, 0, 1))
            x = img

        for anot in y_dict['annotations']:
            xmin, ymin, xmax, ymax = anot['xmin'], anot['ymin'], anot['xmax'], anot['ymax']
            width, height = anot['width'], anot['height']
            class_name, class_id = anot['class_name'], anot['class_id']

            if self.is_width_height:
                values = [xmin, ymin, width, height]
            else:
                values = [xmin, ymin, xmax, ymax]

            rectangles.append(values)
            classes_id.append(class_id)
            classes_name.append(class_name)

        return x, rectangles, classes_id, classes_name

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
    
    data_iter = iter(validset)
    for e in data_iter:
        if len(e[1]) > 1:
            example = e
            break

    print(example)

    img = example[0]
    rect = example[1]
    ids = example[2]
    labels = example[3]

    for rec, id, lbl in zip(rect, ids, labels):
        start_point = (rec[0], rec[1])
        end_point = (rec[2], rec[3])
        color = (0, 255, 0)
        thickness = 1
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1 / 3
        color = (0, 0, 0)
        thickness = 1
        img = cv2.putText(img, '{} ({})'.format(lbl, id), start_point, font, fontScale, color, thickness, cv2.LINE_AA)

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test_dataset()