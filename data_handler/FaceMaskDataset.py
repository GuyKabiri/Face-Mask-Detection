import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from matplotlib import pyplot as plt
from xml.etree import ElementTree as et



class FaceMaskDataset(Dataset):

    '''
        Params:
            -   images      -   list of images' file names
            -   annotations -   list of annotations' file names
            -   img_dit     -   directory of images.
            -   annt_dit    -   directory of annotations.
            -   width       -   the desired width of the images as an input to the model
            -   height      -   the desired height of the images as an input to the model
            -   transforms  -   transformations to perform on both the images and the annotations
    '''
    def __init__(self, images, annotations, img_dir, annt_dir, width, height, transforms=None):
        self.transforms = transforms
        self.imgs = images
        self.ants = annotations
        self.images_dir = img_dir
        self.annotation_dir = annt_dir
        self.height = height
        self.width = width
           
        #   class 0 for background
        self.classes = [None, 'without_mask','with_mask','mask_weared_incorrect']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.images_dir, img_name)                        #   get the exact image path

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)               #   convert to RGB
        img_res = cv2.resize(img, (self.width, self.height), cv2.INTER_AREA)        #   resize to desired size (should modify boundry boxes as well)
        # img_res /= 255.0    #   devide so image values will be between [0, 1]
        
        annt_filename = self.ants[idx]
        annt_file_path = os.path.join(self.annotation_dir, annt_filename)         #   get the exact annotation path
        
        boxes = []
        labels = []
        tree = et.parse(annt_file_path)     #   open as dictionary
        root = tree.getroot()
        
        org_h, org_w = img.shape[0], img.shape[1]   #   original width and height of the image
        
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))     #   append the box text label to the array
            
            xmin = int(member.find('bndbox').find('xmin').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            #   convert corrdinations to image's new size
            xmin_corr = (xmin/(org_w+1))*(self.width-1)
            ymin_corr = (ymin/(org_h+1))*(self.height-1)
            xmax_corr = (xmax/(org_w+1))*(self.width-1)
            ymax_corr = (ymax/(org_h+1))*(self.height-1)
            
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])      #   append the box to the array
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])    #   calculate area of the boxes
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        #   place all data in a dictionary
        target = {
            'boxes':    boxes,
            'labels':   labels,
            'area':     area,
            'iscrowd':  iscrowd,
            'image_id': image_id
        }

        if self.transforms:   
            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)
            
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
            
            
        return img_res, target

    def __len__(self):
        return len(self.imgs)