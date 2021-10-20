import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from matplotlib import pyplot as plt
from xml.etree import ElementTree as et



class FaceMaskDataset(Dataset):

    def __init__(self, images, annotations, img_dir, annt_dir, width, height, transforms=None):
        self.transforms = transforms
        self.imgs = images
        self.ants = annotations
        self.images_dir = img_dir
        self.annotation_dir = annt_dir
        self.height = height
        self.width = width
           
        # classes: 0 index is reserved for background
        self.classes = [None, 'without_mask','with_mask','mask_weared_incorrect']

    def __getitem__(self, idx):

        img_name = self.imgs[idx]
        image_path = os.path.join(self.images_dir, img_name)

        # reading the images and converting them to correct size and color    
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # dividing by 255
        img_res /= 255.0
        
        # annotation file
        annot_filename = self.ants[idx]
        annot_file_path = os.path.join(self.annotation_dir, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]
        
        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            # bounding box
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            
            xmin_corr = (xmin/(wt+1))*(self.width-1)
            xmax_corr = (xmax/(wt+1))*(self.width-1)
            ymin_corr = (ymin/(ht+1))*(self.height-1)
            ymax_corr = (ymax/(ht+1))*(self.height-1)

            # def round_val(val):
            #     if val > 1:
            #         return 1.0
            #     elif val < 0:
            #         return 0.0
            #     else:
            #         return val
       
            # xmin_corr = round_val(xmin_corr)
            # xmax_corr = round_val(xmax_corr)
            # ymin_corr = round_val(ymin_corr)
            # ymax_corr = round_val(ymax_corr)


            
            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id


        if self.transforms:
            
            sample = self.transforms(image = img_res,
                                     bboxes = target['boxes'],
                                     labels = labels)
            
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
            
            
        return img_res, target

    def __len__(self):
        return len(self.imgs)