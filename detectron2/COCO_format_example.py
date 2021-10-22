import torch, torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt
from COCOFaceMaskData import FaceMaskData
from detectron2.utils.visualizer import Visualizer

print(torch.__version__, torchvision.__version__, torch.cuda.is_available())


imgs_path = './images'
msks_path = './annotation.csv'

faceMasksData = FaceMaskData(imgs_path, msks_path)

from detectron2.data import DatasetCatalog, MetadataCatalog

DatasetCatalog.register('mask_dataset', lambda: faceMasksData.load_data(train_size=1))
MetadataCatalog.get('mask_dataset').set(thing_classes=faceMasksData.classes)
chess_metadata = MetadataCatalog.get('mask_dataset')

dataset_dicts = DatasetCatalog.get('mask_dataset')

rows = 2
cols = 5
fig, ax = plt.subplots(rows, cols, figsize = (14, 10))

for i, d in enumerate(np.random.choice(dataset_dicts, rows*cols)):
    rid, cid = i // cols, i % cols
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=chess_metadata)
    vis = visualizer.draw_dataset_dict(d)
    ax[rid, cid].imshow(cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    ax[rid, cid].axis('off')
plt.tight_layout()
plt.show()



# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)