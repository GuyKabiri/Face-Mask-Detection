import os
import time
import gc
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split#, KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from data_handler.FaceMaskDataset import FaceMaskDataset

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transformer(phase):
    if phase == 'train':
        return A.Compose([
                        A.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                        ToTensorV2(p=1.0),
                ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

    return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_writers(path, model_name, fold=None):
    if fold is not None:
        return { phase: SummaryWriter('{}/logs/{}_fold_{}_{}'.format(path,
                                                                    model_name,
                                                                    fold,
                                                                    phase))
                                                                            for phase in ['train', 'valid'] }

    return { phase: SummaryWriter('{}/logs/{}_{}'.format(path, model_name, phase))
                                                                                    for phase in ['train', 'valid'] }


def create_env(path):
    if not os.path.exists(path):
        os.mkdir(path)
    paths = ['logs', 'models', 'metrics']
    for p in paths:
        sub_path = os.path.join(path, p)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)


def train_epochs(model, loaders, writers, optimizer, path, config, scheduler=None, prev_loss=float('inf')):
    
    device = config.device
    model = model.to(device).train()

    best_loss = prev_loss

    #   iterate epochs
    for epch in range(1, config.n_epochs + 1):
        print('Epoch {:3d} of {}:'.format(epch, config.n_epochs), flush=True)
             
        epoch_print = ''
        #   iterate phases
        for phase in ['train', 'valid']:

            with tqdm(total=len(loaders[phase]), desc=phase) as progress_bar:
                samples = 0
                epoch_losses = dict()
                accum_loss = 0.0

                #   iterate batches
                for imgs, annts in loaders[phase]:                      #   get next batch
                    imgs  = list(img.to(device) for img in imgs)    #   move images to GPU
                    annts = [{k: v.to(device) for k, v in t.items()} for t in annts]    #   move targets to GPU

                    batch_size = len(imgs)  
                    samples += batch_size

                    #   calculate batch losses
                    if phase == 'train':
                        loss_dict = model(imgs, annts)
                    else:
                        with torch.no_grad():
                            loss_dict = model(imgs, annts) 

                    losses = sum(loss for loss in loss_dict.values())       #   sum total of all batch loseses
                    if phase == 'train':
                        optimizer.zero_grad()
                        losses.backward()
                        optimizer.step()

                    accum_loss += losses.item()                             #   aggregate to get epoch loss at the end
                    for name, val in loss_dict.items():
                        if name in epoch_losses:
                            epoch_losses[name] += val
                        else:
                            epoch_losses[name] = val

                    del imgs, annts, loss_dict, losses
                    torch.cuda.empty_cache()
                    progress_bar.update(1)


            #   end of epoch, next section will run twice,
            #   once for training phase and once for validation phase of each epoch

            #   print to terminal and summary writers at end of each wpoch
            epoch_print += phase + ':\t'
            for key, val in epoch_losses.items():
                writers[phase].add_scalar(key, val, epch)
                epoch_print += '{}={:.5f}\t'.format(key, val)
            epoch_print += 'total loss={:.5f}{}'.format(accum_loss, '\n' if phase == 'train' else '')
            writers[phase].add_scalar('average_loss', accum_loss, epch)

            del epoch_losses

        #   print outputs to the screen after done both training and validation phases of each epoch
        print(epoch_print, flush=True)
        
        #   write learning rate to the summary writer
        if scheduler is not None:
            writers['train'].add_scalar('lr_epoch', scheduler.get_last_lr()[0], epch)
            scheduler.step()   
        
        # if the model perform better in this epoch, save it's parameters
        if accum_loss < best_loss:
            saveing_path = '{}/models/{}_model.pth'.format(path, config.model_name)
            print('Model saved. Loss < PrevLoss ({:.5f} < {:.5f})\n'.format(accum_loss, best_loss))
            best_loss = accum_loss
            torch.save(model.state_dict(), saveing_path)
        time.sleep(1)
    
    return best_loss

    
def get_dataloaders(x_train, x_valid, y_train, y_valid, config):
    trainset = FaceMaskDataset(x_train, y_train, config.imgs_path, config.msks_path, config.img_width, config.img_height, transforms=get_transformer('train'))
    validset = FaceMaskDataset(x_valid, y_valid, config.imgs_path, config.msks_path, config.img_width, config.img_height, transforms=get_transformer('valid'))

    train_loader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=validset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

    train_loader = DataLoader(dataset=trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=validset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, collate_fn=collate_fn)

    return {    'train':    train_loader,
                'valid':    valid_loader }


def train_folds(model, x, y, path, config, scheduler=None):
    print('This running path is: `{}`\n'.format(path))
    
    if config.n_folds == 1:
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size=config.train_size, random_state=config.seed)
        dataloaders = get_dataloaders(x_train, x_valid, y_train, y_valid, config)
        writers = get_writers(path, config.model_name)

        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        train_epochs(model, dataloaders, writers, optimizer, path, config, scheduler)

    else:
        kfold = MultilabelStratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)

        (y_annts, y_labels) = y

        prev_loss = float('inf')

        #   iterate folds
        for fold, (train_index, valid_index) in enumerate(kfold.split(x, y_labels), start=1): 
            print('\033[1m\033[4mFold {} of {}\033[0m'.format(fold, config.n_folds))

            #   get different training and validation writers for each fold
            writers = get_writers(path, config.model_name, fold)

            #   getting fold's data
            x_train, x_valid = x[train_index], x[valid_index]
            y_train, y_valid = y_annts[train_index], y_annts[valid_index]
            dataloaders = get_dataloaders(x_train, x_valid, y_train, y_valid, config)

            optimizer = get_optimizer(model, config)
            scheduler = get_scheduler(optimizer, config)

            prev_loss = train_epochs(model, dataloaders, writers, optimizer, path, config, scheduler, prev_loss)

            del x_train, x_valid, y_train, y_valid

            #   saving model's state each fold
            saveing_path = '{}/models/{}_fold_{}_model.pth'.format(path, config.model_name, fold)
            torch.save(model.state_dict(), saveing_path)

    for _, w in writers.items():
        w.close()


def train(model, x, y, path, config):
    create_env(path)
    config.save(path)
    try:
        train_folds(model, x, y, path, config)
    except Exception as ex:
        torch.cuda.empty_cache()
        print(ex)
    gc.collect()


def get_model(num_classes, pretrained=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained) #   get model
    in_features = model.roi_heads.box_predictor.cls_score.in_features                   #   get input size of last layer
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)         #   regenerate the last layer
    return model


def get_optimizer(model, config):
    params = [p for p in model.parameters() if p.requires_grad]         #   get optimizeable paramaters
    return config.optimizer(params, **config.optimizer_dict)


def get_scheduler(optimizer, config):
    if not config.scheduler:
        return None
    return config.scheduler(optimizer, **config.scheduler_dict)