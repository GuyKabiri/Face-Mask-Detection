import os
import time
import gc
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

def get_writers(path, model_name):
    writer_train_epoch = SummaryWriter('{}/logs/{}_train'.format(path, model_name))
    writer_valid_epoch = SummaryWriter('{}/logs/{}_valid'.format(path, model_name))

    return {    'train': writer_train_epoch,
                'valid': writer_valid_epoch }

def create_env(path):
    if not os.path.exists(path):
        os.mkdir(path)
    paths = ['logs', 'models']
    for p in paths:
        sub_path = os.path.join(path, p)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

def train2(model, loader, optimizer, path, configurations, scheduler=None):
    device = configurations.device
    model = model.to(device).train()


    for epoch in range(configurations.n_epochs) :
        epoch_loss = 0
        
        for imgs, annotations in tqdm(loader):
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            predict = model(imgs, annotations)
            losses = sum(loss for loss in predict.values())        

            optimizer.zero_grad()
            losses.backward()
            optimizer.step() 
            epoch_loss += losses
            
        print(epoch+1, '/', configurations.n_epochs, ' : {:.5f}'.format(epoch_loss))
        if epoch_loss < 0.1 :
            print('early stop')
            break



def wrapped_train(model, loaders, optimizer, path, configurations, scheduler):
    print('This running path is: `{}`\n'.format(path))
    time.sleep(1)
    create_env(path)
    writers = get_writers(path, configurations.model_name)

    device = configurations.device
    model = model.to(device).train()

    best_loss = 10000

    for epch in range(1, configurations.n_epochs + 1):              #   iterate epoch
        print('Epoch {:3d} of {}:'.format(epch, configurations.n_epochs), flush=True)
             
        epoch_print = ''
        for phase in ['train', 'valid']:    #   iterate phases

            with tqdm(total=len(loaders[phase]), desc=phase) as progress_bar:               #   define progress bas
                samples = 0
                epoch_losses = dict()
                accum_loss = 0.0

                for imgs, annts in loaders[phase]:                                     #   iterate batches
                    imgs  = list(img.to(device) for img in imgs)
                    
                    annts = [{k: v.to(device) for k, v in t.items()} for t in annts]

                    batch_size = len(imgs)
                    samples += batch_size

                    if phase == 'train':
                        loss_dict = model(imgs, annts)                          #   calculate batch losses
                    else:
                        with torch.no_grad():
                            loss_dict = model(imgs, annts) 

                    losses = sum(loss for loss in loss_dict.values())       #   sum total of all batch loseses
                    if phase == 'train':
                        optimizer.zero_grad()
                        losses.backward()
                        optimizer.step()

                    accum_loss += losses.item()                             #   add to get epoch loss at the end
                    #   accumulate losses to get final loss of epoch
                    for name, val in loss_dict.items():
                        if name in epoch_losses:
                            epoch_losses[name] += val #/ batch_size
                        else:
                            epoch_losses[name] = val #/ batch_size

                    del imgs, annts, loss_dict, losses
                    torch.cuda.empty_cache()
                    progress_bar.update(1)

            # accum_loss /= samples

            epoch_print += phase + ':\t'
            for key, val in epoch_losses.items():
                writers[phase].add_scalar(key, val, epch)
                epoch_print += '{}={:.5f}\t'.format(key, val)
            epoch_print += 'total loss={:.5f}\n'.format(accum_loss)
            writers[phase].add_scalar('average_loss', accum_loss, epch)

            del epoch_losses

        print(epoch_print, flush=True)
        
        if scheduler is not None:
            writers['train'].add_scalar('lr_epoch', scheduler.get_last_lr()[0], epch)
            scheduler.step()

        # saveing_path = '{}/models/{}_epoch_{}.pth'.format(path, configurations.model_name, epch)
        # torch.save(model.state_dict(), saveing_path)     
        
        # if the model perform better in this epoch, save it's parameters
        if accum_loss < best_loss:
            saveing_path = '{}/models/{}_model.pth'.format(path, configurations.model_name)
            print('Model saved. Loss < PrevLoss ({:.5f} < {:.5f})'.format(accum_loss, best_loss))
            best_loss = accum_loss
            torch.save(model.state_dict(), saveing_path)


def train(model, loaders, optimizer, path, configurations, scheduler=None):
    try:
        wrapped_train(model, loaders, optimizer, path, configurations, scheduler)
    except Exception:
        torch.cuda.empty_cache()
    gc.collect()