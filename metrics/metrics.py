import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.ops as ops

def get_pred_by_threshold(preds, threshold):
    keep = ops.nms(preds['boxes'], preds['scores'], threshold)
    
    final_prediction = preds
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

def get_pred_by_threshold_batch(preds, threshold):
    
    for i in range(len(preds)):
        preds[i] = get_pred_by_threshold(preds[i], threshold)
        
    return preds


def get_iou_as_df(model, loader, nms_thresh):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    df = pd.DataFrame(columns=['image_id', 'real_class', 'pred_class', 'iou', 'score'])
    
    with tqdm(total=len(loader), desc='evaluating IOU') as progress_bar:               #   define progress bas
        for img_batch, target_batch in loader:
            img_batch  = list(img.to(device) for img in img_batch)

            output_batch = model(img_batch)
            output_batch = get_pred_by_threshold_batch(output_batch, nms_thresh) # remove duplicated boxes
            
            output_batch_cpu = [{k: v.cpu().detach() for k, v in t.items()} for t in output_batch]

            for target, output in zip(target_batch, output_batch_cpu):
                
                ious =  ops.box_iou(output['boxes'], target['boxes']) # get IOU of each bbox
                ious = torch.max(ious, dim=1)
                
                for idx in range(len(ious.values)):
                    id_real_bbox = ious.indices[idx] # to which target box this prediction related
                    df = df.append({
                        'image_id':   target['image_id'][0],
                        'real_class': target['labels'][id_real_bbox],    # target['labels'][ious_ids[idx]]
                        'pred_class': output['labels'][idx],
                        'iou':        ious.values[idx],
                        'score':      output['scores'][idx],                
                    }, ignore_index=True)
                    
                

                del target, output

            del img_batch, target_batch, output_batch, output_batch_cpu
            torch.cuda.empty_cache()
            progress_bar.update(1)
    
    df = df.astype({ 'image_id': np.int32,
                     'real_class': np.int32,
                     'pred_class': np.int32, 
                     'iou':        np.float32,
                     'score':      np.float32 })
    return df.sort_values('score', ascending=False, ignore_index=True)


def calc_precision_recall(df, iou_thresh, eps=1e-6):
    df['precision'] = 0.
    df['recall'] = 0.
    df['TPorFP'] = np.where((df['real_class'] == df['pred_class']) & (df['iou'] >= iou_thresh), True, False)

    total_tp_cls = { c: len(df[(df['real_class']==c) & (df['TPorFP'])]) for c in df['real_class'].unique() }
    count_cls_tp = { c: 0 for c in df['real_class'].unique() }
    count_cls_instance = { c: 0 for c in df['real_class'].unique() }

    for index, row in df.iterrows():
        row_class = row['real_class']
        count_cls_instance[row_class] += 1   # count instances of each class
        if row['TPorFP']:
            count_cls_tp[row_class] += 1     # count TP of each class
        df.at[index, 'precision'] =  count_cls_tp[row_class] / count_cls_instance[row_class]
        df.at[index, 'recall'] = count_cls_tp[row_class] / (total_tp_cls[row_class] + eps)

    df.to_csv('metrics/map_thresh_{:.3}.csv'.format(iou_thresh), index=False)
    return df