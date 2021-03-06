import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.ops as ops


'''
    apply Non-maximum Suppression by threshold for a single prediction (one image) and return the new predictions
'''
def get_pred_by_threshold(preds, nms_threshold, score_treshold=0):
    keep = ops.nms(preds['boxes'], preds['scores'], nms_threshold)

    keep_score = torch.where(preds['scores'] >= score_treshold)[0]

    mask = torch.Tensor([ 1. if x in keep_score else 0. for x in keep ])
    keep = torch.nonzero(mask).flatten()
    
    final_prediction = preds
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction


'''
    apply Non-maximum Suppression by threshold for batch (batch size can be equals to 1) and return new predictions
'''
def get_pred_by_threshold_batch(preds, nms_threshold, score_treshold=0):
    
    for i in range(len(preds)):
        preds[i] = get_pred_by_threshold(preds[i], nms_threshold, score_treshold)
        
    return preds


'''
    run the entire dataloader in the model and save the output prediction in a dataframe
    will the following columns for a single boundary box:
        image_id, real_class, pred_class, score, iou
'''
def get_iou_as_df(model, loader, nms_thresh, score_threshold=0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    df = pd.DataFrame(columns=['image_id', 'real_class', 'pred_class', 'iou', 'score'])
    
    with tqdm(total=len(loader), desc='evaluating IoU') as progress_bar:               #   define progress bas
        for img_batch, target_batch in loader:
            img_batch  = list(img.to(device) for img in img_batch)

            output_batch = model(img_batch)
            output_batch = get_pred_by_threshold_batch(output_batch, nms_thresh, score_threshold) # remove duplicated boxes
            
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
                        'score':      output['scores'][idx],
                        'iou':        ious.values[idx],                
                    }, ignore_index=True)
                    
                

                del target, output

            del img_batch, target_batch, output_batch, output_batch_cpu
            torch.cuda.empty_cache()
            progress_bar.update(1)
    
    df = df.astype({ 'image_id':   np.int32,
                     'real_class': np.int32,
                     'pred_class': np.int32, 
                     'iou':        np.float32,
                     'score':      np.float32 })
    return df.sort_values('score', ascending=False, ignore_index=True)


'''
    calculate recall and precision for the entire predicted data and return the modified dataframe
'''
def calc_precision_recall(df, iou_thresh, path, eps=1e-6):

    df['precision'] = 0.
    df['recall'] = 0.

    #   evaluate TP/FP of each boundary box by specific IOU threshhold
    df['TP/FP'] = np.where((df['real_class'] == df['pred_class']) & (df['iou'] >= iou_thresh), 'TP', 'FP')

    total_tp_cls = { c: len(df[(df['real_class']==c) & (df['TP/FP'] == 'TP')]) for c in df['real_class'].unique() }    #   calculate the total TP for each class
    count_cls_tp = { c: 0 for c in df['real_class'].unique() }          #   TP counter for each class as iterating over the data
    count_cls_instance = { c: 0 for c in df['real_class'].unique() }    #   instances counter for each class

    #   iterate over the dataframe rows
    for index, row in df.iterrows():
        row_class = row['real_class']       #   get the real class each boundary box referring to
        count_cls_instance[row_class] += 1  #   count instances of each class
        if row['TP/FP'] == 'TP':
            count_cls_tp[row_class] += 1    #   count TP of each class
        
        #   calculate precision and recall for each boundary box
        df.at[index, 'precision'] =  count_cls_tp[row_class] / count_cls_instance[row_class]
        df.at[index, 'recall'] = count_cls_tp[row_class] / (total_tp_cls[row_class] + eps)

    df.to_csv('{}/AP@{:.3}.csv'.format(path, iou_thresh), index=False)
    return df

def calc_mAP_from_auc_dict(auc_dict):
    iou_total = dict()

    for iou in auc_dict:
        iou_sum = sum(auc_dict[iou].values())
        mAP = iou_sum / len(auc_dict[iou])
        iou_total[iou] = mAP
    
    return iou_total


def calc_mAP_per_class(auc_dict, iou_dict):
    mAP_classes = { c: .0 for c in auc_dict[0.5] }
    # total_mAP = .0

    for iou, iou_dict in auc_dict.items():
        for c, val in iou_dict.items():
            mAP_classes[c] += val
            
    for c in mAP_classes:
        mAP_classes[c] /= len(auc_dict)

    return mAP_classes