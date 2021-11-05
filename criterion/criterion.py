import torch
import torch.nn.functional as F
from torchvision.ops.boxes import _box_inter_union

def giou_loss(input_boxes, target_boxes, eps=1e-7):
    """
    Args:
        input_boxes: Tensor of shape (N, 4) or (4,).
        target_boxes: Tensor of shape (N, 4) or (4,).
        eps (float): small number to prevent division by zero
    """
    inter, union = _box_inter_union(input_boxes, target_boxes)
    iou = inter / union

    # area of the smallest enclosing box
    min_box = torch.min(input_boxes, target_boxes)
    max_box = torch.max(input_boxes, target_boxes)
    area_c = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

    giou = iou - ((area_c - union) / (area_c + eps))

    loss = 1 - giou

    return loss.sum()


def focal_loss(logits, target, alpha=1, gamma=2, eps=1e-12):
    probs = torch.sigmoid(logits)
    one_subtract_probs = 1.0 - probs
    # add epsilon
    probs_new = probs + eps
    one_subtract_probs_new = one_subtract_probs + eps
    # calculate focal loss
    log_pt = target * torch.log(probs_new) + (1.0 - target) * torch.log(one_subtract_probs_new)
    pt = torch.exp(log_pt)
    focal_loss = -1.0 * (alpha * (1 - pt) ** gamma) * log_pt
    return torch.mean(focal_loss)


def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]


    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = focal_loss(class_logits, labels)
    box_loss = giou_loss(box_regression, regression_targets)


    return classification_loss, box_loss