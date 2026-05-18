import torch

def iou_loss(pred, target, eps=1e-7):
    """
    正确的 IoU 损失计算
    pred / target 形状为 (N, 4)，格式为 (x1, y1, x2, y2)
    """
    # 1. 计算两个框的面积
    area_pred = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_target = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    
    # 2. 计算交集（Intersection）的左上角和右下角
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    
    # 3. 计算交集的宽高，如果不相交则为 0
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    
    # 4. 计算并集（Union）
    union = area_pred + area_target - inter
    
    # 5. 计算 IoU
    iou = inter / (union + eps)
    
    # 6. 返回损失
    return 1.0 - iou
