def compute_ap(recall, precision):
    """
    计算PR曲线下的面积（AP）
    使用11点插值法（Pascal VOC标准）或连续积分法（COCO标准）
    """
    # 这里使用COCO的连续积分方法
    # 在recall值上插值precision
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # 计算precision的累积最大值（从后往前）
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # 找到recall值变化的位置
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # 计算AP（PR曲线下的面积）
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_map(all_detections, all_groundtruths, iou_threshold=0.5):
    """
    计算mAP（对所有类别取平均）
    
    all_detections: 字典 {class_id: [[confidence, x1, y1, x2, y2], ...]}
    all_groundtruths: 字典 {class_id: [[x1, y1, x2, y2], ...]}
    iou_threshold: IoU阈值（0.5表示mAP50）
    """
    aps = []
    
    for class_id in all_groundtruths.keys():
        detections = all_detections.get(class_id, [])
        groundtruths = all_groundtruths[class_id]
        
        if len(detections) == 0:
            continue
            
        # 按置信度从高到低排序
        detections = sorted(detections, key=lambda x: x[0], reverse=True)
        
        # 标记每个真实框是否已被匹配
        gt_matched = [False] * len(groundtruths)
        
        # 存储每个检测结果的TP/FP
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        for i, det in enumerate(detections):
            conf, *bbox = det
            best_iou = 0
            best_gt_idx = -1
            
            # 找到与该检测框IoU最高的真实框
            for j, gt in enumerate(groundtruths):
                iou = compute_iou(bbox, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # 判断是否为TP
            if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                tp[i] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # 计算precision和recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (len(groundtruths) + 1e-6)
        
        # 计算AP
        ap = compute_ap(recall, precision)
        aps.append(ap)
    
    # mAP = 所有类别AP的平均值
    return np.mean(aps) if aps else 0
