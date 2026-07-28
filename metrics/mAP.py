def compute_mAP_at_different_thresholds(all_detections, all_groundtruths):
    """
    计算不同IoU阈值下的mAP
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, 0.6, ..., 0.95]
    
    map_results = {}
    map_50_95 = []
    
    for iou_thr in iou_thresholds:
        map_val = compute_map(all_detections, all_groundtruths, iou_thr)
        if iou_thr == 0.5:
            map_results['mAP50'] = map_val
        if iou_thr == 0.75:
            map_results['mAP75'] = map_val
        map_50_95.append(map_val)
    
    map_results['mAP50-95'] = np.mean(map_50_95)
    return map_results
