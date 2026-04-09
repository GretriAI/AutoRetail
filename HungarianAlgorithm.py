import numpy as np
from scipy.optimize import linear_sum_assignment

def spatial_temporal_matching(track_features, detect_features, track_boxes, detect_boxes, alpha=0.7):
    """
    alpha: 特征相似度权重; (1-alpha): 空间距离权重
    """
    # 1. 计算特征余弦距离
    cost_feat = 1 - cosine_similarity(track_features, detect_features)
    
    # 2. 计算空间距离 (欧氏距离或中心点偏移)
    cost_dist = np.zeros((len(track_boxes), len(detect_boxes)))
    for i, t_box in enumerate(track_boxes):
        for j, d_box in enumerate(detect_boxes):
            # 计算中心点距离
            dist = np.linalg.norm(t_box.center - d_box.center)
            cost_dist[i, j] = dist
            
    # 3. 综合代价矩阵
    # 给空间距离做一个门限限制 (超过阈值直接设为极大值，防止 ID 跳跃)
    cost_dist[cost_dist > 200] = 1e5 
    
    total_cost = alpha * cost_feat + (1 - alpha) * cost_dist
    
    # 4. 匈牙利算法求解最优匹配
    row_ind, col_ind = linear_sum_assignment(total_cost)
    return row_ind, col_ind