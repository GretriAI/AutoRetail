import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def verify_at_exit(exit_feature, gallery_db):
    """
    exit_feature: 当前出口抓拍到的特征向量
    gallery_db: 包含 {id: feature_vector} 的字典
    """
    ids = list(gallery_db.keys())
    gallery_features = np.array(list(gallery_db.values()))
    
    # 计算余弦相似度
    scores = cosine_similarity(exit_feature.reshape(1, -1), gallery_features)[0]
    
    best_match_idx = np.argmax(scores)
    max_score = scores[best_match_idx]
    
    threshold = 0.85 # 工业界常用的严格阈值
    if max_score > threshold:
        return ids[best_match_idx], "Match Success"
    else:
        return None, "Unknown ID / Possible ID Switch"