class PersonFeatureManager:
    def __init__(self, buffer_size=10, momentum=0.9):
        self.gallery = {} # {id: {"features": [], "ema_feat": None}}
        self.buffer_size = buffer_size
        self.momentum = momentum

    def update_feature(self, person_id, new_feat):
        if person_id not in self.gallery:
            self.gallery[person_id] = {
                "features": [new_feat],
                "ema_feat": new_feat
            }
        else:
            # 1. EMA 平滑更新全局特征 (应对光照等小幅变化)
            old_ema = self.gallery[person_id]["ema_feat"]
            self.gallery[person_id]["ema_feat"] = self.momentum * old_ema + (1 - self.momentum) * new_feat
            
            # 2. 滚动维护特征池 (应对姿态大幅改变)
            feat_list = self.gallery[person_id]["features"]
            if len(feat_list) >= self.buffer_size:
                feat_list.pop(0)
            feat_list.append(new_feat)

    def get_match_score(self, person_id, current_feat):
        # 匹配时，取 EMA 特征与特征池均值的加权结果
        pool_mean = np.mean(self.gallery[person_id]["features"], axis=0)
        ema_feat = self.gallery[person_id]["ema_feat"]
        
        # 与当前特征计算相似度
        score_ema = cosine_similarity(current_feat, ema_feat)
        score_pool = cosine_similarity(current_feat, pool_mean)
        return 0.4 * score_ema + 0.6 * score_pool