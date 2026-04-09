class ActionStateInference:
    def __init__(self):
        self.state = "IDLE" # 状态：空闲, 伸向, 抓取中, 持有

    def update(self, visibility, hand_empty_prob, zone):
        """
        visibility: 遮挡程度 (0-1)
        hand_empty_prob: 手部为空的概率 (来自分类器)
        """
        # 如果当前被遮挡 (visibility < 0.2)
        if visibility < 0.2:
            # 保持上一个状态，并标记为“推测中”
            return f"INFERRING_{self.state}"
            
        # 遮挡结束后的逻辑判定
        if self.state == "IDLE" and hand_empty_prob < 0.2 and zone == "Shelf":
            self.state = "CARRYING"
            return "PICK_CONFIRMED"
            
        return self.state
