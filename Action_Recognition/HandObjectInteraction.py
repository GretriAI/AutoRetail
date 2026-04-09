def check_interaction(hand_box, item_box, item_velocity):
    """
    hand_box: [x1, y1, x2, y2]
    item_box: [x1, y1, x2, y2]
    item_velocity: 商品在连续3帧中的位移向量
    """
    # 1. 计算手与商品的 IoU (或距离)
    iou = calculate_iou(hand_box, item_box)
    
    # 2. 状态判定
    if iou > 0.3: # 手部触碰物体
        if np.linalg.norm(item_velocity) > threshold:
            return "PICK_UP"  # 商品随手部移动
        else:
            return "TOUCHING" # 仅触碰，未拿走
    return "NONE"
