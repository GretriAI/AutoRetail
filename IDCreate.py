import torch
from ultralytics import YOLO
from reid_model import BuildReIDModel # 假设已定义的Re-ID网络

# 1. 加载检测和Re-ID模型
detector = YOLO('yolov8n.pt')
reid_net = BuildReIDModel(name='osnet_x1_0', num_classes=0) # 推理模式
reid_net.load_state_dict(torch.load('osnet_weights.pth'))
reid_net.eval()

def register_new_person(frame):
    results = detector(frame)
    for box in results[0].boxes:
        if box.cls == 0:  # Class 0 是行人
            xtl, ytl, xbr, ybr = box.xyxy[0]
            # 裁剪人体区域
            crop = frame[int(ytl):int(ybr), int(xtl):int(xbr)]
            
            # 2. 特征提取
            input_tensor = preprocess(crop) # 缩放至256x128, 归一化
            with torch.no_grad():
                features = reid_net(input_tensor) # 得到 Embedding
            
            # 3. 存储到全局数据库 (Gallary)
            new_id = generate_global_id()
            vector_db.add(id=new_id, vector=features, timestamp=now())