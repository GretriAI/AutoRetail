from ultralytics import YOLO

# 加载模型
model = YOLO('yolo11n.pt')

# 在验证集上评估，直接得到所有指标
results = model.val(data='coco.yaml')

# 输出结果
print(f"mAP50: {results.box.map50:.4f}")      # mAP at IoU=0.5
print(f"mAP75: {results.box.map75:.4f}")      # mAP at IoU=0.75  
print(f"mAP50-95: {results.box.map:.4f}")     # mAP at IoU=0.5:0.95
print(f"mAP (per class): {results.box.maps}")

# 如果是分割模型
# print(f"mAP50 (mask): {results.seg.map50:.4f}")
# print(f"mAP50-95 (mask): {results.seg.map:.4f}")
