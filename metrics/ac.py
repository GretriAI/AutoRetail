import numpy as np

def compute_topk_accuracy(predictions, labels, k=1):
    """
    计算Top-k准确率
    
    predictions: 模型输出的概率矩阵，shape=(N, C)，N为样本数，C为类别数
    labels: 真实标签，shape=(N,)
    k: 取前k个预测
    """
    # 获取预测概率最高的k个类别的索引
    topk_preds = np.argsort(predictions, axis=1)[:, -k:]  # 取最后k个（概率最高的）
    
    # 检查真实标签是否在top-k预测中
    correct = 0
    for i in range(len(labels)):
        if labels[i] in topk_preds[i]:
            correct += 1
    
    accuracy = correct / len(labels)
    return accuracy

# 使用示例
predictions = np.array([
    [0.1, 0.7, 0.2],  # 预测类别1概率最高
    [0.3, 0.3, 0.4],  # 预测类别2概率最高
    [0.8, 0.1, 0.1]   # 预测类别0概率最高
])
labels = np.array([1, 2, 0])  # 真实类别

top1_acc = compute_topk_accuracy(predictions, labels, k=1)
top5_acc = compute_topk_accuracy(predictions, labels, k=5)  # 如果类别数>=5

print(f"Top-1 Accuracy: {top1_acc:.4f}")  # 输出: 1.0（全部预测正确）