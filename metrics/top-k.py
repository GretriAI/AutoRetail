import torch

def compute_topk_accuracy_torch(predictions, labels, k=(1, 5)):
    """
    使用PyTorch计算Top-1和Top-5准确率
    """
    with torch.no_grad():
        maxk = max(k)
        batch_size = labels.size(0)
        
        # 获取概率最高的maxk个预测的索引
        _, pred = predictions.topk(maxk, 1, True, True)
        pred = pred.t()
        
        # 检查预测是否正确
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        
        res = {}
        for k_i in k:
            correct_k = correct[:k_i].reshape(-1).float().sum(0, keepdim=True)
            res[f'top_{k_i}'] = correct_k.mul_(100.0 / batch_size).item()
        
        return res

# 使用示例
# predictions: (batch_size, num_classes) 概率或logits
# labels: (batch_size,)
# result = compute_topk_accuracy_torch(predictions, labels, k=(1, 5))