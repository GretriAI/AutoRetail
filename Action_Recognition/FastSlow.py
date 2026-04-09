import torch.nn as nn

class ActionNet(nn.Module):
    def __init__(self, num_classes=4): # 动作：伸向、抓取、放入、放回
        super().__init__()
        # Slow pathway: 处理空间特征 (特征图大，帧率低)
        self.slow_path = nn.Sequential(nn.Conv3d(3, 64, (1, 7, 7)), nn.ReLU())
        # Fast pathway: 处理时序特征 (特征图小，帧率高)
        self.fast_path = nn.Sequential(nn.Conv3d(3, 8, (5, 7, 7)), nn.ReLU())
        
        self.fc = nn.Linear(64 + 8, num_classes)

    def forward(self, x_slow, x_fast):
        f_slow = self.slow_path(x_slow).mean(dim=[2,3,4])
        f_fast = self.fast_path(x_fast).mean(dim=[2,3,4])
        # 特征融合
        combined = torch.cat([f_slow, f_fast], dim=1)
        return self.fc(combined)
