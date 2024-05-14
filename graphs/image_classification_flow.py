import torch
import torch.nn as nn
import torchvision.models as models

from torchviz import make_dot

class PAtt_Lite(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(PAtt_Lite, self).__init__()
        
        self.base_model = models.mobilenet_v2(pretrained=pretrained).features  # 사전 학습된 모델 불러오기 옵션 수정
        self.base_model[0][0].stride = (1, 1)  # Adjusting stride for fine-tuning
        
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.patch_extraction = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=4, stride=4, padding=16),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=2, padding='valid'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding='valid'),
            nn.ReLU()
        )
        self.global_average_layer = nn.AdaptiveAvgPool2d((1, 1))
        self.pre_classification = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        # Self-attention layer 추가
        self.self_attention = nn.MultiheadAttention(embed_dim=32, num_heads=1)
        
        self.classification_head = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.base_model(x)
        x = self.patch_extraction(x)
        x = self.global_average_layer(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.pre_classification(x)
        
        # Self-attention 적용
        x = x.unsqueeze(1)  # MultiheadAttention을 사용하기 위해 차원 증가
        x, _ = self.self_attention(x, x, x)
        x = x.squeeze(1)  # 차원 복원
        
        x = self.dropout(x)
        x = self.classification_head(x)
        return x


# 모델 인스턴스 생성
model = PAtt_Lite(num_classes=3)

# 모델에 랜덤 데이터 전달하여 계산 그래프 생성
x = torch.randn(16, 3, 224, 224)  # 예시로 input 크기를 지정
out = model(x)

# 계산 그래프 시각화
dot = make_dot(out, params=dict(model.named_parameters()))
dot.render("model_graph")
