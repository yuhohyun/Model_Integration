import torch
import torch.nn as nn

class CustomTransformerModel(nn.Module):
    def __init__(self, model, num_labels, num_cls_vector):
        super(CustomTransformerModel, self).__init__()
        self.num_labels = num_labels
        self.num_cls_vector = num_cls_vector  # 이 줄을 추가하여 num_cls_vector를 클래스 변수로 저장

        self.model = model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(num_cls_vector, num_labels)  # load and initialize weights

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Add custom layers
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state
        # 제공할 최종 logits 계산
        logits = self.classifier(sequence_output[:, 0, :].view(-1, self.num_cls_vector))  # 여기서 self.num_cls_vector를 사용

        return logits