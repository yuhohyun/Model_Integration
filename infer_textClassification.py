import torch
from transformers import ElectraModel, ElectraTokenizer
import sys

sys.path.append('./models/text_classification')
from models.text_classification.model import CustomTransformerModel

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
pretrained_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")

num_labels = 3
num_cls_vector = pretrained_model.config.hidden_size

model = CustomTransformerModel(pretrained_model, num_labels, num_cls_vector)

# 상태 사전 로드 전에 예상치 못한 키 제거
state_dict = torch.load('models/text_classification/model_textClassification.pth', map_location=torch.device('cpu'))
state_dict.pop("model.embeddings.position_ids", None)  # 'model.embeddings.position_ids' 키가 있으면 제거

model.load_state_dict(state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def infer_text_classification(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    # logits = outputs.logits 대신 outputs를 직접 사용
    logits = outputs
    probs = torch.nn.functional.softmax(logits, dim=-1)
    class_idx = torch.argmax(probs, dim=-1).item()
    
    labels = ['negative', 'neutral', 'positive']
    return labels[class_idx]


text = input("Enter the text: ")
result = infer_text_classification(text)
print(f"Predicted class: {result}")
