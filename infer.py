import torch
from torchvision import transforms
from PIL import Image
import sys
from transformers import ElectraModel, ElectraTokenizer, PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration

# 모델 클래스가 정의된 경로 추가
sys.path.append('./models/image_classification')
sys.path.append('./models/text_classification')

from models.image_classification.model import modified_PAtt_Lite # 표정 분류 모델 import
from models.text_classification.model import CustomTransformerModel # 텍스트 감정 분석 모델 import

# 이미지 분류 모델 로드
image_model = modified_PAtt_Lite(num_classes=3, pretrained=False)
image_model.load_state_dict(torch.load('models/image_classification/model_imageClassification.pth', map_location=torch.device('cpu')))
image_model.eval()

# 텍스트 분류 모델 로드
tokenizer_text_classification = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
pretrained_text_classification_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")

num_labels = 3
num_cls_vector = pretrained_text_classification_model.config.hidden_size

text_classification_model = CustomTransformerModel(pretrained_text_classification_model, num_labels, num_cls_vector)

state_dict = torch.load('models/text_classification/model_textClassification.pth', map_location=torch.device('cpu'))
state_dict.pop("model.embeddings.position_ids", None) 

text_classification_model.load_state_dict(state_dict)
text_classification_model.eval()

# 텍스트 요약 모델 로드
model_finetuned = BartForConditionalGeneration.from_pretrained('./models/text_summarization')
tokenizer_summarization = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')

def load_images(image_paths):
    """여러 이미지를 불러오고 전처리하는 함수"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        images.append(image)
    images = torch.stack(images)
    return images

def infer_image_classification(model, image_paths):
    """이미지 분류 추론을 수행하고 결과를 출력하는 함수"""
    model.eval()
    with torch.no_grad():
        images = load_images(image_paths)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        return predictions

def infer_text_classification(text):
    inputs = tokenizer_text_classification(text, return_tensors='pt', padding=True, truncation=True)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    inputs = {k: v for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = text_classification_model(**inputs)
    logits = outputs
    probs = torch.nn.functional.softmax(logits, dim=-1)
    class_idx = torch.argmax(probs, dim=-1).item()
    
    labels = ['negative', 'neutral', 'positive']
    return labels[class_idx]

def infer_text_summarization(text):
    print("========== Original Text ==========")
    print(text)
    print('\n')

    if text:
        input_ids = tokenizer_summarization.encode(text)
        input_ids = torch.tensor(input_ids)
        input_ids = input_ids.unsqueeze(0)
        output = model_finetuned.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
        output = tokenizer_summarization.decode(output[0], skip_special_tokens=True)    
        print("========== Summary ==========")
        print(output)

print("\nSelect the task:")
print("1: Image Classification")
print("2: Text Classification")
print("3: Text Summarization")
print("0: Exit")

while True:
    task_number = int(input("\nEnter the number of the task (or 0 to exit): "))
    if task_number == 1:  # 이미지 분류
    # 추론할 이미지 경로
        image_paths = [
        'C:/Project/Facial_Expression_Recognition/data/test/0cebc58b878df61c71de8e92bf181af059fe272b432e0ff5bab15d4aa12504f6__20___20201202162128-010-006.jpg',
        'C:/Project/Facial_Expression_Recognition/data/test/8be437645d02eee9dddd2cccda6657c3a08e162cdb36b7a9a9c76473582cbb30__20__&()_20210126212344-002-009.jpg',
        'C:/Project/Facial_Expression_Recognition/data/test/e10c6634bc727f46e79ccde300fcf949df69a3634659444e098016d7323871dc__20__&&_20201207011747-001-008.jpg'
        ]
        predictions = infer_image_classification(image_model, image_paths)
        for i, prediction in enumerate(predictions):
            # 예측 결과를 문자열로 변환
            if prediction.item() == 0:
                prediction_str = 'negative'
            elif prediction.item() == 1:
                prediction_str = 'neutral'
            elif prediction.item() == 2:
                prediction_str = 'positive'
            else:
                prediction_str = 'Unknown'
            print(f'Image {i+1} prediction: {prediction_str}')
    elif task_number == 2:  # 텍스트 분류
        text = input("Enter the text for classification: ")
        result = infer_text_classification(text)
        print(f"Predicted class: {result}")
    elif task_number == 3:  # 텍스트 요약
        text = input("Enter the text for summarization: ")
        infer_text_summarization(text)
    elif task_number == 0:  # 프로그램 종료
        print("Exiting the program.")
        break
    else:
        print("Invalid task selected.")

