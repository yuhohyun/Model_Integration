import torch
from torchvision import transforms
from PIL import Image
import sys

# 모델 클래스가 정의된 경로 추가
sys.path.append('./models/image_classification')
from models.image_classification.model import modified_PAtt_Lite  # model.py에서 모델 클래스를 가져옴

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
    images = torch.stack(images)  # 여러 이미지를 하나의 배치로 합침
    return images

def infer(model, image_paths):
    """추론을 수행하고 결과를 출력하는 함수"""
    model.eval()  # 모델을 추론 모드로 설정
    with torch.no_grad():  # 추론 중에는 기울기 계산을 하지 않음
        images = load_images(image_paths)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        return predictions

# 모델 인스턴스 생성 및 가중치 로드
model = modified_PAtt_Lite(num_classes=3, pretrained=False)
model.load_state_dict(torch.load('models/image_classification/model_imageClassification.pth', map_location=torch.device('cpu')))

# 추론할 이미지 경로
image_paths = [
    'C:/Project/Facial_Expression_Recognition/data/test/0cebc58b878df61c71de8e92bf181af059fe272b432e0ff5bab15d4aa12504f6__20___20201202162128-010-006.jpg',
    'C:/Project/Facial_Expression_Recognition/data/test/8be437645d02eee9dddd2cccda6657c3a08e162cdb36b7a9a9c76473582cbb30__20__&()_20210126212344-002-009.jpg',
    'C:/Project/Facial_Expression_Recognition/data/test/e10c6634bc727f46e79ccde300fcf949df69a3634659444e098016d7323871dc__20__&&_20201207011747-001-008.jpg'
]

# 추론 실행
predictions = infer(model, image_paths)
for i, prediction in enumerate(predictions):
    print(f'Image {i+1} prediction: {prediction.item()}')
