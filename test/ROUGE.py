import pandas as pd
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from rouge import Rouge
from korouge_score import rouge_scorer

# CUDA 사용 가능 확인 및 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델 및 토크나이저 로드
model_original = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization').to(device)
model_finetuned = BartForConditionalGeneration.from_pretrained('./models/text_summarization').to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')

# ROUGE 점수 측정기 초기화
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)

def summarize(text, model):
    """주어진 텍스트에 대해 요약을 수행하고 결과를 반환"""
    if not isinstance(text, str):
        text = str(text)  # 숫자 또는 다른 타입을 문자열로 변환
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    input_ids = input_ids.to(device)  # 입력 데이터를 GPU(device)로 이동
    summary_ids = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text


def calculate_rouge_scores(data_path):
    """TSV 파일을 읽어 각 행에 대해 요약을 수행하고 ROUGE 점수를 계산"""
    data = pd.read_csv(data_path, sep='\t', quoting=3)  # quoting=3 : 모든 문자열을 따옴표 없이 읽으라는 의미
    data = data.dropna(subset=['news', 'summary'])  # 누락된 값이 있는 행을 제거
    data['news'] = data['news'].astype(str)  # 데이터 타입을 문자열로 강제 변환
    data['summary'] = data['summary'].astype(str)
    
    # 점수 집계를 위한 초기화
    total_scores_original = {key: {"precision": 0, "recall": 0, "fmeasure": 0} for key in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
    total_scores_finetuned = {key: {"precision": 0, "recall": 0, "fmeasure": 0} for key in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']}
    num_entries = 0

    for index, row in data.iterrows():
        print(f"Processing row {index}...")
        text = row['news']
        summary_label = row['summary']

        summary_original = summarize(text, model_original)
        summary_finetuned = summarize(text, model_finetuned)

        scores_original = scorer.score(summary_label, summary_original)
        scores_finetuned = scorer.score(summary_label, summary_finetuned)

        # 점수 집계
        for key in scores_original:
            total_scores_original[key]["precision"] += scores_original[key].precision
            total_scores_original[key]["recall"] += scores_original[key].recall
            total_scores_original[key]["fmeasure"] += scores_original[key].fmeasure
            total_scores_finetuned[key]["precision"] += scores_finetuned[key].precision
            total_scores_finetuned[key]["recall"] += scores_finetuned[key].recall
            total_scores_finetuned[key]["fmeasure"] += scores_finetuned[key].fmeasure

        num_entries += 1

    # 평균 점수 계산
    average_scores_original = {key: {k: v / num_entries for k, v in total_scores_original[key].items()} for key in total_scores_original}
    average_scores_finetuned = {key: {k: v / num_entries for k, v in total_scores_finetuned[key].items()} for key in total_scores_finetuned}

    return average_scores_original, average_scores_finetuned


# TSV 파일 경로
file_path = 'C:\\Project\\Model_Integration\\test\\tech_test.tsv'

# ROUGE 점수 계산 실행
average_scores_original, average_scores_finetuned = calculate_rouge_scores(file_path)
print("Average ROUGE scores for the original model:", average_scores_original)
print("Average ROUGE scores for the finetuned model:", average_scores_finetuned)