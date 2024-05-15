import torch
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration

# 사전 학습된 모델(Fine-tuning x) <- huggingface에서 불러옴.
model_original = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
# 사전 학습된 모델(Fine-tuning o) <- kobart_summary 디렉토리에서 불러옴.
model_finetuned = BartForConditionalGeneration.from_pretrained('./models/text_summarization')

tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')

text = input()

print("========== Original Text ==========")
print(text)
print('\n')
print("========== input_ids ==========")

if text:
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    output = model_finetuned.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)    
    print("========== Summary ==========")
    print(output)