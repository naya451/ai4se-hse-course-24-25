from transformers import T5ForConditionalGeneration, AutoTokenizer
import evaluate
import pandas as pd
import torch

if torch.backends.mps.is_available():
   mps_device = torch.device("mps")
   x = torch.ones(1, device=mps_device)
   print (x)
else:
   print ("MPS device not found.")  

data = pd.read_excel('./prepared.xlsx')

checkpoint = "Salesforce/codet5p-220m" 
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m" )
model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(mps_device)

def predict_function_name(function_body):
    inputs = tokenizer.encode(function_body, return_tensors="pt").to(mps_device)
    outputs = model.generate(inputs, max_length=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

predictions = [predict_function_name(i).split(' ') for i in data['my_bwc']]
predictions = [i[1] if len(i) > 1 else i[0] for i in predictions]

data['pred_with_comments'] = predictions
exact_match = evaluate.load('exact_match')
rouge = evaluate.load('rouge')

references = data['my_func_name']

em_score = exact_match.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

em_score = em_score["exact_match"]
rouge_score = rouge_score["rouge1"]
print("Metrics for functions with comments:")
print(f'Exact Match: {em_score}')
print(f'ROUGE-1: {rouge_score}')


predictions = [predict_function_name(i).split(' ') for i in data['my_bnc']]
predictions = [i[1] if len(i) > 1 else i[0] for i in predictions]

data['pred_without_comments'] = predictions
data.to_excel('./done.xlsx')
em_score = exact_match.compute(predictions=predictions, references=references)
rouge_score = rouge.compute(predictions=predictions, references=references)

em_score = em_score["exact_match"]
rouge_score = rouge_score["rouge1"]
print("Metrics for functions without comments:")
print(f'Exact Match: {em_score}')
print(f'ROUGE-1: {rouge_score}')