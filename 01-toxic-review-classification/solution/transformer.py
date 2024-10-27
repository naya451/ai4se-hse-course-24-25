
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
import numpy as np
import pandas as pd
from datasets import load_metric
# !pip3 install transformers torch
if torch.backends.mps.is_available():
   mps_device = torch.device("mps")
   x = torch.ones(1, device=mps_device)
   print (x)
else:
   print ("MPS device not found.")  

tr = pd.read_excel("./training.xlsx")

tr.dropna(inplace=True) 


X = tr['X']
y = tr['y']

ev = pd.read_excel("./eval.xlsx")

ev.dropna(inplace=True) 


X_eval = ev['X']
y_eval = ev['y']

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base').to(mps_device)


train_encodings = tokenizer(list(X), return_tensors='pt', padding=True, truncation=True)
train_texts = list(X)  
train_labels = list(y)  
eval_encodings = tokenizer(list(X_eval), return_tensors='pt', padding=True, truncation=True)
eval_labels = list(y_eval)
# SECRET 6e238a1a8cb43d0a05ba97a9453b51165ae3b03c
# Create a dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, train_labels)

eval_dataset = CustomDataset(eval_encodings, eval_labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,

)

def compute_metrics(eval_pred):
    accuracy_metric = load_metric("accuracy")
    precision_recall_fscore_metric = load_metric("precision_recall_fscore")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    precision_recall_fscore = precision_recall_fscore_metric.compute(predictions=predictions, references=labels, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision_recall_fscore['precision'],
        'recall': precision_recall_fscore['recall'],
        'f1': precision_recall_fscore['f1']
    }

trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)
