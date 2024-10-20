from pathlib import Path

import datasets
import pandas as pd
import re

# Функция для очистки текста
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    contractions = {
        "doesn’t": "does not",
        "we’re": "we are"
    }
    for contraction, full_form in contractions.items():
        text = text.replace(contraction, full_form)

    text = re.sub(r'(.)\1+', r'\1', text) 
    text = re.sub(r'[&^#*]', '', text)

    return text

def prepare(raw_data: Path) -> datasets.Dataset:
    dataset = pd.read_excel(raw_data)
    dataset.dropna(inplace=True) 
    dataset.drop_duplicates(inplace=True) 
    dataset['cleaned_text'] = dataset['message'].apply(clean_text)
    return datasets.Dataset.from_dict(dataset)

def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))

def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
