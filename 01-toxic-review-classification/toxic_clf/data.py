from pathlib import Path

import datasets
import pandas as pd


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
    data['cleaned_text'] = data['message'].apply(clean_text)
    return datasets.Dataset.from_dict({'fake_data': [[1, 2], [3, 4]]})


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
