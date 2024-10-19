import argparse
from pathlib import Path

from toxic_clf.data import load_dataset, prepare, save_dataset
from toxic_clf.models import classifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    default_data_path = Path('./prepared-dataset')

    prepare_data_parser = subparsers.add_parser('prepare-data')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        'input',
        help='Path to load raw dataset',
        type=Path,
    )
    prepare_data_parser.add_argument(
        '-o',
        '--output',
        help='Path to save prepared dataset to',
        type=Path,
        default=default_data_path,
    )

    vectorize_data_parser = subparsers.add_parser('vectorize-data')
    vectorize_data_parser.set_defaults(func=vectorize_dataset)
    vectorize_data_parser.add_argument(
        '-p',
        '--prepared',
        help='Path to prepared dataset',
        type=Path,
        default=default_data_path,
    )
    vectorize_data_parser.add_argument(
        '-v',
        '--vectorizer',
        help='Vectorizer',
        choices=['count_vec', 'tfidf'],
        default='count_vec'
    )
    default_data_path = Path('./vectorized-dataset')
    vectorize_data_parser.add_argument(
        '-ov',
        '--outputvectorized',
        help='Path to save vectorized dataset to',
        type=Path,
        default=default_data_path,
    )
    
    predict_parser = subparsers.add_parser('classify')
    predict_parser.set_defaults(func=classify)
    predict_parser.add_argument(
        '-d',
        '--dataset',
        help='Path to vectorized dataset',
        type=Path,
        default=default_data_path,
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        choices=['rand_for', 'log_reg', 'microsoft/codebert-base'],
        default='rand_for',
    )

    return parser.parse_args()


def prepare_data(args):
    dataset = prepare(args.input)
    save_dataset(dataset, args.output)

def vectorize_dataset(args):
    dataset = load_dataset(args.prepared)
    if (args.vectorizer == 'count_vec'):
        vect = CountVectorizer()
    else:
        vect = TfidfVectorizer()
        
    def vectorize_text(data):
        vectors = vect.fit_transform(data['cleaned_text'])
        return {'vectorized': vectors.tolist()}

    dataset = dataset.map(vectorize_text, batched=True)
    print(dataset)
    print(dataset['features'])
    print(dataset['features']['vectorized'])
    save_dataset(dataset, args.outputvectorized)

def classify(args):
    dataset = load_dataset(args.dataset)
    classifier(dataset, args.model)


if __name__ == '__main__':
    main()
