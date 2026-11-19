from datasets import load_dataset
from tokenization import tokenize_dataset

def get_dataset():
    dataset = load_dataset("ashaba1in/conll2003")
    dataset = dataset.remove_columns(["id", "pos_tags", "chunk_tags"])

    return dataset

def get_tokenized_dataset(tokenizer):
    dataset = get_dataset()
    return tokenize_dataset(dataset, tokenizer)
    
