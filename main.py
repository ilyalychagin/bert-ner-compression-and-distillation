from dataset import get_dataset, get_tokenized_dataset
from tokenization import get_tokenizer
from collate import collate
from seqeval.metrics import f1_score
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
import torch

label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_names))

print('Число параметров:', sum(p.numel() for p in model.parameters()))

tokenizer = get_tokenizer()
dataset = get_tokenized_dataset(tokenizer)

train_loader = DataLoader(
    dataset['train'],
    batch_size=64,
    shuffle=True,
    collate_fn=partial(
        collate, 
        trunc_len=512, 
        padding_value=tokenizer.pad_token_id,
        include_labels=True
    ) 
)

val_loader = DataLoader(
    dataset['validation'],
    batch_size=64,
    shuffle=False,
    collate_fn=partial(
        collate,
        trunc_len=512,
        padding_value=tokenizer.pad_token_id,
        include_labels=True,
    ),
)

test_loader = DataLoader(
    dataset['test'],
    batch_size=64,
    shuffle=False,
    collate_fn=partial(
        collate, 
        trunc_len=512, 
        padding_value=tokenizer.pad_token_id,
        include_labels=False
    ) 
)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

for batch in train_loader:
    print(batch)
    break