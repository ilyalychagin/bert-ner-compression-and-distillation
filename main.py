from dataset import get_dataset, get_tokenized_dataset
from tokenization import get_tokenizer
from collate import collate
from seqeval.metrics import f1_score
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
import torch
from train import train, validation

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
        include_labels=True
    ) 
)


device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

model.to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
)

train_losses, val_losses, val_f1_scores = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=3,
    device=device,
    val_every_n_steps=len(train_loader),
)

checkpoint_dir = "checkpoints/fine-tune_bert"
model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)

test_loss, test_f1 = validation(model, test_loader, criterion, device)
print(f"Test loss: {test_loss}, Test F1: {test_f1}")