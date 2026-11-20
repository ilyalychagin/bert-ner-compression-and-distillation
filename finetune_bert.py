import argparse
from functools import partial
from pathlib import Path
from runpy import run_path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification

from dataset.collate import collate
from dataset.dataset import get_tokenized_dataset
from dataset.tokenization import get_tokenizer
from scripts.train import train, validation
from utils.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = run_path(str(config_path))["Config"]()

    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    set_seed(cfg.seed)

    model = AutoModelForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_names))

    print('Число параметров:', sum(p.numel() for p in model.parameters()))

    tokenizer = get_tokenizer()
    dataset = get_tokenized_dataset(tokenizer)

    train_loader = DataLoader(
        dataset['train'],
        batch_size=cfg.batch_size_train,
        shuffle=True,
        collate_fn=partial(
            collate, 
            trunc_len=cfg.trunc_len, 
            padding_value=tokenizer.pad_token_id,
            include_labels=True
        ) 
    )

    val_loader = DataLoader(
        dataset['validation'],
        batch_size=cfg.batch_size_val,
        shuffle=False,
        collate_fn=partial(
            collate,
            trunc_len=cfg.trunc_len,
            padding_value=tokenizer.pad_token_id,
            include_labels=True,
        ),
    )

    test_loader = DataLoader(
        dataset['test'],
        batch_size=cfg.batch_size_test,
        shuffle=False,
        collate_fn=partial(
            collate, 
            trunc_len=cfg.trunc_len, 
            padding_value=tokenizer.pad_token_id,
            include_labels=True
        ) 
    )


    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.sheduler_factor,
        patience=cfg.sheduler_patience,
    )

    train_losses, val_losses, val_f1_scores = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.num_epochs,
        device=device,
        checkpoint_dir=cfg.checkpoint_dir,
        val_every_n_steps=len(train_loader),
    )

    best_model = AutoModelForTokenClassification.from_pretrained(cfg.checkpoint_dir).to(device)

    val_loss, val_f1 = validation(best_model, val_loader, criterion, device)
    test_loss, test_f1 = validation(best_model, test_loader, criterion, device)
    print('Results for best checkpoint:')
    print(f"Val loss: {val_loss}, Val F1: {val_f1}")
    print(f"Test loss: {test_loss}, Test F1: {test_f1}")

if __name__ == "__main__":
    main()