import argparse
from functools import partial
from pathlib import Path
from runpy import run_path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification
from model.model import FactorizedBertForTokenClassification, SharedBertEncoder

from dataset.collate import collate
from dataset.dataset import get_tokenized_dataset
from dataset.tokenization import get_tokenizer
from scripts.train import train, validation
from utils.utils import set_seed
from transformers import BertConfig
from safetensors.torch import load_file
from transformers import get_cosine_schedule_with_warmup



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = run_path(str(config_path))["Config"]()

    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    set_seed(cfg.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    tokenizer = get_tokenizer()
    dataset = get_tokenized_dataset(tokenizer)

    shared_model_config = BertConfig(
        num_hidden_layers=1,
        hidden_size=cfg.hidden_size_student,
        intermediate_size=cfg.intermediate_size_student,
        num_attention_heads=cfg.num_attention_heads_student,
        vocab_size=tokenizer.vocab_size,
        num_labels=len(label_names),
        pad_token_id=tokenizer.pad_token_id,
    )
    shared_model_config.factor_dim = cfg.factor_dim
    shared_model_config.num_shared_steps = cfg.num_shared_steps
    shared_model = FactorizedBertForTokenClassification(shared_model_config)
    shared_model.bert.encoder = SharedBertEncoder(shared_model_config)
    shared_model.to(device)

    if cfg.distill:
        model_teacher = AutoModelForTokenClassification.from_pretrained(cfg.checkpoint_teacher_dir)
        model_teacher.to(device).eval()
    else:
        model_teacher = None

    print('Число параметров:', sum(p.numel() for p in shared_model.parameters()))

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


    total_steps = cfg.num_epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_part * total_steps)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(shared_model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_losses, val_losses, val_f1_scores = train(
        model=shared_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=cfg.num_epochs,
        device=device,
        checkpoint_dir=cfg.checkpoint_dir,
        val_every_n_steps=len(train_loader),
        teacher=model_teacher,
        T=2,
        scheduler_per_step=True,
    )

    weights = load_file(str(cfg.checkpoint_dir / "model.safetensors"))
    shared_model.load_state_dict(weights, strict=True)

    val_loss, val_f1 = validation(shared_model, val_loader, criterion, device)
    test_loss, test_f1 = validation(shared_model, test_loader, criterion, device)
    print('Results for best checkpoint:')
    print(f"Val loss: {val_loss}, Val F1: {val_f1}")
    print(f"Test loss: {test_loss}, Test F1: {test_f1}")

if __name__ == "__main__":
    main()