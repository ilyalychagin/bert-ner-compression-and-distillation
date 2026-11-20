from pathlib import Path

class Config:
    seed: int = 42

    batch_size_train: int = 64
    batch_size_val: int = 64
    batch_size_test: int = 64
    trunc_len: int = 512
    factor_dim: int = 64

    lr: float = 2e-5
    wd: float = 0.01
    sheduler_factor: float = 0.5
    sheduler_patience: int = 1
    num_epochs: int = 10

    checkpoint_dir: Path = Path("checkpoints/fine-tune_factorized_bert")
    checkpoint_teacher_dir: Path = Path("checkpoints/fine-tune_bert_lr5e-5")
