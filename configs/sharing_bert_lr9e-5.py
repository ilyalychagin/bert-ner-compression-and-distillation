from pathlib import Path

class Config:
    seed: int = 42

    batch_size_train: int = 64
    batch_size_val: int = 64
    batch_size_test: int = 64
    trunc_len: int = 512
    factor_dim: int = 128

    # student
    hidden_size_student: int = 1024
    intermediate_size_student: int = 5120
    num_hidden_layers_student: int = 1
    num_attention_heads_student: int = 16
    num_shared_steps: int = 12

    lr: float = 9e-5
    wd: float = 0.01
    sheduler_factor: float = 0.5
    sheduler_patience: int = 1
    num_epochs: int = 100
    warmup_part: float = 0.1
    distill: bool = False

    checkpoint_dir: Path = Path("checkpoints/shared_bert_lr9e-5")
    checkpoint_teacher_dir: Path = Path("checkpoints/fine-tune_bert_lr5e-5")