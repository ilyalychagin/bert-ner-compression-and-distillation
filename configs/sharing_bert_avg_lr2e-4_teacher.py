from pathlib import Path

class Config:
    seed: int = 42

    batch_size_train: int = 64
    batch_size_val: int = 64
    batch_size_test: int = 64
    trunc_len: int = 512
    factor_dim: int = 64

    hidden_size_student: int = 768
    intermediate_size_student: int = 3072
    num_hidden_layers_student: int = 2
    num_attention_heads_student: int = 12
    num_shared_steps: int = 12
    init_from_teacher: bool = True
    lora_adaptation: bool = False
    lora_rank: int = 8

    lr: float = 2e-4
    wd: float = 0.01
    sheduler_factor: float = 0.5
    sheduler_patience: int = 1
    num_epochs: int = 100
    warmup_part: float = 0.1
    distill: bool = True

    checkpoint_dir: Path = Path("checkpoints/shared_bert_avg_lr2e-4_with_teacher")
    checkpoint_teacher_dir: Path = Path("checkpoints/fine-tune_bert_lr5e-5")
