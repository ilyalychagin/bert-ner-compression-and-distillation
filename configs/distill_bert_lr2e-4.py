from pathlib import Path

class Config:
    seed: int = 42

    batch_size_train: int = 64
    batch_size_val: int = 64
    batch_size_test: int = 64
    trunc_len: int = 512
    factor_dim: int = 64

    # student
    hidden_size_student: int = 512
    intermediate_size_student: int = 2048
    num_hidden_layers_student: int = 5
    num_attention_heads_student: int = 8

    lr: float = 2e-4
    wd: float = 0.01
    sheduler_factor: float = 0.5
    sheduler_patience: int = 1
    num_epochs: int = 100
    warmup_part: float = 0.1

    checkpoint_dir: Path = Path("checkpoints/distilled_bert_lr2e-4")
    checkpoint_teacher_dir: Path = Path("checkpoints/fine-tune_bert_lr5e-5")



# bert.encoder.layer.6.attention.self.query.weight 589824
# bert.encoder.layer.6.attention.self.query.bias 768
# bert.encoder.layer.6.attention.self.key.weight 589824
# bert.encoder.layer.6.attention.self.key.bias 768
# bert.encoder.layer.6.attention.self.value.weight 589824
# bert.encoder.layer.6.attention.self.value.bias 768
# bert.encoder.layer.6.attention.output.dense.weight 589824
# bert.encoder.layer.6.attention.output.dense.bias 768
# bert.encoder.layer.6.attention.output.LayerNorm.weight 768
# bert.encoder.layer.6.attention.output.LayerNorm.bias 768
# bert.encoder.layer.6.intermediate.dense.weight 2359296
# bert.encoder.layer.6.intermediate.dense.bias 3072
# bert.encoder.layer.6.output.dense.weight 2359296
# bert.encoder.layer.6.output.dense.bias 768
# bert.encoder.layer.6.output.LayerNorm.weight 768
# bert.encoder.layer.6.output.LayerNorm.bias 768

# classifier.weight 6912

# Слой 1: 7,087,872