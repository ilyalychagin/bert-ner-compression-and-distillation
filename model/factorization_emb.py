from copy import deepcopy

import torch
import torch.nn as nn
from transformers import BertForTokenClassification

class FactorizationEmbedding(nn.Module):
    def __init__(self, original_matrix=None, vocab_size=16000, hidden_size=728, factor_dim=64, padding_idx=None):
        super().__init__()
        if original_matrix is not None:
            U, E, Vt = torch.linalg.svd(original_matrix.detach(), full_matrices=False)
            A = (U[:, :factor_dim] * E[:factor_dim])
            B = Vt[:factor_dim].clone() 
        else:
            A = torch.randn(vocab_size, factor_dim) * 0.02
            B = torch.randn(factor_dim, hidden_size) / factor_dim**0.5
        
        self.A = nn.Embedding.from_pretrained(
            A,
            freeze=False,
            padding_idx=padding_idx,
        )
        self.B = nn.Parameter(B)
    
    def forward(self, input_ids):
        A_embs = self.A(input_ids)
        return A_embs @ self.B

class FactorizedBertForTokenClassification(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        vocab_size=config.vocab_size
        hidden_size=config.hidden_size
        padding_idx=config.pad_token_id

        new_embeddings = FactorizationEmbedding(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            factor_dim=config.factor_dim,
            padding_idx=padding_idx,
        )

        self.set_input_embeddings(new_embeddings)

    @staticmethod
    def from_teacher(teacher, factor_dim=64):
        config = deepcopy(teacher.config)
        config.factor_dim = factor_dim
        model = FactorizedBertForTokenClassification(config)

        original = teacher.get_input_embeddings()

        new_embeddings = FactorizationEmbedding(
            original_matrix=original.weight,
            factor_dim=factor_dim,
            padding_idx=original.padding_idx,
        )

        weights = teacher.state_dict()
        weights.pop("bert.embeddings.word_embeddings.weight")
        for name, tensor in new_embeddings.state_dict().items():
            weights["bert.embeddings.word_embeddings." + name] = tensor
        
        model.load_state_dict(weights)

        return model
