import torch.nn as nn
import torch

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
