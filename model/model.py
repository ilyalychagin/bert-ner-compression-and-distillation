from transformers import BertForTokenClassification
from .factorization_emb import FactorizationEmbedding
from copy import deepcopy

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


