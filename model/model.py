import torch
from torch import nn
from transformers import BertForTokenClassification
from .factorization_emb import FactorizationEmbedding
from copy import deepcopy
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

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

class SharedBertEncoder(BertEncoder):

    def __init__(self, config):
        super().__init__(config)

        if getattr(config, "lora_adaptation", False):
            self.adapters = nn.ModuleList()
            rank = config.lora_rank
            for depth in range(config.num_shared_steps):
                block = self.layer[depth % len(self.layer)]
                shapes = {
                    "query": block.attention.self.query.weight.shape,
                    "key": block.attention.self.key.weight.shape,
                    "value": block.attention.self.value.weight.shape,
                    "attn_output": block.attention.output.dense.weight.shape,
                    "intermediate": block.intermediate.dense.weight.shape,
                    "ffn_output": block.output.dense.weight.shape,
                }
                adapters = nn.ModuleDict()
                for name, (out_features, in_features) in shapes.items():
                    adapters[name] = nn.ParameterDict({
                        "A": nn.Parameter(torch.randn(rank, in_features) * 0.02),
                        "B": nn.Parameter(torch.zeros(out_features, rank)),
                    })

                self.adapters.append(adapters)


    def forward(self, hidden_states, attention_mask=None, **kwargs):
        weight_paths = {
            "query": "attention.self.query.weight",
            "key": "attention.self.key.weight",
            "value": "attention.self.value.weight",
            "attn_output": "attention.output.dense.weight",
            "intermediate": "intermediate.dense.weight",
            "ffn_output": "output.dense.weight",
        }
        use_lora = getattr(self.config, "lora_adaptation", False)

        for depth in range(self.config.num_shared_steps):
            block = self.layer[depth % len(self.layer)]
            if not use_lora:
                hidden_states = block(hidden_states, attention_mask=attention_mask, **kwargs)
                continue

            adapters = self.adapters[depth]
            lora_weights = {
                path: block.get_parameter(path) + adapters[name]["B"] @ adapters[name]["A"]
                for name, path in weight_paths.items()
            }
            hidden_states = torch.func.functional_call(
                block,
                lora_weights,
                args=(hidden_states,),
                kwargs={"attention_mask": attention_mask, **kwargs},
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
        )

    @staticmethod
    def from_teacher(teacher, config):
        shared_bert_encoder = SharedBertEncoder(config)
        teacher_bert_encoder = teacher.bert.encoder

        shared_layers = shared_bert_encoder.layer
        teacher_layers = teacher_bert_encoder.layer

        num_shared_blocks = len(shared_layers)

        with torch.no_grad():
            for i, shared_layer in enumerate(shared_layers):
                group = teacher_layers[i::num_shared_blocks]

                for name, param in shared_layer.named_parameters():
                    teacher_params = [layer.get_parameter(name) for layer in group]
                    param.copy_(torch.stack(teacher_params).mean(dim=0))
            
            use_lora = getattr(config, "lora_adaptation", False)

            if not use_lora:
                return shared_bert_encoder

            weight_paths = {
                "query": "attention.self.query.weight",
                "key": "attention.self.key.weight",
                "value": "attention.self.value.weight",
                "attn_output": "attention.output.dense.weight",
                "intermediate": "intermediate.dense.weight",
                "ffn_output": "output.dense.weight",
            }
            rank = config.lora_rank
            for i, adapters in enumerate(shared_bert_encoder.adapters):
                shared_layer = shared_layers[i % num_shared_blocks]
                teacher_layer = teacher_layers[i]

                for name_adapter, weight_path in weight_paths.items():
                    diff_tensor = teacher_layer.get_parameter(weight_path) - shared_layer.get_parameter(weight_path)
                    U, E, Vt = torch.linalg.svd(diff_tensor.detach(), full_matrices=False)

                    adapter = adapters[name_adapter]
                    adapter["A"].copy_(Vt[:rank, :])
                    adapter["B"].copy_(U[:, :rank] * E[:rank])
        
        return shared_bert_encoder




        


