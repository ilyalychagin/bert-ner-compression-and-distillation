import torch
from torch.nn.utils.rnn import pad_sequence


def collate(batch, trunc_len=None, padding_value=0, include_labels=True):
    if trunc_len is not None:
        inputs = [input['input_ids'][:min(trunc_len-1, len(input['input_ids'])-1)] + [input['input_ids'][-1]] for input in batch]
        attention_masks = [input['attention_mask'][:min(trunc_len-1, len(input['attention_mask'])-1)] + [1] for input in batch]
        if include_labels:
            labels = [input['labels'][:min(trunc_len-1, len(input['labels'])-1)] + [-100] for input in batch]
        else:
            labels = None
    else:
        inputs = [input['input_ids'] for input in batch] 
        attention_masks = [input['attention_mask'] for input in batch]
        if include_labels:
            labels = [input['labels'] for input in batch]
        else:
            labels = None

    
    def pad(batch, padding_value):
        return pad_sequence(
            [torch.tensor(sample) for sample in batch], 
            batch_first=True,
            padding_value=padding_value
        )

    inputs = pad(inputs, padding_value)
    attention_masks = pad(attention_masks, 0)
    if include_labels:
        labels = pad(labels, -100)

    collate_output = dict({
        'input_ids' : inputs,
        'attention_mask' : attention_masks
    })

    if include_labels:
        collate_output['labels'] = labels

    return collate_output