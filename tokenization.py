from transformers import AutoTokenizer

# label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return tokenizer

def align_labels_with_tokens(ner_tags, ids):
    aligned_tags = [-100]
    pos = 0
    for _id in range(1, len(ids) - 1):
        if _id == 1:
            aligned_tags.append(ner_tags[pos])
            continue
    
        if ids[_id] == ids[_id-1]:
            if ner_tags[pos] == 0:
                aligned_tags.append(ner_tags[pos])
            else:
                if (ner_tags[pos] % 2 == 1):
                    aligned_tags.append(ner_tags[pos] + 1)
                else:
                    aligned_tags.append(ner_tags[pos])
        else:
            pos += 1
            aligned_tags.append(ner_tags[pos])
    aligned_tags.append(-100)
    
    return aligned_tags
            
        

def tokenize_dataset(dataset, tokenizer):
    def tokenize_batch(batch):
        words = batch['tokens']
        labels = batch['ner_tags']

        tokenized_text = tokenizer(words, is_split_into_words=True)

        aligned_labels = []

        for i, ner_tags in enumerate(labels):
            cur_aligned_labels = align_labels_with_tokens(ner_tags, tokenized_text.word_ids(batch_index=i))
            aligned_labels.append(cur_aligned_labels)

        return {'input_ids': tokenized_text["input_ids"],
                'attention_mask': tokenized_text["attention_mask"],
                'labels': aligned_labels}
        

    return dataset.map(tokenize_batch, batched=True, batch_size=128)