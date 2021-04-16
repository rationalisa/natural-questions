from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from utils_data import simplify_nq_example
import random
import os
from datasets import concatenate_datasets

# Training preprocessing
def prepare_train_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    examples = simplify_nq_example(examples)

    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        start_char = examples["annotations"][sample_index]["long_answer"]["start_char"]
        # If no answers are given, set the cls_index as answer.
        if start_char == -1:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            end_char = examples["annotations"][sample_index]["long_answer"]["end_char"]

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


print('loading dataset...')
datasets = load_dataset("natural_questions",cache_dir='/storage/datset')

model_pretrain = "bert-large-cased-whole-word-masking-finetuned-squad"#'roberta-large'   #"bert-large-uncased" 
#tokenizer =  PreTrainedTokenizerFast.from_pretrained(model_pretrain)
tokenizer = AutoTokenizer.from_pretrained(
    model_pretrain,
    use_fast=True,
)


max_length = 512
doc_stride = 128
batch_size = 1
pad_on_right = tokenizer.padding_side == "right"

sample_size = 4000
pre_size = 220000
total_size = 300000
save_dir = os.path.join('/storage','BERT_SQUAD_TOK')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
data_list = []
for i in range(pre_size, total_size,sample_size):
    test_train = datasets['train'].select(range(i,min(i+sample_size,len(datasets['train'])))) #307373
    tokenized_test = test_train.map(prepare_train_features, batched=True, remove_columns=test_train.column_names)
    print('i: ',i ,'sample_size', sample_size, 'padded sentences',len(tokenized_test))

    neg = tokenized_test.filter(lambda example: example['start_positions']==0)
    pos = tokenized_test.filter(lambda example: example['start_positions']!=0)
    print('neg: ', len(neg),'pos: ', len(pos))

    neg_down = neg.select(random.sample(range(0,len(neg)-1), len(pos)))
    data_list.extend([neg_down, pos])
total_train = concatenate_datasets(data_list)
total_train.save_to_disk(os.path.join(save_dir, "train_{}_{}".format(total_size-pre_size, len(total_train))))
