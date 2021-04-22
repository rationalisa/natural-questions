from transformers import PreTrainedTokenizerFast, AutoTokenizer
from utils_data import simplify_nq_example
from datasets import concatenate_datasets,load_dataset
from absl import app
from absl import flags
from absl import logging
import random
import os

max_length = 512
doc_stride = 128

hyper_tokens = ['<P>','</P>', '<Table>','</Table>','<Li>','</Li>','<Th>','</Th>','<Td>','</Td>','Ul','/Ul']
model_pretrain = "bert-large-cased-whole-word-masking-finetuned-squad"

flags.DEFINE_string('data_dir','/storage/BERT_SQUAD_TOK', 'Path to the diretory to save the training datasets')

flags.DEFINE_string('cache_dir','/storage/datset', 'Cache dir to store temp training set')

flags.DEFINE_integer('pre_size', 0, 'Starting example index in the dataset')

flags.DEFINE_integer('total_size', 300000, 'Ending example index(not included) in the dataset. Max is 307373')

flags.DEFINE_integer('sample_size', 4000, 'Size of processed examples each iteration. For RAM consideration')

flags.DEFINE_bool('TOK', True, 'Wheather to tokenize hyperlink special characters as single tokens')

flags.DEFINE_bool('train', False, 'Wheather to tokenize train dataset or validation dataset')

FLAGS = flags.FLAGS

def main(_):
    logging.info('loading dataset...')
    datasets = load_dataset("natural_questions",cache_dir=FLAGS.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_pretrain,
        use_fast=True,
    )   
    pad_on_right = tokenizer.padding_side == "right"
    if FLAGS.TOK:
        tokenizer.add_tokens(hyper_tokens,special_tokens=True)
        
    if not os.path.isdir(FLAGS.data_dir):
        os.mkdir(FLAGS.data_dir)
    
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

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.       

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

        offset_mapping = tokenized_examples["offset_mapping"]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)           
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

            # One example can give several spans, this is the index of the example containing this span of text.
            start_char = examples["annotations"][sample_index]["long_answer"]["start_char"]
            ## TODO for gz file
            '''
            start_char, end_char = -1, -1
            # Start/end character index of the answer in the text.
            for candidate in examples["annotations"][sample_index]:
                cur = candidate["long_answer"]["start_char"]
                if cur != -1:
                    start_char = cur
                    end_char = candidate["long_answer"]["end_char"]
            '''
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
    
    data_list = []
    example_list = []
    for i in range(FLAGS.pre_size, FLAGS.total_size, FLAGS.sample_size):
        batch = datasets['train'].select(range(i,min(i+FLAGS.sample_size,len(datasets['train'])))) 
        if FLAGS.train:
            tokenized_test = batch.map(prepare_train_features, batched=True, remove_columns=batch.column_names)
            logging.info('Current i: {}, sample_size: {}, padded sentences: {}'.format(i ,FLAGS.sample_size,len(tokenized_test)))
            
            # downsample examples with null answer to the size with answers
            neg = tokenized_test.filter(lambda example: example['start_positions']==0)
            pos = tokenized_test.filter(lambda example: example['start_positions']!=0)

            neg_down = neg.select(random.sample(range(0,len(neg)-1), len(pos)))
            data_list.extend([neg_down, pos])
            logging.info('Current processed size'.format(len(pos)*2))
        else:
            examples = batch.map(simplify_nq_example, batched=True)
            tokenized_test = examples.map(prepare_validation_features, batched=True, remove_columns=examples.column_names)
            example_list.append(examples)
            data_list.append(tokenized_test)
    total = concatenate_datasets(data_list)
    if FLAGS.train:
        total.save_to_disk(os.path.join(FLAGS.data_dir, "train_{}_{}".format(FLAGS.total_size-FLAGS.pre_size, len(total))))
    else:
        examples = concatenate_datasets(example_list)
        total.save_to_disk(os.path.join(FLAGS.data_dir, "valid_{}".format(FLAGS.total_size-FLAGS.pre_size)))
        examples.save_to_disk(os.path.join(FLAGS.data_dir, "example_valid_{}".format(FLAGS.total_size-FLAGS.pre_size)))
            

if __name__ == "__main__":
    app.run(main)