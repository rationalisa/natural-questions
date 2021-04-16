import transformers
from transformers.utils import check_min_version
from transformers import PreTrainedTokenizerFast,AutoConfig, AutoTokenizer
from datasets import concatenate_datasets, load_from_disk, Dataset
from utils_qa import postprocess_qa_predictions
from trainer_qa import QuestionAnsweringTrainer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, EvalPrediction
from transformers import default_data_collator
from utils_data import read_annotation_gzip
import os
from absl import flags
from absl import logging
check_min_version("4.5.0.dev0")


max_length = 512
doc_stride = 128
max_val_samples= 50

val_path = '/storage/datset/v1.0_sample_nq-dev-sample.jsonl.gz'
model_pretrain = "bert-large-cased-whole-word-masking-finetuned-squad" 
root_path = "/storage/model"
hyper_tokens = ['<P>','</P>', '<Table>','</Table>','<Li>','</Li>','<Th>','</Th>','<Td>','</Td>','Ul','/Ul']

flags.DEFINE_integer(
    'batch_size', 4, 'Batch size during training')
flags.DEFINE_string('dir_name','BERT_SQUAD_TOK_L_REL', 'Path to the diretory to save the models')

flags.DEFINE_string('cp','checkpoint-0', 'name of the pretrained checkpoint under dir_name')

flags.DEFINE_integer('steps', 200, 'Frequency to evaluate and save model during training')
flags.DEFINE_integer('epochs', 1, 'Finetune epochs')
flags.DEFINE_bool('pretty_print', False, 'Whether to pretty print output.')

FLAGS = flags.FLAGS

# Validation preprocessing
def prepare_validation_features(examples):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    #examples = simplify_nq_example(examples)
    
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

        ## TODO for gz file
        start_char, end_char = -1, -1
        # Start/end character index of the answer in the text.
        for candidate in examples["annotations"][sample_index]:
            cur = candidate["long_answer"]["start_char"]
            if cur != -1:
                start_char = cur
                end_char = candidate["long_answer"]["end_char"]
        # If no answers are given, set the cls_index as answer.
        if start_char == -1:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:

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


# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        n_best_size=20,
        output_dir= args.output_dir,
        is_world_process_zero=trainer.is_world_process_zero(),
        prefix=stage,
        version_2_with_negative=True,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex['id'], 'start_token': ex['annotations'][0]['long_answer']['start_token'], 
                   'end_token': ex['annotations'][0]['long_answer']['end_token']
      } for ex in examples ]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)



save_dir = os.path.join(root_path, FLAGS.dir_name)
checkpoint = os.path.join(save_dir,FLAGS.cp)


def main(_):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # Build model
    if 'REL' in FLAGS.dir_name:
        if os.path.isdir(checkpoint):
            print('Loading from {}'.format(checkpoint))
            config = AutoConfig.from_pretrained(checkpoint)
        else:
            config = AutoConfig.from_pretrained(model_pretrain)
        print(config.position_embedding_type)
        config.position_embedding_type="relative_key_query"
        model = AutoModelForQuestionAnswering.from_config(config)
    else:
        if os.path.isdir(checkpoint):
            print('Loading from {}'.format(checkpoint))
            model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(model_pretrain)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_pretrain,
        use_fast=True,
    )
    if 'TOK' in FLAGS.dir_name:
        tokenizer.add_tokens(hyper_tokens,special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
    
    # dataset
    pad_on_right = tokenizer.padding_side == "right"
    
    #train_dataset_22 =load_from_disk("/storage/BERT_SQUAD_TOK/train_{}_{}".format(220000, 177878)).shuffle()
    train_dataset =load_from_disk("/storage/BERT_SQUAD_TOK/train_{}_{}".format(80000, 63244)).shuffle()
    #train_dataset=load_from_disk("/storage/BERT_SQUAD_TOK/train_{}_{}".format(100000, 82522)).shuffle()
    #train_dataset = concatenate_datasets([train_dataset_22, train_dataset_8])
    
    val_dic = read_annotation_gzip(val_path)
    eval_examples = Dataset.from_dict(val_dic).select(range(max_val_samples))
    eval_dataset = eval_examples.map(prepare_validation_features, batched=True, remove_columns=eval_examples.column_names)

    args = TrainingArguments(
        save_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=FLAGS.batch_size,
        per_device_eval_batch_size=FLAGS.batch_size*16,
        num_train_epochs=FLAGS.epoch,
        save_steps = FLAGS.steps,
        eval_steps = FLAGS.steps,
        logging_steps = FLAGS.steps,
        evaluation_strategy ='steps',
        gradient_accumulation_steps=8,
    )

    data_collator = default_data_collator

    # Initialize Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
    )

    if os.path.isdir(checkpoint):
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    trainer.save_model(os.path.join(save_dir,'last'))


if __name__ == "__main__":
    app.run(main)