from datasets import concatenate_datasets, Dataset
from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer
from utils_qa import postprocess_qa_predictions
from trainer_qa import QuestionAnsweringTrainer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer,EvalPrediction
from transformers import default_data_collator
from utils_data import read_annotation_gzip, postprocess_qa_predictions
from absl import app
from absl import flags
from absl import logging
import os


max_length = 512
doc_stride = 128


#val_path = '/storage/datset/v1.0_sample_nq-dev-sample.jsonl.gz'
val_path = '/storage/datset/v1.0-simplified_nq-dev-all.jsonl.gz'
model_pretrain = "bert-large-cased-whole-word-masking-finetuned-squad" 
root_path = "/storage/model"
hyper_tokens = ['<P>','</P>', '<Table>','</Table>','<Li>','</Li>','<Th>','</Th>','<Td>','</Td>','Ul','/Ul']

flags.DEFINE_integer('batch_size', 64, 'Batch size during evaluation')

flags.DEFINE_string('dir_name','BERT_SQUAD_TOK_L3', 'Path to the diretory to save the models')

flags.DEFINE_string('cp','checkpoint-0', 'Name of the pretrained checkpoint under dir_name')

flags.DEFINE_integer('max_answer_length', 500, 'The maximum length of an answer that can be generated. This is\
                    needed because the start and end predictions are not conditioned on one another')

flags.DEFINE_integer('n_best_size', 20, 'The total number of n-best predictions to generate when looking for an answer')

flags.DEFINE_bool('TOK', True, 'Wheather to tokenize hyperlink special characters as single tokens')

flags.DEFINE_integer('max_val_samples', -1, 'max number of validation samples from validation set. For nq-dev-samples\
                    size should be no more than 200, -1 to use all the data')

FLAGS = flags.FLAGS




def main(_):    
    tokenizer = AutoTokenizer.from_pretrained(
        model_pretrain,
        use_fast=True,
    )
    if FLAGS.TOK:
        tokenizer.add_tokens(hyper_tokens,special_tokens=True)
    pad_on_right = tokenizer.padding_side == "right"

    save_dir = os.path.join(root_path, FLAGS.dir_name)
    checkpoint = os.path.join(save_dir,FLAGS.cp)
    assert os.path.exists(checkpoint)
    
    model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)

    
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
    
    dic = read_annotation_gzip(val_path)
    eval_examples = Dataset.from_dict(dic)
    if FLAGS.max_val_samples != -1:
        eval_examples = eval_examples.select(range(FLAGS.max_val_samples))
    print(len(eval_examples))
    eval_dataset = eval_examples.map(prepare_validation_features, batched=True, remove_columns=eval_examples.column_names)

    
    data_collator = default_data_collator

    args = TrainingArguments(
        save_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=FLAGS.batch_size,
        per_device_eval_batch_size=FLAGS.batch_size,
        num_train_epochs=1,
        save_steps = 600,
        eval_steps = 600,
        evaluation_strategy ='steps',
        gradient_accumulation_steps=8,
    )
    
    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best_size=FLAGS.n_best_size,
            max_answer_length=FLAGS.max_answer_length,
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
    
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
    )

    metrics = trainer.evaluate()
    logging.info(metrics)
    
if __name__ == "__main__":
    app.run(main)   
