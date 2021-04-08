from src import transformers
from transformers import PreTrainedTokenizerFast
from datasets import concatenate_datasets
from datasets import load_from_disk
from transformers import AutoTokenizer
from utils_qa import postprocess_qa_predictions
from trainer_qa import QuestionAnsweringTrainer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator


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

    # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
    # corresponding example_id and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
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

    return tokenized_examples

# Post-processing:
def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=data_args.version_2_with_negative,
        n_best_size=data_args.n_best_size,
        max_answer_length=data_args.max_answer_length,
        null_score_diff_threshold=data_args.null_score_diff_threshold,
        output_dir=training_args.output_dir,
        is_world_process_zero=trainer.is_world_process_zero(),
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if data_args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)



max_length = 512
doc_stride = 128
batch_size = 1
max_val_samples= 100
dir_name = 'roberta'
model_pretrain = 'roberta-large'   # "bert-large-uncased" 
tokenizer = AutoTokenizer.from_pretrained(
    model_pretrain,
    use_fast=True,
)
pad_on_right = tokenizer.padding_side == "right"
train_dataset=load_from_disk("/storage/{}/train_{}_{}".format(dir_name, 20000,15950)).shuffle()
eval_examples = load_from_disk("/storage/{}/val_example".format(dir_name))
eval_examples = eval_examples.select(range(max_val_samples))
eval_dataset = eval_examples.map(prepare_validation_features, batched=True, remove_columns=eval_examples.column_names)
#eval_dataset = load_from_disk("/storage/{}/val".format(dir_name))


#weight_decay=0.01
model = AutoModelForQuestionAnswering.from_pretrained(model_pretrain)
args = TrainingArguments(
    "/storage/model/{}_3".format(dir_name),
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    save_steps = 600,
    eval_steps = 600,
    evaluation_strategy ='steps',
    gradient_accumulation_steps=8,
)

data_collator = default_data_collator

# Initialize our Trainer
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

trainer.train()