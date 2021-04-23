data_dir="/storage/BERT_SQUAD_TOK" # dir to save the preprocessed json file
save_dir="/storage/model/BERT_SQUAD_TOK_L3" # dir to save the models
val_path="/storage/datset/v1.0-simplified_nq-dev-all.jsonl.gz" # path to the validation dataset downloaded
cache_dir="/storage/datset" # dir to save the cached training data to disk
train_name="train_220000_177878" #name of training set under data_dir

# Prepare training examples
python3 preprocess.py --pre_size 0 --total_size 220000 --train --data_dir ${data_dir} --cache_dir ${cache_dir}

# Prepare validation examples
python3 preprocess.py --pre_size 300000 --total_size 300050 --sample_size 50 --data_dir ${data_dir} --cache_dir ${cache_dir}

# Train the model
python3 train_squad.py --cp checkpoint-1200 --save_dir ${save_dir} --data_dir ${data_dir} --train_name ${train_name}

# Evaluate using dev-all dataset
python3 evaluate.py --cp checkpoint-1200 --save_dir ${save_dir} --diff_threshold -5 --val_path ${val_path}

# Compute the F1 score
pred_path="$save_dir/eval_predictions.json"
## ensemble ## 
# pred_path = "$save_dir/eval_predictions1.json@$save_dir/eval_predictions2.json"
python3 nq/nq_eval.py --gold_path=${val_path} --predictions_path=${pred_path} --pretty_print