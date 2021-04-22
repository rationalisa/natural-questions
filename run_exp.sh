data_dir = "/storage/BERT_SQUAD_TOK" # dir to save the preprocessed json file
save_dir = "/storage/model/BERT_SQUAD_TOK_L3" # dir to save the models
val_path = "/storage/datset/v1.0-simplified_nq-dev-all.jsonl.gz"
cache_dir = "/storage/datset" 

# Prepare training examples
python3 preprocess.py --pre_size 0 --total_size 300000 --train --data_dir ${data_dir} --cache_dir ${cache_dir}

# Prepare validation examples
python3 preprocess.py --pre_size 300000 --total_size 300050 --sample_size 50 --data_dir ${data_dir} --cache_dir ${cache_dir}

# Train the model
python3 train_squad.py --cp checkpoint-1200 --save_dir ${save_dir} --data_dir ${data_dir}

# Evaluate using dev-all dataset
python3 evaluate.py --cp checkpoint-1200 --dir_name ${save_dir} --diff_threshold -5 --val_path ${val_path}
#python3 evaluate.py --cp checkpoint-6500 --dir_name BERT_SQUAD_TOK_L2 --diff_threshold -5

# Compute the F1 score
pred_path = "$save_dir/eval_predictions.json"
## ensemble ## 
# pred_path = "$save_dir/eval_predictions1.json@$save_dir/eval_predictions2.json"
python3 nq/nq_eval.py --gold_path=${val_path} --predictions_path=${pred_path} --pretty_print