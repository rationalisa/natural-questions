# prepare training examples
#python3 preprocess.py --pre_size 0 --total_size 300000 --train
# prepare validation examples
#python3 preprocess.py --pre_size 300000 --total_size 300050 --sample_size 50
#python3 train_squad.py --cp checkpoint-1200
python3 evaluate.py --cp checkpoint-1200 --dir_name BERT_SQUAD_TOK_L3 --max_val_samples -1 
#python3 evaluate.py --cp checkpoint-6500 --dir_name BERT_SQUAD_TOK_L2 --max_val_samples -1