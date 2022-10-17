# Example of pruning bert-base 
export CUDA_VISIBLE_DEVICES=0 
# MNLI 
python train.py \
--initial_threshold 1. --final_threshold 0.10 \
--warmup_steps 5400 --initial_warmup 1 --final_warmup 5 \
--beta1 0.85 --beta2 0.85 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets mnli --test_datasets mnli_matched,mnli_mismatched \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 32 --batch_size_eval 256 \
--optimizer adamax --learning_rate 8e-5 \
--epochs 8 --seed 7 \
--log_per_updates 100 --eval_per_updates 6000 \
--output_dir log/run_mnli --log_file log.log --tensorboard 

# RTE 
python train.py \
--initial_threshold 1. --final_threshold 0.20 \
--warmup_steps 200 --initial_warmup 1 --final_warmup 6 \
--beta1 0.85 --beta2 0.95 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets rte --test_datasets rte \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 16 --batch_size_eval 128 \
--optimizer adamax --learning_rate 1e-4 \
--epochs 20 --seed 8 \
--log_per_updates 10 --eval_per_updates 100 \
--output_dir log/run_rte --log_file log.log --tensorboard 

# QQP 
python train.py \
--warmup_steps 5400 --initial_warmup 1 --final_warmup 5 \
--initial_threshold 1. --final_threshold 0.20 \
--beta1 0.85 --beta2 0.85 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets qqp --test_datasets qqp \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--optimizer adamax --learning_rate 1e-4 \
--batch_size 32 --batch_size_eval 128 \
--epochs 10 --seed 9 \
--log_per_updates 100 --eval_per_updates 3000 \
--output_dir log/run_qqp --log_file log.log --tensorboard 

# SST-2 
python train.py \
--warmup_steps 1000 --initial_warmup 1 --final_warmup 5 \
--initial_threshold 1. --final_threshold 0.10 \
--beta1 0.85 --beta2 0.85 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets sst --test_datasets sst \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 32 --batch_size_eval 256 \
--optimizer adamax --learning_rate 8e-5 \
--epochs 6 --seed 8 \
--log_per_updates 100 --eval_per_updates 500 \
--output_dir log/run_sst --log_file log.log --tensorboard 

# QNLI 
python train.py \
--warmup_steps 2000 --initial_warmup 1 --final_warmup 6 \
--initial_threshold 1. --final_threshold 0.10 \
--beta1 0.85 --beta2 0.90 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets qnli --test_datasets qnli \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 32 --batch_size_eval 512 \
--optimizer adamax --learning_rate 1e-4 \
--epochs 10 --seed 9 \
--log_per_updates 100 --eval_per_updates 1000 \
--output_dir log/run_qnli --log_file log.log --tensorboard  

# STS-B
python train.py \
--warmup_steps 500 --initial_warmup 1 --final_warmup 5 \
--initial_threshold 1. --final_threshold 0.10 \
--beta1 0.85 --beta2 0.85 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets stsb --test_datasets stsb \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 16 --batch_size_eval 512 \
--optimizer adamax --learning_rate 1e-4 \
--epochs 15 --seed 9 \
--log_per_updates 100 --eval_per_updates 500 \
--output_dir log/run_stsb --log_file log.log --tensorboard  

# MRPC 
python train.py \
--warmup_steps 300 --initial_warmup 1 --final_warmup 3 \
--initial_threshold 1. --final_threshold 0.10 \
--beta1 0.85 --beta2 0.95 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets mrpc --test_datasets mrpc \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 8 --batch_size_eval 512 \
--optimizer adamax --learning_rate 1e-4 \
--epochs 10  --seed 9 \
--log_per_updates 100 --eval_per_updates 500 \
--output_dir log/run_mrpc --log_file log.log --tensorboard 

# CoLA
python train.py \
--warmup_steps 500 --initial_warmup 1 --final_warmup 3 \
--initial_threshold 1. --final_threshold 0.10 \
--beta1 0.85 --beta2 0.95 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets cola --test_datasets cola\
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 32 --batch_size_eval 512 \
--optimizer adamax --learning_rate 1e-4 \
--epochs 15 --seed 9 \
--log_per_updates 100 --eval_per_updates 500 \
--output_dir log/run_cola --log_file log.log --tensorboard 

