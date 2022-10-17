# MNLI
export CUDA_VISIBLE_DEVICES=0 
python train.py \
--initial_threshold 1. --final_threshold 0.20 \
--warmup_steps 5400 --initial_warmup 1 --final_warmup 5 \
--beta1 0.85 --beta2 0.95 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets mnli --test_datasets mnli_matched,mnli_mismatched \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 32 --batch_size_eval 256 \
--optimizer adamax --learning_rate 8e-5 \
--epochs 8 --seed 7 \
--log_per_updates 100 --eval_per_updates 6000 \
--output_dir log/run --log_file log.log --tensorboard 


python train.py \
--initial_threshold 1. --final_threshold 0.15 \
--warmup_steps 5400 --initial_warmup 1 --final_warmup 5 \
--beta1 0.85 --beta2 0.90 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets mnli --test_datasets mnli_matched,mnli_mismatched \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 32 --batch_size_eval 256 \
--optimizer adamax --learning_rate 8e-5 \
--epochs 8 --seed 7 \
--log_per_updates 100 --eval_per_updates 6000 \
--output_dir log/run --log_file log.log --tensorboard 

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
--output_dir log/run --log_file log.log --tensorboard 
