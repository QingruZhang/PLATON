# RTE
export CUDA_VISIBLE_DEVICES=0 
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

python train.py \
--initial_threshold 1. --final_threshold 0.15 \
--warmup_steps 200 --initial_warmup 1 --final_warmup 6 \
--beta1 0.85 --beta2 0.65 --deltaT 10 \
--data_dir data/canonical_data/bert-base-uncased \
--train_datasets rte --test_datasets rte \
--init_checkpoint mt_dnn_models/bert_model_base_uncased.pt \
--batch_size 16 --batch_size_eval 128 \
--optimizer adamax --learning_rate 1e-4 \
--epochs 20 --seed 8 \
--log_per_updates 10 --eval_per_updates 100 \
--output_dir log/run_rte --log_file log.log --tensorboard 

python train.py \
--initial_threshold 1. --final_threshold 0.10 \
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
