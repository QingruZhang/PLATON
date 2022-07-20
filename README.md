## PLATON for Question Answering Task

This pytorch package implements (PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance)[https://arxiv.org/pdf/2206.12562.pdf] (ICML 2022).

### Setup
```bash
conda create -n pruning python=3.7
conda activate pruning 
pip install -r requirement.txt
bash download_dataset.sh 
```

### Iterative pruning for Bert-base
```bash
python run_squad.py --pruner_name PLATON \
--initial_threshold 1 --final_threshold 0.10 \
--warmup_steps 5400 --initial_warmup 1 --final_warmup 5 \
--beta1 0.85 --beta2 0.950 --deltaT 10 \
--num_train_epochs 10 --seed 9 --learning_rate 3e-5 \
--per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 256 \
--do_train --do_eval --do_lower_case \
--model_type bert --model_name_or_path bert-base-uncased \
--logging_steps 300 --eval_steps 3000 --save_steps 100000 \
--data_dir data/squad \
--output_dir log/deberta-v3-base/PLATON/ --overwrite_output_dir
```


### Iterative pruning for DebertaV3-base
```bash 
python run_squad.py --pruner_name PLATON \
--initial_threshold 1 --final_threshold 0.10 \
--warmup_steps 5400 --initial_warmup 1 --final_warmup 5 \
--beta1 0.85 --beta2 0.950 --deltaT 10 \
--num_train_epochs 10 --seed 9 --learning_rate 3e-5 \
--per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 256 \
--do_train --do_eval --do_lower_case \
--model_type deberta --model_name_or_path microsoft/deberta-v3-base \
--logging_steps 300 --eval_steps 3000 --save_steps 100000 \
--data_dir data/squad \
--output_dir log/deberta-v3-base/PLATON/ --overwrite_output_dir
```

### Instructions

#### Hyperparameter Setup

+ `initial_threshold`: initial remaining ratio $r^{(0)}$. 
+ `final_threshold`: final remaining ratio $r^{(T)}$. 
+ initial warmup steps for pruning is equal to `initial_warmup` $\times$ `warmup_steps`. 
+ final warmup steps for pruning is equal to `final_warmup` $\times$ `warmup_steps`. 
+ `beta1`: $\beta_1$ for PLATON. 
+ `beta2`: $\beta_2$ for PLATON. 

#### Other Comment

+ Thanks for checking our repo. The source code of GLUE will be relased within one week. 


### Citation
```
@inproceedings{zhang2022platon,
  title={PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance},
  author={Zhang, Qingru and Zuo, Simiao and Liang, Chen and Bukharin, Alexander and He, Pengcheng and Chen, Weizhu and Zhao, Tuo},
  booktitle={International Conference on Machine Learning},
  pages={26809--26823},
  year={2022},
  organization={PMLR}
}
```