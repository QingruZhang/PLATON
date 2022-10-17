## PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance

This pytorch package implements [PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance](https://arxiv.org/pdf/2206.12562.pdf) (ICML 2022).

### Setup environment

```bash
conda create -n pruning python=3.7
conda activate pruning 
pip install -r requirement.txt
```

### Usage of PLATON

* Import Pruner:
```
from Pruner import Pruner 
```
* Initialize the pruner: 
```
PLATON = Pruner(model, args=args, pruner_name="PLATON", total_step=t_total, 
		mask_param_name=['attention.self', 'attention.output.dense', 'output.dense', 'intermediate.dense'])
```

+ `model`: the model to be pruned. 
+ `args.initial_threshold`: initial remaining ratio $r^{(0)}$. 
+ `args.final_threshold`: final remaining ratio $r^{(T)}$. 
+ initial warmup steps for pruning is equal to `initial_warmup` $\times$ `warmup_steps`. 
+ final warmup steps for pruning is equal to `final_warmup` $\times$ `warmup_steps`. 
+ `args.beta1`: $\beta_1$ for PLATON. 
+ `args.beta2`: $\beta_2$ for PLATON. 
+ `args.deltaT`: the length of local average window. 
+ `mask_param_name`: the list of substrings of names of pruned parameters. 

* After each step of `optimizer.step()`, add the following line to update $\overline{I}$, $\overline{U}$ and prune the model iteratively. 
```
threshold, mask_threshold = PLATON.update_and_pruning(model, global_step)
```

### GLUE benchmark

Check the folder `GLUE` for more details about reproducing the GLUE results. 
An example of iterative pruning for BERT-base on MNLI: 

```bash
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
```

Please see [`GLUE/scripts`](https://github.com/QingruZhang/PLATON/tree/main/GLUE/scripts) for more examples of GLUE. 


### Question Answering Task

Check the folder `SQuAD` for more details about reproducing the results of SQuAD. 
An example of iterative pruning for BERT-base on SQuADv1.1: 

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

