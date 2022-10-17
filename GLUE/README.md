## PLATON for GLUE benchmark

This pytorch package implements [PLATON: Pruning Large Transformer Models with Upper Confidence Bound of Weight Importance](https://arxiv.org/pdf/2206.12562.pdf) (ICML 2022).

### Setup Environment
```bash
conda create -n pruning python=3.7
conda activate pruning 
pip install -r requirement.txt
```

### Download Datasets

1. Download data and models </br>
   ```> sh download.sh``` </br>
   Please refer to download GLUE dataset: https://gluebenchmark.com/

2. Preprocess data </br>
   ```> sh experiments/glue/prepro.sh``` 


### Iterative pruning for BERT-base
```bash
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
```

### Instructions

#### Hyperparameter Setup

+ `initial_threshold`: initial remaining ratio $r^{(0)}$. 
+ `final_threshold`: final remaining ratio $r^{(T)}$. 
+ initial warmup steps for pruning is equal to `initial_warmup` $\times$ `warmup_steps`. 
+ final warmup steps for pruning is equal to `final_warmup` $\times$ `warmup_steps`. 
+ `beta1`: $\beta_1$ for PLATON. 
+ `beta2`: $\beta_2$ for PLATON. 
+ `deltaT`: the length of local average window. 

#### Other examples

The floder `scripts` contains more examples of pruning BERT-base on GLUE datasets. 


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