# Multi-Representation Variational Autoencoder via Iterative Latent Attention and Implicit Differentiation
This repository is the official implementation of paper (https://dl.acm.org/doi/10.1145/3583780.3614980) 
> Nhu-Thuat Tran and Hady W. Lauw. 2023. Multi-Representation Variational Autoencoder via Iterative Latent Attention and Implicit Differentiation. In Proceedings of 32nd ACM International Conference on Information and Knowledge Management (CIKM'2023), Birmingham, UK, October 21-25, 2023.

## Environment
* Anaconda: 4.12.0  
* Python: 3.7.5
* OS: MacOS

## Data
Please follow the instruction in `README.md` file under data folder

## Requirements

Create virtual environment

```conda create --prefix ./valid python=3.7.5 -y```

Activate environment

```conda activate ./valid```

Install requirements

```pip install -r requirements.txt```

## Training and Evaluation

1. Create a YAML config file under ```configs``` folder as samples.

2. Prepare ```run.sh``` file as follows

```python run_valid.py --dataset <dataset_name> --config_file <your_config_file> --device_id <ID of GPU machine>```

3. To run training and evaluation

```bash run.sh```


### Hyper-parameter tuning when training multi-representation VAEs for recommendation
For multi-representation VAEs, temperature hyper-parameter `tau` and `tau_dec` often play the key role in achieving favorable recommendation accuracy.
We first tune their values in the following ranges (while setting others with default values: e.g., the number of interest per user `kfac` is `4`, number of iterative latent attention steps `num_iters` is `1`).
- `tau`: `0.08, 0.1, 0.15, 0.2`
- `tau_dec`: `0.08, 0.1, 0.15, 0.2`

Next, we tune other hyper-parameters, which have data-dependent influence
- `std`: `0.05, 0.075, 0.1`
- `anneal_cap`: `0.1, 0.2, 0.5, 1`
- `total_anneal_steps`: `1000, 3000, 5000, 10000, 20000`

Then, we tune other parameters: the number of interests per user `kfac`, the number of iterative latent attention steps `num_iters`. Please follow the experimental results in the paper.

*Given the tuned hyper-parameters, setting the `num_iters` to `1` will reproduce the results of MacridVAE* 


### Notes on RecBole setting to reproduce results
The default setting of RecBole 1.1 is to exclude user historical interactions in training and validation sets when evaluating on test set. 
However, some existing baselines only exclude user historical interactions in training set when evaluating on test set.
Thus, for fair comparison, after installing RecBole, change line 264 RecBole/recbole/data/utils.py to
```
test_sampler = sampler.set_phase('train')
test_sampler.phase = 'test'
```

## Citation
If you find our work useful for your research, please cite our paper as
```
@inproceedings{VALID,
  author       = {Nhu{-}Thuat Tran and
                  Hady W. Lauw},
  title        = {Multi-Representation Variational Autoencoder via Iterative Latent
                  Attention and Implicit Differentiation},
  booktitle    = {Proceedings of the 32nd {ACM} International Conference on Information
                  and Knowledge Management, {CIKM} 2023, Birmingham, United Kingdom,
                  October 21-25, 2023},
  pages        = {2462--2471},
  year         = {2023}
}
```