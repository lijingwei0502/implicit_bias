# Understanding Nonlinear Implicit Bias via Region Counts in Input Space
This repository contains the code of the paper [Understanding Nonlinear Implicit Bias via Region Counts in Input Space](https://arxiv.org/pdf/2505.11370).

## Requirements

Install dependencies from the provided conda environment file:

```bash
conda env create -f environment.yml
````

---

## Train the Model

All training options are configurable via command-line arguments.

**Basic training example (CIFAR10, ResNet18):**

```bash
python main.py \
    --dataset cifar10 \
    --net Resnet18 \
    --training_epochs 200 \
    --batch_size 256 \
    --lr 0.01 \
    --optimizer sgd \
    --scheduler cosine \
    --data_choose 1
```

**Common arguments:**

* `--dataset` — dataset name (`cifar10`, `cifar100`, `imagenet`, etc.)
* `--net` — network architecture (`Resnet18`, `Resnet34`, …)
* `--training_epochs` — number of training epochs (default `200`)
* `--batch_size` — batch size (default `256`)
* `--lr` — learning rate (default `0.01`)
* `--optimizer` — optimizer type (`sgd`, `adam`, etc.)
* `--scheduler` — LR scheduler (`cosine`, `step`, etc.)
* `--data_choose` — how to choose data to generate calculate plane (`1d`, `2d`, etc.)

The script will:

1. Load the specified dataset and apply optional data augmentation.
2. Initialize the model, optimizer, and scheduler.
3. Train the model and compute **region count** in the input space.
4. Save checkpoints if `--resume True` is used.

---

**Acknowledgements**: this repository uses codes and resources from [Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar).

## Citation

```
@inproceedings{li2025understanding,
  title={Understanding Nonlinear Implicit Bias via Region Counts in Input Space},
  author={Li, Jingwei and Xu, Jing and Wang, Zifan and Zhang, Huishuai and Zhang, Jingzhao},
  booktitle={International Conference on Machine Learning},
  year={2025},
  organization={PMLR}
}
```
