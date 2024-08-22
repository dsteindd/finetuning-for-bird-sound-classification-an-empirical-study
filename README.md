# Code for the Paper "Fine-tuning for Bird Sound Classification: An Empirical Study"

This is the code for "Fine-tuning for Bird Sound Classification: An Empirical Study", accepted at the ECCV'24 workshop "Computer Vision for Ecology".

If you find the paper and code valuable, please cite
```bibtex
@inproceedings{
  stein2024,
  title={Fine-tuning for Bird Sound Classification: An Empirical Study},
  author={Stein, David and Andres, Bjoern},
  booktitle={ECCV Workshops},
  year={2024}
}
```

## Requirements
You need to have Python 3.9 installed. Then, run the following commands to setup a virtual environment: 
```shell
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

### ESC-50 
Download ESC-50 from [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50).

Put it into `./data/ESC-50-master`.

### Metadata Files
Create metadata files `./data/metadata-finetune.json` and `./data/metadata-pretrain.json` like below both for your finetuning and pretraining datasets.

```json
{
  "A.wav": [0],
  "B.wav": [1]
}
```

### Species Lists
Create species lists `./data/species-list-finetune.txt` and `./data/species-list-pretrain.txt` like below both for your finetuning and pretraining datasets.

```csv
0 Species_A
1 Species_B
...
```

### Folds
Create five folds files called `./data/folds-all/fold-0.txt` through `./data/folds-all/fold-4.txt` for your finetuning data.
A fold file is a text file containing one filename of your finetuning dataset per line.

## How to run pretraining?
Run the following command to pretrain all networks discussed in the paper on your pretraining dataset.
```shell
python pretrain.py --networks resnet18 resnet34 resnet50 wide-resnet50 bird-net efficient-net-s
```

## How to run fine-tuning?
Run the following command to train a network architecture on your finetuning folds from scratch. 
Replace <network> with the network architecture you want to finetune on.
```shell
python downstream.py --network <network> --augment 0.0
```

Run the following command to train a network architecture on your finetuning folds from a pretrained checkpoint. 
Replace <network> with the network architecture you want to finetune on.
```shell
python downstream.py --network <network> --augment 0.0 --pretrain-path ./models-pretrain/your-network
```

## How to evaluate?

Run 

```shell
python evaluate.py --augment 0.2 --model-dir ./models-downstream/from_scratch/ --batch-size ${BATCH_SIZE}
```
