# Part A

## HyperParameter Searching 

- No. of epochs
- No. of filters
- Batch normalization
- Learning Rate
- Data Augmentation
- Batch Size
- Dropout
- Filter Size
- Filter Organization
- Activation

## Best Model Configuration

- Learning Rate : 1e-4
- Activation Function : GELU
- Optimizer : Adam
- No. of epochs : 5
- Batch Size : 32
- Batch Normalization : True
- Data Augmentation : False
- Kernel Size : [2, 2, 2, 2, 2]
- Dropout : 0.2
- Filter Organization : same

## Python Script for Training Model

I designed a python script (train.py) to train above CNN with different parameters. Inorder to execute it just run below command :

```
python train_parta.py --wandb_project "myprojectname" --wandb_entity "myname" --epoch 5 --batch_size 32  --learning_rate 0.0001 --activation "GELU" --batch_norm True --data_aug False --dropout 0.2 --filter_org "same" --filter_size [2, 2, 2, 2, 2]
```

OR

To just run best model just execute :

```
python train_parta.py
```

## Command-line Arguments

| Argument          | Shorthand | Description                                                                           | Choices          | Default Value |
|-------------------|-----------|---------------------------------------------------------------------------------------|------------------|---------------|
| `-wp, --wandb_project` | `WAND_PROJECT` | Project name used to track experiments in the Weights & Biases dashboard.             | Any string       | "DL_Assignment_2" |
| `-we, --wandb_entity`  | `WAND_ENTITY`  | Wandb Entity used to track experiments in the Weights & Biases dashboard.             | Any string       | "cs23m009"    |
| `-n, --num_filters`    | `NUM_FILTERS`  | Number of filters.                                                                    | 32, 64           | 32            |
| `-e, --epochs`         | `EPOCHS`       | Number of epochs to train the model.                                                  | Any integer      | 5             |
| `-b, --batch_size`     | `BATCH_SIZE`   | Batch size used to train the model.                                                    | Any integer      | 32            |
| `-bn, --batch_norm`    | `BATCH_NORM`   | Whether to use Batch Normalization or not.                                             | True, False      | True          |
| `-da, --data_aug`      | `DATA_AUG`     | Whether to perform Data Augmentation or not.                                           | True, False      | False         |
| `-lr, --learning_rate` | `LEARNING_RATE`| Learning rate used to optimize model parameters.                                       | Any float        | 0.0001        |
| `-dp, --dropout`       | `DROPOUT`      | Dropout value.                                                                         | Any float        | 0.2           |
| `-fs, --filter_size`   | `FILTER_SIZE`  | Filter size for 5 layers.                                                              | [2, 2, 2, 2, 2], [3, 3, 3, 3, 3] | [2, 2, 2, 2, 2] |
| `-fo, --filter_org`    | `FILTER_ORG`   | Filter organization choices.                                                           | "same", "double", "half" | "same" |
| `-a, --activation`     | `ACTIVATION`   | Activation function choices.                                                            | "Mish", "GELU", "ReLU", "SiLU" | "ReLU" |
