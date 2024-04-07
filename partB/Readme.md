# Part B

I have implemented a pre-trained ResNet Model and fine-tuned it on naturalist dataset. Here, instead of randomly initializing weights of network I am using weights resulting from training the model on the ImageNet data (torchvision directly provides these weights). Inorder to train it on my dataset I had to make few modifications like transforming dimensions of image and modifying last of pre-trained model to classify only 10 classes.

Following are the 3 fine-tuning techniques that I learned and understood:

a] Fine-tuning All Layers

b] Freezing all layers except last

c] Freezing upto k layers

## Usage
To run the script, use the following command-line arguments:

### Command-line Arguments

| Argument             | Shorthand   | Description                                                                     | Choices                   | Default Value      |
|----------------------|-------------|---------------------------------------------------------------------------------|---------------------------|--------------------|
| `-wp, --wandb_project` | `WAND_PROJECT` | Project name for Weights & Biases dashboard experiments.                        | Any string                | "DL_Assignment_2" |
| `-we, --wandb_entity`  | `WAND_ENTITY`  | Wandb Entity used to track experiments in the Weights & Biases dashboard.       | Any string                | "cs23m009"        |
| `-e, --epochs`         | `EPOCHS`       | Number of epochs to train the model.                                             | Any integer               | 5                  |
| `-b, --batch_size`     | `BATCH_SIZE`   | Batch size used to train the model.                                               | Any integer               | 32                 |
| `-f, --finetune_strategy` | `FINETUNE_STRATEGY` | Different Fine-tuning strategies.                                              | "feature_extraction", "fine_tuning_all", "layer_wise_fine_tuning" | "feature_extraction" |
| `-lr, --learning_rate` | `LEARNING_RATE` | Learning rate used to optimize model parameters.                                | Any float                 | 0.0001             |

## Example

```
python train_partb.py --wandb_project "My_Project" --wandb_entity "my_username" --epochs 10 --batch_size 32 --finetune_strategy "feature_extraction" --learning_rate 0.001

```

OR 

```
python train_partb.py
```
