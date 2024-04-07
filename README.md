# CS6910_Assignment2

Following are the 2 main goals of this Assignment : 
(i) train a CNN model from scratch and learn how to tune the hyperparameters and visualize filters 
(ii) finetune a pre-trained model just as you would do in many real-world applications

I have split it into 2 parts : partA and partB. 

## Part A

In partA, I have build a small CNN model consisting of 5 convolution layers where each convolution layer is followed by an activation & a maxpooling layer.
The model is trained on iNaturalist Dataset which contains 10,000 training images and 2,000 test images. Also, I have set aside 20% of training data as validation data for hyperparameter tuning where I made sure each class is equally represented. Following are my hyperparameter configurations :

```
sweep_config = {
    'parameters' : {
        'epochs' : {
            'values' : [5, 10]
        },
        'num_filters' : {
            'values' : [32, 64]
        },
        'filter_size' : {
            'values' : [[3, 3, 3, 3, 3], [2, 2, 2, 2, 2]]
        },
        'dropout': {
            'values' : [0.2, 0.3, 0.4, 0.5]
        },
        'batch_size' : {
            'values' : [32, 64]
        },
        'data_aug' :  {
            'values' : [True, False]
        },
        'batch_norm' : {
            'values' : [True, False]
        },
        'learning_rate' : {
            'values' : [1e-3, 5e-3, 1e-4]
        },
        'activation' : {
            'values' : ['Mish', 'GELU', 'ReLU', 'SiLU']
        },
        'filter_org' : {
            'values' : ['same', 'double', 'half']
        }
    }
}
```
After performing many sweeps I have obtained best model with training accuracy 41.81%: , validation accuracy : 35.92% and test accuracy : 37.15%.


## Part B

In part B, I have implemented a pre-trained ResNet Model and fine-tuned it on naturalist dataset. Here, instead of randomly initializing weights of network I am using weights resulting from training the model on the ImageNet data (torchvision directly provides these weights). Inorder to train it on my dataset I had to make few modifications like transforming dimensions of image and modifying last of pre-trained model to classify only 10 classes. 

Following are the 3 fine-tuning techniques that I learned and understood:

a] Fine-tuning All Layers

b] Freezing all layers except last

c] Freezing upto k layers

From the above techniques, I have implemented Fine-tuning by freezing all the layers except the final layer for ResNet model and obtained validation accuracy of  82.29 % and test accuracy of 80.1 %.


