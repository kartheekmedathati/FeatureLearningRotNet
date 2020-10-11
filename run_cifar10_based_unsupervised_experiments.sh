#!/bin/bash

# Train a RotNet (with a NIN architecture of 4 conv. blocks) on training images of CIFAR10.
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_Orig_RotNet
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_Split_Bands
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_Hue_Rotate
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_Hue_Rotate_8
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_Geo_Photo
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_Geo_Photo_from_182
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_RotNet_NIN4blocks_Geo_Photo_all_combinations

# Train & evaluate an object classifier (for the CIFAR10 task) with convolutional 
# layers on the feature maps of the 2nd conv. block of the RotNet model trained above.
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Orig_RotNet
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Split_Bands
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Hue_Rotate
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Hue_Rotate_8
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Geo_Photo
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Geo_Photo_from_18
CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_ConvClassifier_on_RotNet_NIN4blocks_Conv2_feats_Geo_Photo_all_combinations


# Train & evaluate an object classifier (for the CIFAR10 task) with 3 fully connected layers 
# on the feature maps of the 2nd conv. block of the above RotNet model trained above.
#CUDA_VISIBLE_DEVICES=0 python main.py --exp=CIFAR10_MultLayerClassifier_on_RotNet_NIN4blocks_Conv2_feats
