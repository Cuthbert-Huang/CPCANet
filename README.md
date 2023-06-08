# Channel prior convolutional attention for medical image segmentation

by Hejun Huang, Zuguo Chen*, Ying Zou, Ming Lu, Chaoyang Chen

## Introduction

This repository is for our paper 'Channel prior convolutional attention for medical image segmentation'.

## Installation

```
git clone https://github.com/Cuthbert-Huang/CPCANet
cd CPCANet
conda env create -f environment.yml
source activate CPCANet
pip install -e .
```

## Data-Preparation

CPCANet is a 2D based network, and all data should be expressed in 2D form with ```.nii.gz``` format. You can download the organized dataset from the [link](https://drive.google.com/drive/folders/1b4IVd9pOCFwpwoqfnVpsKZ6b3vfBNL6x?usp=sharing) or download the original data from the link below. If you need to convert other formats (such as ```.jpg```) to the ```.nii.gz```, you can look up the file and modify the [file]() based on your own datasets.

**Dataset I**
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)

**Dataset II**
[ISIC2016](https://www.isic-archive.com/#!/topWithHeader/wideContentTop/main), [PH2](https://www.fc.up.pt/addi/ph2%20database.html)

The dataset should be finally organized as follows:

```
./DATASET/
  ├── nnUNet_raw/
      ├── nnUNet_raw_data/
          ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
              ├── evaulate.py

          ├── Task02_Isic/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
              ├── evaulate.py              
          ......
      ├── nnUNet_cropped_data/
  ├── nnUNet_trained_models/
  ├── nnUNet_preprocessed/
```

One thing you should be careful of is that folder imagesTr contains both training set and validation set, and correspondingly, the value of ```numTraining``` in dataset.json equals the case number in the imagesTr. The division of the training set and validation set will be done in the network configuration located at ```nnunet/network_configuration/config.py```.

The evaulate.py is used for calculating the evaulation metrics and can be found in the [link](https://drive.google.com/drive/folders/1b4IVd9pOCFwpwoqfnVpsKZ6b3vfBNL6x?usp=sharing) of the organized datasets or you can write it by yourself. The existing of evaulate.py will not affect the data preprocessing, training and testing.

## Data-Preprocessing

```
nnUNet_convert_decathlon_task -i path/to/nnUNet_raw_data/Task01_ACDC
```

This step will convert the name of folder from Task01 to Task001, and make the name of each nifti files end with '_000x.nii.gz'.

```
nnUNet_plan_and_preprocess -t 1
```

Where ```-t 1``` means the command will preprocess the data of the Task001_ACDC.
Before this step, you should set the environment variables to ensure the framework could know the path of ```nnUNet_raw```, ```nnUNet_preprocessed```, and ```nnUNet_trained_models```. The detailed construction can be found in [nnUNet](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/setting_up_paths.md).

## Training

If you want to train CPCANet on ACDC.

```
bash train_cpcanet_acdc.sh
```

If you want to train CPCANet on ISIC.

```
bash train_cpcanet_isic.sh
```

## Testing

The trained model is placed at this link for model testing.

If you want to test CPCANet on ACDC.

```
bash test_cpcanet_acdc.sh
```

If you want to test CPCANet on ISIC.

```
bash test_cpcanet_isic.sh
```

## Acknowledgements

Our code is origin from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [UNET-2022](https://github.com/282857341/UNet-2022). Thanks to these authors for their excellent work.
