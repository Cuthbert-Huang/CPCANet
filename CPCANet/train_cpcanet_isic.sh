export nnUNet_preprocessed='/opt/data/private/CPCANetFrame/DATASET/nnUNet_preprocessed'
export RESULTS_FOLDER='/opt/data/private/CPCANetFrame/DATASET/nnUNet_trained_models'
export nnUNet_raw_data_base='/opt/data/private/CPCANetFrame/DATASET/nnUNet_raw'

name='cpcanet_isic'

task=002

nnUNet_train 2d nnUNetTrainerV2_${name} ${task} 0
