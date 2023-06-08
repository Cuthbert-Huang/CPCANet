export nnUNet_preprocessed='/opt/data/private/CPCANetFrame/DATASET/nnUNet_preprocessed'
export RESULTS_FOLDER='/opt/data/private/CPCANetFrame/DATASET/nnUNet_trained_models'
export nnUNet_raw_data_base='/opt/data/private/CPCANetFrame/DATASET/nnUNet_raw'

name='cpcanet_isic'

step=0.5

task=002

for file_a in $nnUNet_raw_data_base/nnUNet_raw_data/*; do
    temp_folder=`basename $file_a`
	
	if [ ${temp_folder:4:3} = $task ] ; then
	
		target_folder=$temp_folder
	fi
done 

cd $nnUNet_raw_data_base/nnUNet_raw_data/$target_folder/

nnUNet_predict -i imagesTs -o inferTs/${name}_${step}step -m 2d -t ${task} -chk model_best -tr nnUNetTrainerV2_${name} --num_threads_preprocessing 16 --num_threads_nifti_save 16 --step_size ${step}
