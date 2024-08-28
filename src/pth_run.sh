#!/bin/bash

# define the experiment
experiment(){
    # root=$1
    per_class_num=$1
    sound_save_path=$2
	mnist_save_path=$3
    soundmnist_save_path=$4
    sound_mean_save_path=$5
    sound_mean_name=$6
    metadropout_save_path=$7
    finetune_save_path=$8
    batch_size=$9
	vis_device=${10}

    # cd "$root"
	printf "Batch size $batch_size \n"
	printf "==========> start experiment $per_class_num: %(%Y-%m-%d %H:%M:%S)T <==========\n"
	printf "==========> start experiment $per_class_num: %(%Y-%m-%d %H:%M:%S)T <==========\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt
	
	printf "\n==========> train_mnist\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt
    printf "\n==> start train_mnist !!!\n"
	
    python train_mnist.py \
    	--checkpoint $mnist_save_path \
			--vis_device $vis_device \
    	--per_class_num $per_class_num >> experiment-$vis_device:per-class-num-$per_class_num.txt   

    #get best mnist model
    max_mnist=0.0
    for entry in "$mnist_save_path"/*;do
    	# echo $entry
    	arr=($(find "$entry"/* -name "*best_model*"))
    	for idx in "${arr[@]}";do 
    		IFS='_' read -ra ADDR <<< "$idx"
	    	if (( $(echo "$max_mnist < ${ADDR[4]}" |bc -l) ));then 
	    		max_mnist=${ADDR[4]}
	    		best_mnist_path=$entry
	    		IFS='/' read -ra SNAME <<< "$idx"
	    		best_mnist_name=${SNAME[-1]} 
    		fi
		done		
	done

	printf "\n==> done train_mnist !!!\n"

	printf "\n==========> train_sound\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt      
    
    # train sound model
    printf "\n==> start train_sound !!!\n"

    python train_sound.py \
    	--checkpoint $sound_save_path \
			--vis_device $vis_device \
    	--per_class_num $per_class_num >> experiment-$vis_device:per-class-num-$per_class_num.txt   

    #get best sound model
    max_sound=0.0
    for entry in "$sound_save_path"/*;do
    	# echo $entry
        printf "\n==> Entry in sound: $entry !!!\n"

    	arr=($(find "$entry"/* -name "*best_model*"))
    	for idx in "${arr[@]}";do 
    		IFS='_' read -ra ADDR <<< "$idx"
	    	if (( $(echo "$max_sound < ${ADDR[4]}" |bc -l) ));then 
	    		max_sound=${ADDR[4]}
	    		best_sound_path=$entry
	    		IFS='/' read -ra SNAME <<< "$idx"
	    		best_sound_name=${SNAME[-1]} 
    		fi
		done		
	done

	printf "\n==> done train_sound !!!\n"

	printf "\n==> start train_soundmnist !!!\n"
	printf "\n==========> train_soundmnist\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt   
	# train sound mnist
    
    # printf "\n==> Best_sound_name $best_sound_name !!!\n"
    # printf "\n==> Sound model path $best_sound_path !!!\n"
    # printf "\n==> Img model path $best_mnist_path !!!\n"


	python train_soundmnist.py \
		--checkpoint $soundmnist_save_path \
		--per_class_num $per_class_num \
		--vis_device $vis_device \
		--sound_model_path $best_sound_path \
		--img_model_path $best_mnist_path \
		--sound_model_name $best_sound_name >> experiment-$vis_device:per-class-num-$per_class_num.txt \
        --img_model_name  $best_mnist_name >> experiment-$vis_device:per-class-num-$per_class_num.txt  

	#get best soundmnist model
    max_soundmnist=0.0
    for entry in "$soundmnist_save_path"/*;do
    	# echo $entry
    	arr=($(find "$entry"/* -name "*best_model*"))
    	for idx in "${arr[@]}";do 
    		IFS='_' read -ra ADDR <<< "$idx"
	    	if (( $(echo "$max_soundmnist < ${ADDR[4]}" |bc -l) ));then 
	    		max_soundmnist=${ADDR[4]}
	    		best_soundmnist_path=$entry
	    		IFS='/' read -ra SNAME <<< "$idx"
	    		best_soundmnist_name=${SNAME[-1]} 
    		fi
		done		
	done


	printf "\n==> done train_soundmnist !!!\n"

	printf "\n==> start sound_mean !!!\n"
	printf "\n==========> get_sound_mean\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt      
	# get sound_mean
	python get_sound_mean_kmean.py \
		--checkpoint $sound_mean_save_path \
		--per_class_num $per_class_num \
		--vis_device $vis_device \
		--soundmnist_model_path $best_soundmnist_path \
		--soundmnist_model_name $best_soundmnist_name >> experiment-$vis_device:per-class-num-$per_class_num.txt      

	printf "\n==> done sound_mean !!!\n"

	printf "==> star training with missing modality\n"
	printf "\n==> star training with missing modality\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt   
	
	python train_missing_eval_missing.py \
		--checkpoint $metadropout_save_path \
		--per_class_num $per_class_num \
		--sound_mean_path $sound_mean_save_path \
		--sound_mean_name $sound_mean_name \
		--batch_size $batch_size \
		--vis_device $vis_device \
		--soundmnist_model_path $best_soundmnist_path \
		--soundmnist_model_name $best_soundmnist_name >> experiment-$vis_device:per-class-num-$per_class_num.txt   
	
	printf "==> end training with missing modality\n"
	printf "\n==> end training with missing modality\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt   

	printf "==========> end experiment $per_class_num: %(%Y-%m-%d %H:%M:%S)T <==========\n" >> experiment-$vis_device:per-class-num-$per_class_num.txt    
	printf "==========> end experiment $per_class_num: %(%Y-%m-%d %H:%M:%S)T <==========\n"

}

run (){
	experiment "$@"
}

# run 15 ./save/sound/450/new/15  ./save/soundmnist/new/15 ./save/sound_mean/new/ sound_mean_150.npy ./save/metadrop/feature/new/15 ./save/finetune/15 2048 1
run 15 ./save/sound/new/15 ./save/mnist/new/15 ./save/soundmnist/new/15 ./save/sound_mean/new/ sound_mean_150.npy ./save/metadrop/feature/new/15 ./save/finetune/15 4096 1
