# STA-GCN

This repo is the official implementation for [Spatial-Temporal Adaptive Graph Convolutional Network for Skeleton-based Action Recognition](https://openaccess.thecvf.com/content/ACCV2022/papers/Hang_Spatial-Temporal_Adaptive_Graph_Convolutional_Network_for_Skeleton-based_Action_Recognition_ACCV_2022_paper.pdf). The paper is accepted to ACCV2022.

# Prerequisites

- This code is based on [Python3](https://www.anaconda.com/) (anaconda, >= 3.7) and [PyTorch](http://pytorch.org/) (>= 1.7.0).
- Other Python libraries are presented in the **'scripts/requirements.txt'**, which can be installed by
    
    ```
    pip install -r scripts/requirements.txt
  ```
    

# Data Preparation

## There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- Kinetics Skeleton

## Download datasets.

- Download the raw data from [NTU-RGB+D 60 & 120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) and [Kinetics Skeleton](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md#kinetics-skeleton). Then put them under the data directory:
    
    ```
       -data\  
       	-Kinetics_skeleton\  
       		-kinetics_train\
       			...
       		-kinetics_val\
       			...
       		-kinetics_train_label.json
       		-keintics_val_label.json
       	-NTU_skeleton\
       		-nturgbd_skeletons_s001_to_s017
       			...
       		-nturgbd_skeletons_s018_to_s032
       			...
  ```
    

## Generating Data.

- Generate numpy datasets by using **'scripts/auto\_gen\_data.sh'**.
    
    ```
  sh scripts/auto_gen_data.sh
  ```
    

# Training & Testing

## Training

- You can simply train the model by
    
    ```
  python main.py -c <config>
  ```
    
    **Note:** **'--config'** or **'-c'**: The config of STA-GCN. You **must** use this parameter on the command line, otherwise the program will output an error. There are 20 configs given in the **configs** folder
	
	- Example: train the model of STA-GCN using joint modality on NTU RGB+D 60 cross subject

		```
		python main.py -c ntu60_xsub_j
	  ```
## Testing

- To evaluate the trained models saved in &lt;work_dir&gt;, run the following command:
    
    ```
 	python main.py -c <config> -e
  ```
	- Example: evaluate the trained model of STA-GCN using joint modality on NTU RGB+D 60 cross subject

		```
		python main.py -c ntu60_xsub_j -e
	  ```

## Ensemble
- To ensemble the results of different modalities, the first step is to obtain scores for different modalities
    
    ```
  python main.py -c <config> -e -sc
  ```
    
	- Example: obtain scores of four modalities of STA-GCN on NTU RGB+D 60 cross subject

		```
		python main.py -c ntu60_xsub_j -sc
		python main.py -c ntu60_xsub_jm -sc
		python main.py -c ntu60_xsub_b -sc
		python main.py -c ntu60_xsub_bm -sc
	  ```

- Then,  ensemble the scores of different modalities
    ```
  python ensemble.py -d <dataset> -n <number of streams>
  ```

	- Example:  ensemble the scores of four modalities of STA-GCN on NTU RGB+D 60 cross subject

		```
		python ensemble.py -d ntu60 -n 4
	 	```

## Pretrained Models
Pretrained models are provided, These models can be downloaded from [BaiduYun](https://pan.baidu.com/s/1fn5kAqbgi1z2KV318zEZZQ) (Extraction code: **azsa**). Download and extract the **pretrained** folder to the root directory of STA-GCN.
- To evaluate the pretrained models, run the following command:
    ```
 	python main.py -c <config> -e -pre
	```
	- Example: evaluate the pretrained model of STA-GCN using joint modality on NTU RGB+D 60 cross subject
		```
		python main.py -c ntu60_xsub_j -e -pre
	  ```
- To obtain scores for different modalities of pretrained models, run
	```
  python main.py -c <config> -e -sc -pre
  ```
- ensemble the scores of different modalities
	```
	python ensemble.py -d <dataset> -n <number of streams>
	```

# Acknowledgements

This repo is based on [EfficientGCN](https://github.com/yfsong0709/EfficientGCNv1). Thanks to the original authors for their work!