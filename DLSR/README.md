
Create a new conda environment and install PyTorch:

conda create -n RSCAN python=3.8 numpy
conda activate RSCAN
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia

==================================================================
Requirement File
=================================================================

 'scipy>=1.5',
 'scikit-learn>=0.23.1',
 'scikit-image>=0.17.2',
 'matplotlib>=3.3.0',
 'yacs>=0.1.8',
  'imageio>=2.9.0',
  'GPUtil>=1.4.0',
  'tqdm>=4.62.0'

====================================================================
Structure of Dataset for  training
=================================================================
To train on your own dataset (supposing the dataset contains 99 training images in this example). 
First, structure the data as follows in the dataset directory:

```
MyData/
  MyData_train_HR/
    0001.png
    0002.png
    ...
    0099.png
  MyData_train_LR_bicubic/
    X2/
      0001x2.png
      ...
    X3/
      0001x3.png
      ...
    X4/
      0001x4.png
      ...
 MyData_train_LR_bicubic_blur/
    X2/
      0001x2.png
      ...
    X3/
      0001x3.png
      ...
    X4/
      0001x4.png
      ...

```

Then update the configuration options in the YAML file:

```yaml
DATASET:
  DATA_EXT: bin
  DATA_DIR: path/to/data
  DATA_TRAIN: ['MyData']
  DATA_VAL: ['MyData']
  DATA_RANGE: [[1, 95], [96, 99]] # split training and validation
```

Note that `DATASET.DATA_EXT: bin` will create a `bin` folder in the dataset directory and save individual images as a single binary file for fast data loading.


======================================================================
Command used for training the network 
=======================================================================

#zooming factor 2
CUDA_VISIBLE_DEVICES=1 python main.py --config-base configs/RCAN/RCAN_Improved.yaml 
--config-file configs/RCAN/RCAN_x2.yaml

#zooming factor 3
CUDA_VISIBLE_DEVICES=1 python main.py --config-base configs/RCAN/RCAN_Improved.yaml 
--config-file configs/RCAN/RCAN_x3.yaml

#zooming factor 4
CUDA_VISIBLE_DEVICES=1 python main.py --config-base configs/RCAN/RCAN_Improved.yaml 
--config-file configs/RCAN/RCAN_x4.yaml

=================================================================
Trained models are stored in "output" folder
===============================================================

==============================================================
Testing images are kept in the following way: "sr_test/HR"
================================================================
#zooming factor 2
HR images are kept in "sr_test/HR/MyData_Test/x2"
LR imgaes are kept in "sr_test/LRBI/MyData_Test/x2"
LR sharp imgaes are kept in "sr_test/LRBI_sharp/MyData_Test/x2"

#zooming factor 3
HR images are kept in "sr_test/HR/MyData_Test/x3"
LR imgaes are kept in "sr_test/LRBI/MyData_Test/x3"


#zooming factor 4
HR images are kept in "sr_test/HR/MyData_Test/x4"
LR imgaes are kept in "sr_test/LRBI/MyData_Test/x4"
=================================================================
Command used for testing the network 
=====================================================================
CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9992 main.py --distributed -
-config-base configs/RCAN/RCAN_Improved.yaml --config-file configs/RCAN/RCAN_x2.yaml 
MODEL.PRE_TRAIN "outputs/RCAN_x2_Dec2822_1114/model/model_best.pth.tar" 
SOLVER.TEST_ONLY True

==============================================================
The SR output is stored in "output" folder
============================================================== 