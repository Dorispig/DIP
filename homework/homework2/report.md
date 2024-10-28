# Assignment 2 - DIP with PyTorch

This repository is the official implementation of [Assignment 2 - DIP with PyTorch](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/02_DIPwithPyTorch).  
Include [Poisson Blending](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/02_DIPwithPyTorch) and [Pix2Pix](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/02_DIPwithPyTorch/Pix2Pix)
## 1.Poisson Blending
This repository is the official implementation of [Poisson Blending](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/02_DIPwithPyTorch). 
>![鲨鱼](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/result/water.png "鲨鱼")

### Requirements

To install requirements:

```setup
numpy
gradio==3.36.1
pillow
pytorch
```

>pip install numpy  
pip install gradio==3.36.1  
conda install pillow  
conda install pytorch torchvision torchaudio -c pytorch

### Training

To train the model, run this command:

```python
python run_blending_gradio.py
```
And then click the link
### Results
![蒙娜丽莎](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/result/monalisa.png "蒙娜丽莎")


## 2.Pix2Pix

This repository is the official implementation of [Pix2Pix](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/02_DIPwithPyTorch/Pix2Pix). 

>![epoch250](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/train_results/facades/epoch_250/result_3.png "epoch250_result3")

### Requirements

To install requirements:

```setup
numpy
opencv
pytorch
```
>pip install numpy  
conda install opencv  
conda install pytorch torchvision torchaudio -c pytorch

dataset:  
[facades](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz)   [cityscapes](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz) 

### Training

To train the model, run this command:

```train
bash download_facades_dataset.sh
python train.py
```
### Evaluation

evaluation is in the train.py

### Pre-trained Models

No Pre-trained Models

### Results

Our model achieves the following performance on :

### [facades](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz)

| Model name /epoch  | Train loss      | Val loss       |
| ------------------ |---------------- | -------------- |
| FCN_network/25     |     0.38        |      0.40      |
| FCN_network/250    |     0.16        |      0.39      |
| FCN_network/795    |     0.11        |      0.40      |

>![epoch25](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/facades/epoch_25/result_3.png "epoch25_result3")
![epoch250](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/facades/epoch_250/result_3.png "epoch250_result3")
![epoch795](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/facades/epoch_795/result_3.png "epoch795_result3")


### [cityscapes](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz)

| Model name /epoch  | Train loss      | Val loss       |
| ------------------ |---------------- | -------------- |
| FCN_network/25     |     0.1507        |      0.1641      |
| FCN_network/250    |     0.0768        |      0.1355      |
| FCN_network/795    |     0.0642        |      0.1367      |

>![epoch25](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/cityscapes/epoch_25/result_2.png "epoch25_result2")
![epoch250](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/cityscapes/epoch_250/result_2.png "epoch250_result2")
![epoch795](https://github.com/Dorispig/DIP/tree/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/cityscapes/epoch_795/result_2.png "epoch795_result2")
