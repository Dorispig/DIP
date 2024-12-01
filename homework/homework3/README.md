# Assignment 3 - play with GANs

This repository is the official implementation of [Assignment 3 - play with GANs](https://github.com/Dorispig/DIP/tree/main/homework/homework3).  
Include [cGAN](https://github.com/Dorispig/DIP/tree/main/homework/homework3/my_conditional_gan) and [DragGAN](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/02_DIPwithPyTorch/Pix2Pix)
## 1.cGAN
This repository is the official implementation of [cGAN](https://github.com/Dorispig/DIP/tree/main/homework/homework3/my_conditional_gan). 

### Requirements

To install requirements:

```setup
numpy
os
cv2
pytorch
```

>pip install numpy  
pip install os
conda install python-opencv  
conda install pytorch torchvision torchaudio -c pytorch

### Training

To train the model, run this command:

```python
bash download_facades_dataset.sh
python train.py
```
### Results

#### [cityscapes](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/cityscapes.tar.gz)

| Model name /epoch  | Train loss(generate loss/discrimination loss)      | Val loss(generate loss/discrimination loss)       |
| ------------------ |---------------- | -------------- |
| cGAN_network/400     |     0.933/1.300        |      7.817/4.236      |
| cGAN_network/800    |     0.731/1.941        |      7.800/4.115      |
| cGAN_network/1595    |     0.861/1.575        |      7.790/4.126      |

>![epoch400](https://raw.githubusercontent.com/Dorispig/DIP/refs/heads/main/homework/homework3/my_conditional_gan/val_results/cityscapes/epoch_400/result_4.png "epoch400_result4")
![epoch800](https://raw.githubusercontent.com/Dorispig/DIP/refs/heads/main/homework/homework3/my_conditional_gan/val_results/cityscapes/epoch_800/result_4.png "epoch800_result4")
![epoch1595](https://raw.githubusercontent.com/Dorispig/DIP/refs/heads/main/homework/homework3/my_conditional_gan/val_results/cityscapes/epoch_1595/result_4.png "epoch1595_result4")









## 2.DragGAN

This repository is the official implementation of [DragGAN](https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/02_DIPwithPyTorch/Pix2Pix). 

### Results

Our model achieves the following performance on :

### [facades](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz)

>![epoch25](https://raw.githubusercontent.com/Dorispig/DIP/refs/heads/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/facades/epoch_25/result_3.png "epoch25_result3")
![epoch250](https://raw.githubusercontent.com/Dorispig/DIP/refs/heads/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/facades/epoch_250/result_3.png "epoch250_result3")
![epoch795](https://raw.githubusercontent.com/Dorispig/DIP/refs/heads/main/homework/homework2/02_DIPwithPyTorch/Pix2Pix/val_results/facades/epoch_795/result_3.png "epoch795_result3")

