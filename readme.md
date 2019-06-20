# SOSNet: Second Order Similarity Regularization for Local Descriptor Learning

SOSNet model implementation in PyTorch for the paper:

_SOSNet: Second Order Similarity Regularization for Local Descriptor Learning_  
Yurun Tian, Xin Yu, Bin Fan, Fuchao Wu, Huub Heijnen and Vassileios Balntas  
CVPR 2019

[[Project page]](https://research.scape.io/sosnet/) [[Paper]](https://arxiv.org/abs/1904.05019) [[Poster]](imgs/sosnet-poster.pdf) [[Slides]](imgs/sosnet-oral.pdf)


## Loading the SOSNet demo
The SOSNet model definition can be found in [sosnet_model.py](sosnet_model.py).

Below we show an example on how to load the network and run it on a minibatch.


```python 
# Init the 32x32 version of SOSNet
sosnet32 = sosnet_model.SOSNet32x32()
net_name = "liberty"
sosnet32.load_state_dict(torch.load(os.path.join('sosnet-weights',"sosnet-32x32-"+net_name+".pth")))
sosnet32.cuda().eval();

# create a random mini-batch of 100 items
x = torch.rand(100,1,32,32).cuda()
# forward feed it to the network
fx = sosnet32(x)
print(fx.size())
# fx.size() -> (100,128)
```


## Matching demo
We provide a demo on how to match two images in the [SOSNet-demo notebook](SOSNet-demo.ipynb).

## Citation
Please cite the following work if you use the code:

```
@InProceedings{sosnet2019cvpr,
author={Yurun Tian and Xin Yu and Bin Fan and Fuchao Wu and Huub Heijnen and Vassileios Balntas },
title = {SOSNet: Second Order Similarity Regularization for Local Descriptor Learning},
booktitle = {CVPR},
year = {2019}}
```
