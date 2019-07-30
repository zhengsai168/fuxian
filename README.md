# fuxian
fuxian paper and algorithm
可视化: visdom
```
python -m visdom.server
```
然后根据提示在浏览器中打开localhost:8097

![test_pic](https://github.com/zhengsai168/fuxian/blob/master/picture/input.png)

## DCGAN(生成动漫头像)

![DCGAN_pic](https://github.com/zhengsai168/fuxian/blob/master/picture/DCGAN.png)

## Fast Neural Style(图片风格迁移)

#### 风格迁移网络Transfomer_Net结构
![transfomer_net_pic](https://github.com/zhengsai168/fuxian/blob/master/picture/transformer_net.png)

#### 网络整体结构(VGG部分不更新梯度，只用来算Loss)
![loss_net](https://github.com/zhengsai168/fuxian/blob/master/picture/Fast_Neural_Style.png)

## CharRNN （生成唐诗）

#### 训练
![charRNN_net](https://github.com/zhengsai168/fuxian/blob/master/picture/CharRNN%20.png)

#### 生成
将开头的一句话或一个字输入网络，然后将输出的词（概率最大的词）当作输入直到达到maxlen或<EOP>结束标志。


## VAE （mnist数据集）

![vae_pic](https://github.com/zhengsai168/fuxian/blob/master/VAE/VAE.png)

首先，拟合出x的均值和方差，以这一均值和方差的正态分布采样，得到噪声，通过decoder生成出x。
loss需要有原x和生成出x的一个距离函数的度量，代码里用了binary_crossentropy。
但是如果仅仅如此是不够的，这样会使整个模型退化为普通的AutoEncoder，因为为了最小化loss，模型会让拟合出的方差的尽可能小从而稳定噪声，直到没有噪声，
所以我们需要让这个正态分布逼近一个标准正态分布，最简单的就是加一个loss，这里这个loss选择了N(0,1)和N(a,b)的KL散度作为loss。
更多细节参考苏神的博客：
https://spaces.ac.cn/archives/5253

