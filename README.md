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
