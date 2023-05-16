---
# try also 'default' to start simple
theme: seriph
# theme: penguin
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: "text-center"
# https://sli.dev/custom/highlighters.html
highlighter: shiki
# show line numbers in code blocks
lineNumbers: true
# some information about the slides, markdown enabled
info: |
  ## Slidev Starter Template
  Presentation slides for developers.

  Learn more at [Sli.dev](https://sli.dev)
# persist drawings in exports and build
drawings:
  persist: false
# page transition
transition: slide-left
# use UnoCSS
css: unocss
download: true
---

# 卷积神经网络与循环神经网络

#### 江西财经大学 | 易亚伟 | 2202291160@stu.jxufe.edu.cn

---
layout: center
class: text-center
---

# 卷积神经网络(CNN)

## Convolutional Neural Network
---

# 卷积

从数学上讲，卷积就是一种运算,其定义为

连续形式：
$$
(f*g)(n)=\int_{ -\infty }^{ \infty }f(\tau)g(n-\tau)d\tau
$$
离散形式：
$$
(f*g)(n)=\sum_{\tau=-\infty}^{\infty}f(\tau)g(n-\tau)
$$

卷积的**物理意义**:系统某一时刻的输出是由多个输入共同作用（**加权叠加**）的结果

卷积描述了一个动态过程，表达了系统不断衰减同时又不断受到激励的综合结果

卷积的应用：在统计学，物理学，电子信号处理，计算机科学等领域中，卷积起到了至关重要的作用。

因为面对一些复杂情况时，作为一种强有力的处理方法，卷积给出了简单却有效的输出。对于机器学习领域，尤其是深度学习，最著名的**CNN卷积神经网络(Convolutional Neural Network, CNN)**，在图像领域取得了非常好的实际效果。
---

# 卷积神经网络 CNN

定义：包含卷积运算的神经网络。卷积层负责提取特征，采样层负责特征选择，全连接层负责分类

![image-20230419115252955](http://hellon.hellon.top/202304191153140.png)

与之前的神经网络相比，CNN通过引入卷积层来提取输入信息的特征，降低运算量

<style>
img{
  display:block;
  margin:0 auto;
  margin-top:20px;
}
</style>
---

# 卷积(convolution)层 

将输入与卷积核进行卷积运算，得到特征值

![image-20230409141330774](http://hellon.hellon.top/202304091413942.png)
卷积核作用的结果是修改图像大小，并提高通道数，通过卷积运算，提取出输入值的某些指定特征

通过填充padding，在输入层周围填充一圈0，可以保持图像的大小不发生改变

```python
conv_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=2)
```

pytorch中可以使用上面的代码，构建一层卷积
<style>
img{
  display:block;
  margin:0 auto;
  margin-top:20px;
}
</style>
---

# 池化(pooling)层

池化（Pooling）也叫做下采样（subsampling），用一个像素代替原图上邻近的若干像素，在保留feature map特征的同时压缩其大小。

池化的作用：

* 防止数据爆炸，**节省运算量**和运算时间。
* 用大而化之的方法防止过拟合、过学习。

常见的两种池化方式：
* 最大值池化:在一组数据中求最大值，只保留最大值
* 平均值池化：在一组数据中求平均值，只保留平均值

![image-20230419121145227](http://hellon.hellon.top/202304191211303.png)
<style>
img{
  display:block;
  margin-left:1500px;
  margin-top:-400px;
  zoom:30%;

}
</style>
---

# 全连接层

全连接层和之前的一样，将卷积层的输出传入神经网络，通过softmax分类器进行分类

<img src="https://pic1.zhimg.com/80/v2-92ffcbd5e901d5d462b5cfbb9a4c4bf4_1440w.webp" alt="img" style="zoom:80%;margin:0 auto" />

---

# GoogLeNet结构

**GoogLeNet**：是一种很常用的基础架构。

![image-20230409154012668](http://hellon.hellon.top/202304091540916.png)

可以看出网络存在大量重复的结构，因此考虑都可以对其进行封装，以减少代码的冗余
<style>
img{
  display:block;
  margin:0 auto;
  margin-top:20px;
}
</style>
---

# 构建Inception模块

Inception模块是对一系列基本组件(包括卷积核、池化、softmax以及拼接结构)的封装。

![image-20230409160201779](http://hellon.hellon.top/202304091602993.png)

核心思想:不知道卷积核多大好，就把各种卷积核混在一起给一个权重；引入1*1的卷积核可以提高运算速度
<style>
img{
  display:block;
  margin:0 auto;
  margin-top:20px;
  zoom:20%;
}
</style>
---

# Inception模块代码


<div grid="~ cols-2 gap-4">
<div>


```python
#InceptionA module
class InceptionA(nn.Module):
    def __init__(self,in_channels):

        super(InceptionA, self).__init__()
        #1x1的卷积分支
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        #5x5卷积分支上的一个1x1卷积核以及5x5卷积核
        self.branch5x5_1 = nn.Conv2d(in_channels, 16,kernel_size=1)
        #为保证二维数据大小不变，5x5的卷积核对应的padding=2
        self.branch5x5_2 = nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = nn.Conv2d(24,24,kernel_size=3,padding=1)

        #平均池化层分支上的1x1卷积操作
        self.branch_pool = nn.Conv2d(in_channels,24,kernel_size=1)

```

</div>
<div>

```python
    def forward(self,x):
        #单个1x1卷积核分支的1x1卷积操作
        branch1x1 = self.branch1x1(x)

        #5x5卷积核分支的卷积操作
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        #3x3卷积核分支的卷积操作
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        #平均池化层分支上的操作，先卷积再池化
        branch_pool = F.avg_pool2d(x, kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)

        #进行拼接操作,先按照列表形式按顺序写入分支，再使用torch的cat方法沿通道方向(dim=1)进行拼接
        outputs = [branch1x1,branch5x5,branch3x3,branch_pool]
        return torch.cat(outputs,dim=1)
```
</div>
</div>

---

# 梯度消失与解决方案

神经网络的层级非常深层的情况下，Loss关于各网络参数的梯度都逼近0，相乘就更加逼近0。在反向传播的算法下，距离输出层越远，梯度会趋近于0，以至出现梯度消失的情况

# 残差神经网络结构
将某一参数x和经过网络层相加得到损失函数相加，此时关于该参数的导数就>1，若Loss函数对所有参数的导数都大于1，相乘的结果就不会逼近0，就解决了梯度消失的问题

![image-20230412103643025](http://hellon.hellon.top/202304121036280.png)


<style>
img{
  display:block;
  margin:0 auto;
  margin-top:20px;
  zoom:15%;
}
</style>

---
layout: center
class: text-center
---

# 循环神经网络(RNN)

## Recurrent Neural Network
---

# 循环神经网络(RNN)

循环神经网络是一种具有记忆功能的神经网络，适合序列数据的建模。即一个序列当前的输出与前面的输出也有关。适用于股市，天气预测，文本生成等具有序列关系应用场景。

![img](https://pic4.zhimg.com/v2-3884f344d71e92d70ec3c44d2795141f_r.jpg)

<style>
img{
  display:block;
  margin:0 auto;
  margin-top:20px;
  zoom:50%;
}
</style>
