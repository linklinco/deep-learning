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

# 反向传播和基于PyTorch的线性回归

#### 江西财经大学 | 易亚伟 | 2202291160@stu.jxufe.edu.cn

---
layout: center
class: text-center
---

# 反向传播

---

# 反向传播的引入

之前研究了线性模型，并且引入了梯度下降算法，使得模型参数w能自动逼近全局最优解w=2，这里w*(Input)可以看作一个简单的神经元。

<img src="http://hellon.hellon.top/202303161627738.webp" alt="img" style="zoom: 50%;" />

但是，线性模型参数比较少，仅有1-2个，当模型参数增加时，所需进行地偏导计算就急剧增加。

<style>
img{
  display:block;
  margin:0 auto;
}
</style>

---

<img src="http://hellon.hellon.top/202303161627770.webp" alt="img" style="zoom:80%;" />

如图4.2所示，输入层 x¯={x1,x2,x3,x4,x5} ，为5x1维向量，而中间隐藏层有4层，输出层1层 y¯={y1,y2,y3,y4,y5} ,为5x1维向量，对于中间隐藏层来说只要有第1层，后续可依次类推，隐藏层1的神经元可表示为 h1¯={h1(1),h1(2),h1(3),h1(4),h1(5),h1(6)}，其中， h1(1)=∑i=15ω1ixi+bi1 其它以此类推，每个隐藏层的神经元需要进行6次运算，6个神经元共需要进行30次加和运算，同时也包含30=5x6个 ωij 模型参数，其他层之间交互以此类推。

计算得到图中神经网络需要193个ωij 的模型参数，最后的损失函数需要对每个线性模型的参数求梯度，常规思路就是将表达式依次累乘后累加得到表达式后逐个求导，每个表达式的参数未必相同，对应的计算量就十分庞大了，

<style>
img{
  display:block;
  margin:0 auto;
}
</style>

---

#  反向传播的核心

反向传播本质上就是链式法则。

<img src="http://hellon.hellon.top/202303161627161.webp" alt="img" style="zoom:40%;margin:0 auto;" />

在正向传播的过程中，计算每个当前神经元损失关于神经元模型参数的偏导，利用图的思想，传播过程中进行梯度的计算并存储在相应的单元中

后续进行反向传播，依次按照链式法则和存储的梯度向前递推，就可以得到最后的损失函数关于每个模型参数的梯度。

---

# 实例


下图表示了一个神经元进行反向传播的具体实例，首先前向传播，逐个参数进行计算，并以此求梯度，后面损失函数关于此神经元的梯度为5，依次向前传播，依据链式法则，可计算得到损失函数Loss关于x的梯度为15，关于模型参数w的梯度为10，使用反向传播能够极大地简化计算。

<img src="http://hellon.hellon.top/202303161627953.webp" alt="img" style="zoom:80%;margin:0 auto;" />

---

# 损失函数关于某一神经元的模型参数计算

1. 首先按照前向传播方向，经过一个神经元进行对应的计算，并计算梯度，直至传播至输出端的神经元；
2. 开始进行反向传播，依据前面每个神经元关于该神经元模型参数的梯度向前累乘，直至最初指定的那一个神经元，由此即可完成关于此神经元的反向传播
3. 其他的神经元也是按照此步骤
4. 由此整个神经网络的损失函数关于每个神经元的若干模型参数即可计算出来
<img src="http://hellon.hellon.top/202303161627967.webp" alt="img" style="zoom:60%;margin:0 auto;" />
得到了损失函数关于参数的梯度，后面就可以对参数进行优化。

---
layout: image-right
image: http://hellon.hellon.top/202303221317384.png
---

# pytorch 的基本使用

- tensor：张量，是pytorch中最重要的数据结构，包含date和梯度

利用损失函数对线性模型w的表达式进行手动求梯度表达式写入代码中进行梯度的计算，当神经元数量增加/参数增加，利用手动计算就十分的繁琐.

而pytorch张量能够提供自动求导(本质是反向传播，tensor.backward())的功能，还是以单神经元网络的线性模型为例，利用pytorch的自动求梯度的功能，进行代码的编写


---

# 代码对比

forward和loss函数都没有改变
<div grid="~ cols-2 gap-4">
<div>

### 之前的代码

```python{all|12-14|all}
w = 1.0  #猜测的初始权重
a = 0.01 #学习率

# 梯度函数,算每个样本点的平均梯度
def gradient(xs,ys):
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x * (x * w - y)
    return grad /len(xs)
#更新权重 新权重 = 旧权重 - 学习率 * 梯度
for epoch in range(100):
    lost_val = lost(x_data,y_data) #先算成本
    grad_val = gradient(x_data,y_data) #训练过程
    w -= a * grad_val

    # 输出训练日志
    print('Epoch:',epoch,'w=',w,'梯度值：',grad_val,'cost=',cost_val)
print('Predict (after training)',4,forward(4))

```

</div>
<div>

### 使用tensor

```python{all|9-16|all}
import torch

w = torch.Tensor([1.0])
w.requires_grad = True
#forward 和 loss 函数都没变

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        loss_val = loss(x,y)
        loss_val.backward() 
        #在计算图中需要计算的梯度都算出来 ，进行反向传播后会自动释放计算图
        print('\tgrad:',x,y,w.grad.item())#把梯度中的数值直接拿出来
        # 使用data计算不会建立计算图，
        w.data = w.data - 0.01 * w.grad.data #权重更新直接使用grad.data

        w.grad.data.zero_() #用完需要对梯度清零
    print("progress:",epoch,loss_val.item())
print('predict (after training)',4,forward(4).item())



```

</div>
</div>

---
layout: center
class: text-center
---

# 使用Pytorch进行线性回归

---

# Pytorch训练模型的基本步骤

PyTorch提供了`nn.Module`，通过继承这个类，可以调用其封装好的一系列方法，能够降低编程的难度。

PyTorch的基本步骤包括：
> 1. 准备数据集
> 2. 使用Class定义模型(继承自`nn.Module`),计算$\hat{y}$
> 3. 构造损失函数和优化器(使用PyTorch的API)
> 4. 训练周期(由epoch决定，主要包含forward、backward以及update)

按照以上步骤展开基于的线性模型的构造。

---

### 5.1准备数据集

还是采用前面的x_data，y_data三组数据作为数据集，但是与前面不同的是，数据集也修改为Tensor变量，代码如下所示.y_pred为3x1向量，x也为3x1向量，ω与x向量进行数乘，但b不是3x1向量，而是一个标量，需要对其进行广播，扩展为3x1向量，数值均为b。

```python
import torch

#Preparing dataset，Mini-Batch
x_data =torch.Tensor([[1.0],[2.0],[3.0]])
y_data =torch.Tensor([[2.0],[4.0],[6.0]])
```

上面是针对单个计算块的神经网络，而针对多个计算块组成的神经网络，对应的 ω 应该就是矩阵了，z是3x1向量，x是4x1向量，b与z同维，而ω符合矩阵的基本计算准则，应该是3x4矩阵。

<img src="https://pic4.zhimg.com/80/v2-1db14ab6fe5c8143dab8d4f1b74e830b_1440w.webp" alt="img" style="zoom:80%;margin:0 auto;" />
---

### 5.2 设计模型

设计模型可以通过封装一个类来定义，在`pytorch`，父类torch.nn.Module提供了大量的方法。我们可以通过继承这个类来使用。如下所示，首先定义线性模型类；然后初始化，继承Module父类。

接下来定义了一个Linear模型变量，通过torch.nn.Linear(1,1)返回，对于torch.nn.Linear第一个变量in_features表示输入变量的维度，第二个变量out_features表示输出变量的维度，而第三个变量表示偏置的开关，是一个bool变量，Ture表示需要偏置量，False表示不需要偏置量，默认为True。

```python{all|2,5|7|9-12|all}
#设计模型
class LinearModel(torch.nn.Module):
    def __init__(self): #初始化
        #继承Module父类必须进行这一步
        super(LinearModel, self).__init__()
        #定义了Linear模型中一个变量，输入为1维张量x，输出为1维张量y，变量包含w和b
        self.linear = torch.nn.Linear(1,1)
    #reride(重载)forward函数(foward在__call__()中有定义)
    def forward(self,x):
        #self.linear是torch.nn.Linear(1,1)的返回值,通过赋值，self.linear是一个可调用对象，本质看作self.linear.__call__
        y_pred =self.linear(x)
        return y_pred
#模型实例化
model = LinearModel()
```
最后重载foward函数，利用可调用对象self.linear(x)实现预测值的计算，最后返回预测值。
---

## 5.3 损失函数和优化器的构建

均方差损失(MSE)函数和随机梯度下降(SGD)算法的优化器都在PyTorch中有封装好的程序，直接进行调用即可。如下所示，首先构造了损失函数，调用了torch.nn.MSELoss，然后调用了优化器SGD，学习率为0.01.

```python{all|4|7|all}
#构造损失函数和优化器
#均方差的误差函数(一个类)，已经在torch.nn模块中进行了定义,size_average表示是否需要降维处理
#检查model中所有成员，若成员中有权重(模型参数)，则将训练结果添加到列表中
criterion = torch.nn.MSELoss(size_average=False)

#优化器在torch.optim模块中也有相应的定义，学习率为0.01，这里采用SGD(随机梯度下降算法)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
```

MSE函数即为之前的方差和，而对于torch.optim.SGD，官方文档说明下图所示，其中参数是需要优化的，通过迭代学习实现优化。

![img](https://pic4.zhimg.com/80/v2-201131898de4bb18f1c38bb360ab0ce7_1440w.webp)


至此，构造损失函数和优化都已经完成，最后进行训练。

<style>
img{
  display:block;
  margin:0 auto;
}
</style>
---

## 5.4 训练

接着进行训练，如下所示。和前几章不同，这里调用了大量torch中的方法，稍微复杂，但是可扩展性强，对于较为复杂的模型采用这种方式表达反而会更为简便。

```python{all|5,6,10-12|all}
epoch_list =[]
loss_list=[]
for epoch in range(100):
    #进行前向传播，实现预测
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)#以及MSE计算损失函数 
    print(epoch,loss.item())#输出轮次和损失,其中张量loss会自动调用其__str__()方法，实现输出
    epoch_list.append(epoch)
    loss_list.append(loss.item())
    optimizer.zero_grad()#将各张量的梯度归零
    loss.backward()#对损失函数进行反向传播，自动梯度计算
    optimizer.step()#对各模型参数进行更新
#线性模型包含两个参量w和b，输出最终迭代结果
print('w = ',model.linear.weight.item(),'b = ',model.linear.bias.item())
#基于线性模型训练的参数对测试集进行测试
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data)
plt.plot(epoch_list,loss_list)
plt.show()
```