---
# try also 'default' to start simple
theme: seriph
# theme: penguin
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://source.unsplash.com/collection/94734566/1920x1080
# apply any windi css classes to the current slide
class: 'text-center'
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

# 线性模型和梯度下降算法




---
layout: center
class: text-center
---

# 线性模型

---

# 1、模型定义
线性模型，即假定输入和输出的关系是线性的。通过属性的线性组合来进行预测,线性回归模型可以表示为：
$$
\hat{y}=x \ast \omega + b
$$

或者向量形式
$$
f(x)=\omega ^{T}x+b
$$

学习方式：根据数据集，确定各个属性的权重$\omega$和b

<img src="http://hellon.hellon.top/202303162227238.png" alt="image-20230316222724822" style="zoom: 15%; margin: 0 auto" />

线性模型形式简单，易于建模，并且有很好的可解释性。
---
layout:default
---

# 2、一元线性回归

为分析模型，先考虑数据集只有一个属性，线性模型为
$$
f(x_{i})=\omega \ast x_{i} + b
$$
模型求解：数据集中的样本点不可能都落在假定的线性模型上，每个样本点与预测值之间的误差可以用：

$$
loss = (\hat{y}-y)^2 = (x \ast \omega + b - y)^2
$$

来表示。

模型的好坏可以用均平方误差函数MSE(mean square error):

$$
cost = \frac{1}{N}\sum_{n=1}^{N}(\hat{y_{n}}-y_{n})^2
$$

来进行评价。

基于上述公式，**均方误差最小化**来进行模型求解的方法称为最小二乘法。最小二乘法就是试图找一条直线，是所有样本到直线上的欧氏距离之和最小。

---

<img src="http://hellon.hellon.top/202303161020393.png" alt="image-20230316102041828" style="zoom:33%;" />

---


# 2.1求解$\omega$和$b$

实例：给定数据集：三组x、y，以及一组x，通过三组x、y得到y与x的映射关系，进而对第四组x对应的y进行预测

<img src="http://hellon.hellon.top/202303162301939.png" alt="image-20230316230141640" style="zoom:28%; margin:0 auto" />

---

# 2.2穷举法

<img src="http://hellon.hellon.top/202303162255431.png" alt="image-20230316104235288" style="zoom:20%; margin:0 auto" />

通过计算机穷举$\omega$和b，模型会逐渐实现收敛，在训练集中误差会逐渐减少，训练出最优的收敛点。后面会使用训练轮数轮数来判定是否收敛成功。

---
class: px-20
---

# 2.3 计算$\omega$
<div grid="~ cols-2 gap-2" m="-t-2">

```python
import numpy as np
import matplotlib.pyplot as plt
# 引入numpy和matplotlab两个包

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
# 数据集

def forward(x):
    global w
    return x * w

def loss(x,y):
    # 损失函数
    y_pred = forward(x)
    return (y_pred - y) * (y_pred -y)

# 记录权重和权重的损失值
w_list = []
mse_list = []

```
<!-- ::right:: -->

```python
for w in np.arange(0.0,4.1,0.1):
    print('w=',w)
    l_sum = 0
    # zip函数，把多个列表按顺序依次拼接为多个元组，返回一个迭代器
    for x_val, y_val in zip(x_data,y_data):
        # 计算预测值和损失
        y_pred_val = forward(x_val)
        loss_val = loss(x_val,y_val)
        # 求损失和
        l_sum += loss_val
        print('\t',x_val,y_val,y_pred_val,loss_val)

    print('MSE=',l_sum / 3)
    # 把权重和损失值记录在列表中

    w_list.append(w)
    mse_list.append(l_sum / 3)

plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()

```
</div>

---

# 2.4穷举结果

输出结果如图所示，容易看出w=2是一个最小值，故模型为y=2*x.

<img src="http://hellon.hellon.top/202303162259983.png" alt="线性模式的结果" style="zoom:80%; margin: 0 auto" />
<h5 style="text-align: center;">图2.4 MSE-w结果示意图</h5>

---

# 3、多元回归模型

## 3.1多元线性模型

更一般的情形是：假设数据集D，样本由 d 个属性描述。此时我们试图学得
$$
f(x)=\omega ^{T}x+b
$$

这称为“多元线性回归”，同样多元线性回归模型也可以用最小二乘法来做：

$$
\hat{\omega^{*}} = arg(\hat{w})min(y-X\hat{\omega})^{T}(y-X\hat{\omega}))
$$

即使得函数$(y-X\hat{\omega})^{T}(y-X\hat{\omega})$取最小值时的参数$\hat{\omega^{*}}$是最优解。

上述函数可以通过穷举或者数学的方式算出最优解：对$\hat{\omega}$求导可得

$$
\frac{\partial E_{\hat{\omega}}}{\partial \hat{\omega}} = 2X^{T}(X\hat{\omega} - y)
$$

另上式为零，可以得到$\hat{\omega}$最优解的闭式解。

---
layout: two-cols
---


# 4、线性模型的拓展


## 4.1对数线性回归

之前的模型都是直接把预测值逼近y，如果数据集对应的输出标记是在指数尺度上变化，可以把输出标记的对数作为线性模型的逼近目标。
$$
\ln y = \omega^{T}x + b
$$
这个模型被称为“对数线性回归“，虽然形式上是线性模式，但实质上是求输入空间到输出空间的非线性函数映射

::right::

## 4.2广义线性模型  

更进一步，如果一个单调可微函数$g(x)$，令
$$
y = g^{-1}(\omega^{T}x + b)
$$
这个模型称为广义线性模型

参数估计也可以通过加权最小二乘法来进行
<img src="http://hellon.hellon.top/202303171058562.png" alt="image-20230317105823428" style="zoom:40%;" />

---
layout: center
class: text-center
---

# 梯度下降算法

---

# 1.1 梯度下降算法的引入

上一章我们在线性模型(也即最小二乘法)，对于参数w的范围选择是基于图像以及对于二次函数(MSE)的先验知识而确定的，但是面对一个复杂且难以通过肉眼观测最小值范围的损失函数时，模型参数的范围比较难确定。

- 穷举法：当自变量维度增加，搜索的区间范围就增加，很难搜索最小值
- 分治法：容易陷入**局部最优**，且容易错失最优解
<img src="https://pic4.zhimg.com/80/v2-60164b909629b52e00229610507cafcb_1440w.webp" alt="img" style="zoom:33%; margin:0 auto;" />

---

# 1.2 梯度下降

> 百度百科：梯度(gradient)的本意是一个向量（矢量），表示某一函数在该点处的方向导数沿着该方向取得最大值，即函数在该点处沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）。

<img src="http://hellon.hellon.top/202303161124398.png" alt="image-20230316112402511" style="zoom:15%;margin:0 auto;" />

梯度指向的是函数增长的方向，那么我们想要得到损失函数逐渐减少，那么我们每次就得让模型的参数往梯度相反的方向走，这样才能使得损失函数不断下降，以期得到最小值

---

# 2.1 梯度下降算法

核心概念：不断迭代每一步都向梯度下降的方向前进.梯度下降可以理解为你站在山的某处，想要下山，此时最快的下山方式就是你环顾四周，哪里最陡峭，朝哪里下山，一直执行这个策略，在第N个循环后，你就到达了山的最低.

算法步骤：
1. 给定待优化连续可微分的函数$J(\theta)$,步长,和一组样本值
2. 计算待优化函数的梯度
3. 更新迭代
4. 再次计算梯度
5. 向量模小于一定值或达到目标迭代次数

<img src="http://hellon.hellon.top/202303171217697.jpeg" alt="img" style="zoom:80%; margin-left:40%;margin-top:-30%" />

---

# 2.2 梯度下降迭代公式

<img src="http://hellon.hellon.top/202303161132711.png" alt="image-20230316113210177" style="zoom:20%;margin:0 auto;" />

---

# 3.1 批量梯度下降算法

前面所讨论中使用的梯度下降算法公式为
$$
\frac{\partial J(\theta )}{\partial \theta_{j}} =\frac{1}{n}\sum_{i=1}^{n}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}
$$

可以看出，计算机会每次从所有数据中计算梯度，然后求平均值，作为一次迭代的梯度，对于高维数据，计算量相当大，因此，把这种梯度下降算法称之为**批量梯度下降算法**。
---

# 3.2 随机梯度下降

随机梯度下降算法是利用批量梯度下降算法每次计算所有数据的缺点，**随机抽取某个数据**来计算梯度作为该次迭代的梯度，梯度计算公式：

$$
\frac{\partial J(\theta )}{\partial \theta_{j}} =(h-{\theta }(x^{(i)})-y^{(i)})x_{j}^{(i)}
$$

迭代公式：
$$
\theta = \theta - \alpha \cdot \triangledown _{\theta}J(\theta;x^{(i);}y^{(i)})
$$
由于随机选取某个点，省略了求和和求平均的过程，降低了计算复杂度，提升了计算速度，但由于随机选取的原因，存在较大的震荡性
---

# 3.3 小批量梯度下降算法

小批量梯度下降算法是综合了批量梯度下降算法和随机梯度下降算法的优缺点，随机选取样本中的一部分数据，梯度计算公式
$$
\frac{\partial J(\theta )}{\partial \theta_{j}} =\frac{1}{k}\sum_{i}^{i+k}(h_{\theta}(x^{(i)})-y^{(i)})x_{j}^{(i)}
$$

迭代公式：
$$
\theta = \theta - \alpha \cdot \triangledown _{\theta}J(\theta;x^{(i:i+k);}y^{(i;I+k)})
$$

---

# 4 代码及效果

### 批量梯度下降算法
```python
def cost(xs,ys):# 计算MSE
    cost = 0
    for x,y in zip(xs,ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost /len(xs)
def gradient(xs,ys):# 梯度函数,算每个样本点的平均梯度
    grad = 0
    for x,y in zip(xs,ys):
        grad += 2 * x * (x * w - y)
    return grad /len(xs)
#更新权重 新权重 = 旧权重 - 学习率 * 梯度
for epoch in range(100):
    cost_val = cost(x_data,y_data) #先算成本
    grad_val = gradient(x_data,y_data) #训练过程
    w -= a * grad_val
    epoch_list.append(epoch)
    w_list.append(w)
    cost_list.append(cost_val)
    # 输出训练日志
```
---

### 随机批次梯度下降算法
```python
def loss(x,y):#定义损失函数
    y_pred = forward(x)
    return (y_pred-y)**2
#梯度函数
def gradient(x,y):
    return 2*x*(x*w-y)

print('Predict (before training)',4,forward(4))
epoch_list=[]
loss_list=[]
#epoch为迭代次数,共训练100轮次
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        #梯度
        grad = gradient(x,y)
        #若在某一点梯度为0，则停止迭代，对应的w值即稳定不变
        w -= 0.01 * grad
        print("\tgrad:",x,y,grad)
        l =loss(x,y)
        epoch_list.append(epoch)
        loss_list.append(l)
    print("progress:",epoch,"w=",w,"loss=",l)
```
---
layout: two-cols
---

### 批量梯度下降算法

<img src="http://hellon.hellon.top/202303171318626.png" alt="image-20230317131841584" style="zoom:50%;margin:0 auto;" />
::right::

### 随机批次梯度下降算法


<img src="http://hellon.hellon.top/202303171319492.png" alt="image-20230317131911459" style="zoom:50%;margin:0 auto;" />

