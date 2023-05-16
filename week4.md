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

# 构建数据加载器、多分类问题和模型的导入导出

#### 江西财经大学 | 易亚伟 | 2202291160@stu.jxufe.edu.cn

---
layout: center
class: text-center
---

# 使用Dataset和DataLoader构建加载器

---

# 加载器引入

对所有数据统一求梯度均值后进行权重的优化，效果不好；而对于每个数据逐个求均值效果比较好，但是由于存在数据相关，影响程序运行的效率，故采用**batch方法**，将若干数据划分为一个batch后逐个求梯度并优化。

* Epoch：每一次进行一次前向传播和反向传播，都属于一个epoch

* Batch-Size：指的是每一批放在一起计算梯度的数据数量
* Interation：表示共需要处理多少批次数量的数据。因此数据集数量 =BatchSize∗Interation

# Dataset和DataLoader

Dataset和DataLoader都是PyTorch中封装好的类。


```python
from torch.utils.data import Dataset #抽象类，只能继承，不能实例化
from torch.utils.data import  DataLoader  #可以实例化
```

Dataset类的作用是加载数据集以及对数据进行索引

DataLoader则提供了构造Mini-Batch的方法,每次循环，都提供不同batch的数据

---

# 构建Dataset

通过继承Dataset可以构建我们的数据加载类，DataSet是一个抽象类(不能实例化，只能进行继承)
* __getitem__是提供索引的魔法方法
* __len__是返回数据集的长度的魔法方法


```python{all|2-8|9-10|11-12|all}
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype = np.float32)
        # 除了最后一行数据均放入x_data矩阵中
        self.len = xy.shape[0];
        self.x_data = torch.from_numpy(xy[:, :-1])
        # 只把最后一行放入y_data向量
        self.y_data = torch.from_numpy(xy[:, [-1]])
    def __getitem__(self, index):  #可以拿出来数据
        return self.x_data[index],self.y_data[index]
    def __len__(self): #拿出数据长度
        return self.len
```

加载数据的两种方式
1. 直接从文件读到内存
2. 指定数据所在的文件路径，按需加载

---

# 构建DataLoader

DataLoader是数据加载器，起到打乱数据分批次的作用，返回可迭代的loader。


<div grid="~ cols-2 gap-4">
<div>

## 代码
```python
# 参数num_worker判断是否并行加载，表示进程个数，在windows下改为0就可以了
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=0)
```
参数的含义：

* dataset：数据集
* batch_size：一个batch包含多少个数据
* shffule：bool类型，是否打乱数据顺序，True为打乱，False为不打乱；
* num_workers指的是并行处理进程的个数

</div>
<div>

## 图示

<img src="http://hellon.hellon.top/202304050953473.png" alt="image-20230405095351302" style="zoom:67%;" />

</div>
</div>

windows报错：在Windows下并行处理会报错，需要在程序前加入Python一个入口的魔法函数，并且num_worker在windows下改为0

<style>
img{
  display:block;
  margin:0 auto;
  margin-top:20px;
}
</style>

---
layout: center
class: text-center
---

# softmax函数解决多分类问题

---

# 多分类问题的引入

之前我们解决的都是二分类问题，对于多分类问题，我们之前的模型无法解决，Softmax分类器就是为了解决多分类问题的神经网络的组件

<img src="http://hellon.hellon.top/202304051821999.png" alt="image-20230405182103806" style="zoom:40%;" />

---

# MNIST数据集

MNIST数据集是一个手写数字识别数据集，对该数据集进行分类的任务是典型的多分类问题，因为输出结果的类别有0-9共10种

<div grid="~ cols-2 gap-4">
<div>

<img src="http://hellon.hellon.top/202303161627097.webp" alt="img" style="zoom:55%;margin:0 auto;" />


</div>
<div>

<img src="https://pic1.zhimg.com/80/v2-92ffcbd5e901d5d462b5cfbb9a4c4bf4_1440w.webp" alt="img" style="zoom:50%;margin:0 auto" />

</div>
</div>

如果采用之前的sigmoid函数处理每个类别的概率，各数据之间不能够形成互斥的分布。

多分类最终输出的结果应满足以下条件：
1. 每一类型概率应在0-1之间
2. 所有分类的概率和为1

---

# 使用softmax解决多分类问题

为了解决多分类问题，我们引入Softmax分布

Softmax层的概率计算表示为
$$
p(y=i)=\frac{e^{Z_j}}{\sum^{K-1}_{J=0}e^{Z_j}},i \in \left\{ 0,...,K-1 \right\}
$$
softmax函数的性质满足了多分类的条件：
* 大于0由指数函数性质所决定
* 采用归一化，使得各概率之和=1

<img src="http://hellon.hellon.top/202304051149690.png" alt="image-20230405114936568" style="zoom: 20%;margin:0 auto" />

---

# 多分类问题的损失函数

对于多分类问题的损失函数，也进行了修改，从该损失函数的形式看出，采用独热码的编码方式:只考虑正确类别的数据的损失，因为其它类别的数据均为0

<img src="http://hellon.hellon.top/202304051153943.png" alt="image-20230405115346808" style="zoom:80%;" />

---

# 交叉熵损失函数CrossEntropyLossr

<img src="http://hellon.hellon.top/202304051156004.png" alt="image-20230405115613869" style="zoom:28%;margin:0 auto" />

CrossEntropyLossr交叉熵损失函数是整体计算的。从最后一层的线性层开始计算到最后，里面包含了softmax

---

# 构建多分类神经网络分析MNIST数据集

## 构建加载器

```python
batch_size = 64
transform = transforms.Compose(# 把中括号中的张量
    [
        transforms.ToTensor(),  # 转换成图像张量
        transforms.Normalize((0.1307,), (0.3081,))
    #     归一化，均值和标准差，是数据满足0-1分布
    ]
)
# 读取数据先调用transform转换一下
tarin_dataset = datasets.MNIST(root="../dataset/mnist",
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(tarin_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root="../dataset/mnist",
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)
```

---

## 构建多分类神经网络

```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64, 10)
    def forward(self,x):
        x = x.view(-1,784)
        # 使用relu激活
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        # 第5层直接返回
        return self.l5(x)

model = Net()
# 使用交叉熵计算损失
criterion = torch.nn.CrossEntropyLoss()
# 使用带冲量的SGD作为优化器
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

```

---

## 封装训练函数

```python
def train(epoch):
    # 把单论循环封装成函数
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        # 优化器先清零
        optimizer.zero_grad()

        # 前馈，反馈和更新
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if(batch_idx % 300 == 299):
            print('[%d,%5d] loss:%.3f' % (epoch+1,batch_idx+1,running_loss/300))
            running_loss = 0.0
```

---

## 封装测试函数

```python
def test():
    correct = 0
    total = 0
    # 在这个部分中的代码不会计算梯度
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = model(images)
            # 沿着维度1，即行维度，找最大值  max返回最大值和最大值下标
            _,predicted = torch.max(outputs.data,dim=1)
            # total先加上下标的种类
            total += labels.size(0)
            # 比较运算，求和
            correct += (predicted == labels).sum().item()
    print('测试集中有效率：%d %%' % (100* correct/total))

```

## 训练
```python
if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

---
layout: center
class: text-center
---

# pytorch模型的导入与导出

---

## 导出模型
使用`model.save()`函数导出训练好的模型

```python
    model_save_path = os.path.join("", 'model.pt')
    torch.save(model.state_dict(), model_save_path)
```

## 导入模型

使用`model.load*()`函数加载模型

```python
    model_save_path = os.path.join("", 'model.pt')
    loaded_paras = torch.load(model_save_path)
    model.load_state_dict(loaded_paras)  # 用本地已有模型来重新初始化网络权重参数
    model.eval()  # 注意不要忘记
```