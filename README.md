# learn-python

## 1. 基本内容
### 文本读入与分词

    with open() as file:
    # 一般用file而不用ite

file是一个iterator，每一步都储存着一个string（一行的内容）

对于csv文件，这个string是用,分开的，分词的话，需要用string的split(',')函数，把字符串转化为较短的list

    g = open()
    reviews = list(map(lambda x:x[:-1],g)) # g.readlines()，python2.5之后g是iterator
    g.close()

reviews是个list，每个元素是读入的行

如果数据很大，可以分成多个batch，每个batch有多个sample，分批读入

### iterable, iterator或者generator代替list

An iterable is an object that has an __iter__ method which returns an iterator or a generator, or which defines a __getitem__ method that can take sequential indexes starting from zero (and raises an IndexError when the indexes are no longer valid). So an iterable is an object that you can get an iterator from.

An iterator is an object with a next (Python 2) or __next__ (Python 3) method.

iterator是一个类，有next()等方法，generator是一个特殊的函数，也能用next()函数调用，能用在for循环中，函数返回值不断被调用，但用在for循环中时，看上去和iterator是一样的

对于iterable，由于有iter method，所以可以在多个for循环中使用；而对于iterator和generator，只能被一个for循环使用一次，就没有内容了。

当list很长时，应该用iterator或者generator代替list，节省内存空间

iterator或者generator有三种产生办法，一种是在函数中用yield语句产生generator，另一种是在list comprehension中使用括号产生generator，第三种是用某个类的构造函数或者其他函数返回一个iterator（一个具有iter()和next()的类）.

(i for i in list)，这个语句产生的是generator，和在函数中使用for循环和yield语句产生的generator完全一样，只是换一种简洁的写法

generator的好处是非常简单，可以说，只实现iter和next这两个步骤，而iterator是某些类的统称，这些类包含iter和next这两个方法，但是，不只是这两个方法，可能还有很多method和attribute，可能非常占空间。

### 对for循环的替代及与for循环的比较

generator的生成、list comprehension、dict comprehension

Python函数式编程之map()、filter()

map()带两类参数，第一类参数是个函数，第二类参数是一个或者多个iterator(比如list、tuple、字符串等)，返回的也是一个iterator（有可能只是个generator），具体值是函数作用于iterator的结果

[python中，for循环，map函数，list comprehension列表推导的效率比较](https://www.cnblogs.com/superxuezhazha/p/5714970.html)

filter带两类参数，第一类参数是个函数，第二类参数是一个或者多个iterator(比如list、tuple、字符串等)，返回的也是一个iterator，具体值是函数作用于iterator后结果为True的值

### 单列频数统计与Counter from collections

collections是python的built-in module

Couter class返回的是一个dict-like project，key可以是需要统计频数的词，value是具体的频数

    counts = Counter()
    for i in generator: # generator or list of words
        counts[i]+=1
        
### dict相对于list的以空间换时间

redis list的lrange得到的就是二维list

从二维list当中得到某两列的对应关系：

{item[0]:item[1] for item in list}

把二维list的某一列替换为与之对应的另一列：

list_a与list_b对应

the_dict = {i:j for i,j in zip(list_a, list_b)}

for item in list:
    item[0]= the_dict[item[0]]
        
### 多列数据中一列A中多个类别在另一列B中的词频统计

大循环是B列中的list或者generator、iterator，A列与B列共享index

小循环是A列中的set，然后用if语句判断A列情况(list或者generator、iterator)，操作B列数据，比如B列数据的Counter

    for i in range(len(list_B)):
        for j in set_A:
            if list_A[i]==j:
                counts[list_B[i]]+=1

### 把2个list合并成一个list，然后用在for循环中

zip()

    a=[1, 2, 3]
    b=[4, 5, 6]
    c=zip(a,b)
    # c=[(1,4),(2,5),(3,60)]
    
map()

    d=map(None,a,b)
    
### 打印格式的控制

    print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))
    # {}中的:表示格式控制，>10表示10个空格，.4或者.6表示四位或者六位小数
    
### Dictrionary-based string formatting

    params = {'server': 'mpilgrm', 'database': 'master'}
    "%(database)s" % params
    # master

## 2. 第三方模块

### pandas
* 大数据文件的读入，所需时间的估计与比较

### sklearn (scikit-learn)
* 多列数据中一列A中多个类别在另一列B中的词频统计

### numpy
    import numpy as np
* one-hot encoding

代码：

    x = [0, 2, 9]
    n_classes = 10
    np.eye(n_classes)[x]
    # 每一个类用不同的行向量表示
    
除了用numpy，也可以用sklearn中的LabelBinarizer or OneHotEncoder。
    
### tensorflow
* Difference between tf.placeholder and tf.Variable
In short, you use tf.Variable for trainable variables such as weithts and biases for your model.
tf.placeholder is used to feed actual training samples.

* tensor的definition
tf.placeholder（tensor）的argument用到list参数时（表示这个tensor的维度），具体元素的值可能是None: None for shapes in TensorFlow allow for a dynamic size，None表示该维度可以是任意值。给tf.placeholder（tensor）命名时，需要使用tf.placeholder的name argument. 如果是一个标量，在tf.placeholder中不需要特意去给维度，只需要给数据类型，比如tf.float32(32位浮点数).

* 获取tensor的维度
有两种方法，第一，x\_tensor.get\_shape()是一个object，可以print，用x\_tensor.get\_shape().to\_list()可以转换为list，具体元素是int；第二，x\_tensor.shape[0]可以给出维度的第一个值，但也是一个object，不是int，要转换为int，需要使用x\_tensor.shape[3].value

* tf.truncated\_normal
第一个参数是维度，用list给（哪怕是一维向量，只有一个维度），参数还有mean和stddev，默认值分别是0和1.

* tf.nn.conv2d
用来得到二维扫描的convnet

* tf.nn.bias\_add
在conv layer上添加bias

* tf.nn.relu
relu函数

* tf.nn.max\_pool
max pool函数

* 存在未知维度(None)的tensor的展开，从高维向量变成低维向量，比如二维向量（flatten layer所做的事情）
代码：

    shape = x_tensor.get_shape().as_list() #首先拿到张量的维度
    dim = np.prod(shape[1:]) #把需要合并的维度乘起来，np.prod表示连乘
    tf.reshape(x_tensor, [-1, dim]) #使用tf.reshape改变维度，2D list表示改变成2维，第一个维度是-1，第二个维度是
    # 确定值的话，表示在保证总的元素个数的情况下，第二个维度固定，第一个维度需要具体计算

* tf.layers.dense
fully connected neural network, 可以选择activation function，也可以使用$f(x)=x$来做activation（也就是不加activation）

* tf.nn.dropout
用来添加dropout，防止过拟合

* tensor的命名

代码：
    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')
    
* cross entropy and mean

代码：
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    
* AdamOptimizer
不需要设置learning rate
代码：
    optimizer = tf.train.AdamOptimizer().minimize(cost)

* accuracy的计算

代码：
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)) #0和1组成的向量
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
* train_neural_network

代码：
    session.run(optimizer, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: keep_probability})
                