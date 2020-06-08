# learn-python

## 1. 基本内容

### base directory of a project

[base_dir](https://github.com/arfu2016/nlp/tree/master/nlp_models/base_dir)

### 去掉非中文词汇

[batch-t2s_seg_clean](https://github.com/arfu2016/nlp/tree/master/nlp_models/batch-t2s_seg_clean)

[select_chinese_words](https://github.com/arfu2016/nlp/tree/master/nlp_models/select_chinese_words)

### binary search

[binary_search](https://github.com/arfu2016/nlp/tree/master/nlp_models/binary_search)

### cache

[cache](https://github.com/arfu2016/nlp/tree/master/nlp_models/cache)

### callback

[callback](https://github.com/arfu2016/nlp/tree/master/nlp_models/callback)

### cartesian product

[cartesian_product](https://github.com/arfu2016/nlp/tree/master/nlp_models/cartesian_product)

### concurrency

[concurrent](https://github.com/arfu2016/nlp/tree/master/nlp_models/concurrent)

### confusion matrix

[confusion_matrix](https://github.com/arfu2016/nlp/tree/master/nlp_models/confusion_matrix)

### construct LineSentence

[construct_LineSentence](https://github.com/arfu2016/nlp/tree/master/nlp_models/construct_LineSentence)

### data analysis

[data_analysis](https://github.com/arfu2016/nlp/tree/master/nlp_models/data_analysis)

### decision tree

[decision_tree](https://github.com/arfu2016/nlp/tree/master/nlp_models/decision_tree)

决策树建模非常灵活自由，是一种nonparametric的方法，可以用来处理各种数据，同时具有很强的可解释性。

自变量可以是连续或者离散变量，因变量也可以是连续或者，因变量可以有多个，表现为矩阵的多列。

### feature scaling

[feature_scaling](https://github.com/arfu2016/nlp/tree/master/nlp_models/feature_scaling)

### file read

[file_read](https://github.com/arfu2016/nlp/tree/master/nlp_models/file_read)

[load_table_sql_csv_excel](https://github.com/arfu2016/nlp/tree/master/nlp_models/load_table_sql_csv_excel)

[path_file_object](https://github.com/arfu2016/nlp/tree/master/nlp_models/path_file_object)

### file write

[file_write](https://github.com/arfu2016/nlp/tree/master/nlp_models/file_write)

[part_text](https://github.com/arfu2016/nlp/tree/master/nlp_models/part_text)

### gradient descent

[gradient_descent](https://github.com/arfu2016/nlp/tree/master/nlp_models/gradient_descent)

### greedy algorithm

[greedy](https://github.com/arfu2016/nlp/tree/master/nlp_models/greedy)

### Kmeans clustering

[kmeans](https://github.com/arfu2016/nlp/tree/master/nlp_models/kmeans)

### lagrange interpolation

[lagrange_interpolation](https://github.com/arfu2016/nlp/tree/master/nlp_models/lagrange_interpolation)

### multiple layer perceptron

[multiple_layer_perceptron](https://github.com/arfu2016/nlp/tree/master/nlp_models/multiple_layer_perceptron)

### named tuple

[namedtuple](https://github.com/arfu2016/nlp/tree/master/nlp_models/namedtuple)

### pca

[pca](https://github.com/arfu2016/nlp/tree/master/nlp_models/pca)

### pearson corelation coefficient

[pearson](https://github.com/arfu2016/nlp/tree/master/nlp_models/pearson)

### pipeline

[pipeline](https://github.com/arfu2016/nlp/tree/master/nlp_models/pipeline)

### pprint

[pprint_list](https://github.com/arfu2016/nlp/tree/master/nlp_models/pprint_list)

### property decorator

[property](https://github.com/arfu2016/nlp/tree/master/nlp_models/property)

### q2answer

[q2answer](https://github.com/arfu2016/nlp/tree/master/nlp_models/q2answer)

### deal with punctuation

[punctuation](https://github.com/arfu2016/nlp/tree/master/nlp_models/punctuation)

### random compliment for dataset

[random_compliment](https://github.com/arfu2016/nlp/tree/master/nlp_models/random_compliment)

### random data generation

[random_data](https://github.com/arfu2016/nlp/tree/master/nlp_models/random_data)

### logging

[set-logger](https://github.com/arfu2016/nlp/tree/master/nlp_models/set-logger)

[tf-logger](https://github.com/arfu2016/nlp/tree/master/nlp_models/tf-logger)

### sort

[sort](https://github.com/arfu2016/nlp/tree/master/nlp_models/sort)

[shellsort](https://github.com/arfu2016/nlp/tree/master/nlp_models/shellsort)

### str and repr

[str_repr](https://github.com/arfu2016/nlp/tree/master/nlp_models/str_repr)

### timeit in python

[timeit_in_python](https://github.com/arfu2016/nlp/tree/master/nlp_models/timeit_in_python)

### tokenizer

[tokenizer](https://github.com/arfu2016/nlp/tree/master/nlp_models/tokenizer)

### train validation test split

[train_val_test](https://github.com/arfu2016/nlp/tree/master/nlp_models/train_val_test)

### unit test

[unit_test](https://github.com/arfu2016/nlp/tree/master/nlp_models/unit_test)

### vector calculation

[vector_calculation](https://github.com/arfu2016/nlp/tree/master/nlp_models/vector_calculation)

###  词汇表与id互转

[vocab_deeplearning](https://github.com/arfu2016/nlp/tree/master/nlp_models/vocab_deeplearning)

[word_index](https://github.com/arfu2016/nlp/tree/master/nlp_models/word_index)

### 维基百科数据的处理

[wiki_process](https://github.com/arfu2016/nlp/tree/master/nlp_models/wiki_process)

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

### serializing and deserializing

serializing是把python中的对象转化为字符串，然后可以保存到文本文件或者数据库中。

虽然都是转化为字符串，但serializing有多种方法，所以deserializing也有多种方法。

    encode()与decode()
    str()与int()
    repr()与eval()
    json.dumps()与json.loads()
    pickle.dumps()与pick.loads()

### iterable, iterator或者generator代替list

[construct_iterable](https://github.com/arfu2016/nlp/tree/master/nlp_models/construct_iterable)

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

### math (python standard library)

[math](https://github.com/arfu2016/nlp/tree/master/nlp_models/math)

### pandas

* 大数据文件的读入，所需时间的估计与比较

* 索引

df.loc等

    df.loc['index_name', 'column_name']
    
可以像matlab一样在'index_name'或者'column_name'用中括号括起来的向量

df.iloc是类似的，用的是序号；df.ix既可以用序号，也可以用名字

如果要交换两列，直接用这样的方法是不对的：

    df.loc[:,['B', 'A']] = df[['A', 'B']]
    
这是因为pandas默认在赋值的时候回匹配列名，这里面的AB和BA实际上没有区别。如果想要交换两列的话，应该使用AB两列的值作为右值，这样就不带列索引名了。

    df.loc[:,['B', 'A']] = df[['A', 'B']].values
    
.values其实已经把pandas.DataFrame转换成了numpy.array.

* 替换

df.replace()

参数是字典，标明要把什么替换为什么，可作用于整个dataframe，真的很方便

### sklearn (scikit-learn)
* 多列数据中一列A中多个类别在另一列B中的词频统计

### numpy

[numba_det](https://github.com/arfu2016/nlp/tree/master/nlp_models/numba_det)

    import numpy as np
* one-hot encoding

代码：

    x = [0, 2, 9]
    n_classes = 10
    np.eye(n_classes)[x]
    # 每一个类用不同的行向量表示
    
除了用numpy，也可以用sklearn中的LabelBinarizer or OneHotEncoder。

### matplotlib

[matplotlib](https://github.com/arfu2016/nlp/tree/master/nlp_models/matplotlib)

[matplotlib绘图系列文章](https://zhuanlan.zhihu.com/p/37595853)

[matplot同时画两条线](https://blog.csdn.net/x_i_y_u_e/article/details/50319441)

    fig, ax = plt.subplots()
    ax.set_xlim(1,4)  # 设定x轴范围
    ax.set_ylim(-8.5,11) # 设定y轴范围
    
    plt.plot(x, color='red', label="x")
    plt.plot(y, color='blue', label="y")
    ax.set_ylim(ymin=12)
    plt.title('Mutualism Model', fontweight='bold', fontsize='large')
    plt.xlabel('Time', fontweight='bold', fontsize='large')
    plt.ylabel('Population', fontweight='bold', fontsize='large')
    xticks = range(0,200,50)
    ax.set_xticks(xticks)
    plt.legend(bbox_to_anchor=(1, 0), loc=4, borderaxespad=0.)
    plt.show()
    
* 在matplotlib中显示中文

下载SimHei.ttf字体  
找到matplotlib字体文件夹，例如：matplotlib/mpl-data/fonts/ttf，将SimHei.ttf拷贝到ttf文件夹下面  
在文件中添加代码

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus']=False  #用来正常显示负号
    
改了配置之后并不会生效，需要重新加载字体，在Python中运行如下代码即可：

    from matplotlib.font_manager import _rebuild
    _rebuild() #reload一下
    
线条的属性linestyle

'-' 实线  
'--' 破折线  
'-.' 点划线  
':' 虚线

线条标记marker

'o' 圆圈  
'D' 菱形
'p' 五边形  
'+' 加号  
's' 正方形  
'*' 星号
'd' 小菱形
'x' X

绘图

    plt.figure(num = 5, figsize = (4, 4))
    # num可以是int，也可以是字符串

title例子：

    plt.title('Interesting Graph',fontsize='large'，fontweight='bold') 设置字体大小与格式
    plt.title('Interesting Graph',color='blue') 设置字体颜色
    plt.title('Interesting Graph',loc ='left') 设置字体位置
    plt.title('Interesting Graph',verticalalignment='bottom') 设置垂直对齐方式
    plt.title('Interesting Graph',rotation=45) 设置字体旋转角度
    plt.title('Interesting',bbox=dict(facecolor='g', edgecolor='blue', alpha=0.65 )) 标题边框
    
### tensorflow

[parameter_number](https://github.com/arfu2016/nlp/tree/master/nlp_models/parameter_number)

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

* bucketing

[bucketing](https://github.com/arfu2016/nlp/tree/master/nlp_models/bucketing)

