# learn-python

## 1. 基本内容
### 文本读入与分词

with open() as ite:

ite是一个iterator，每一步都储存着一个string（一行的内容）

对于csv文件，这个string是用,分开的，分词的话，需要用string的split(',')函数，把字符串转化为较短的list

    g = open()
    reviews = list(map(lambda x:x[:-1],g)) # g.readlines()，python2.5之后g是iterator
    g.close()

reviews是个list，每个元素是读入的行

### iterator或者generator代替list

iterator是一个类，有__iter__()、next()等方法，generator是一个特殊的函数，能用在for循环中，函数返回值不断被调用，但用在for循环中时，看上去和iterator是一样的

当list很长时，应该用iterator或者generator代替list，节省内存空间

iterator或者generator有三种产生办法，一种是在函数中用yield语句产生generator，另一种是在list comprehension中使用括号产生generator，第三种是用某个类的构造函数或者其他函数返回一个iterator（一个具有iter()和next()的类）.

(i for i in list)

### 对for循环的替代

generator的生成、list comprehension、dict comprehension

Python函数式编程之map()、filter()

map()带两类参数，第一类参数是个函数，第二类参数是一个或者多个iterator(比如list、tuple、字符串等)，返回的也是一个iterator，具体值是函数作用于iterator的结果

filter带两类参数，第一类参数是个函数，第二类参数是一个或者多个iterator(比如list、tuple、字符串等)，返回的也是一个iterator，具体值是函数作用于iterator后结果为True的值

### 单列频数统计与Counter from collections

collections是python的built-in module

Couter class返回的是一个dict-like project，key可以是需要统计频数的词，value是具体的频数

    counts = Counter()
    for i in generator: # generator or list of words
        counts[i]+=1
        
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

## 2. 第三方模块

### pandas
* 大数据文件的读入，所需时间的估计与比较

### sklearn
* 多列数据中一列A中多个类别在另一列B中的词频统计

### tensorflow
