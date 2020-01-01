---
title: Spark Basic
categories: 
  - Spark
tags: spark
---

## Spark 优势
- 基于 RDD（弹性分布式数据集），抽象层次更高
- 计算模式属于 MapReduce，但还提供了多种转换和动作，编程模型比MapReduce 更灵活
- 提供内存计算，中间结果放在内存中，迭代运算效率更高（Hadoop将中间结果存放在磁盘，IO开销大）
- 基于有向无环图（DAG）的任务调度执行机制
- 支持实时处理流数据（拆分成 batch）

## Spark 生态系统
- spark core：包含 spark 基本功能，如内存计算、任务调度、部署模式、存储管理等。Spark 建立在 RDD 上
- spark SQL：允许开发人员直接处理 RDD，同时查询 Hive，HBase 等外部数据源。使得开发人员能使用 SQL 命令进行查询和复杂的数据分析
- spark streaming：支持高吞吐量、可容错处理的实时流数据处理。核心思路是将流式计算分解成短小的批处理作业。支持多种数据源，例如 Kafka，TCP sockets、Flume等
- MLlib：提供常用的机器学习算法的实现
- GraphX：图计算

## 基本概念
### RDD
弹性分布式数据集。是分布式内存的抽象概念，提供了一种高度受限的共享内存模型（不可变的分区记录集合）

通常 RDD 很大，会被分成多个分区，保存在不同节点上。原则：分区个数尽量等于集群中 CPU core 数目

RDD 基本操作包括：
- 转换：Transformation，定义 RDD 间相互依赖关系，每一次转换操作即生成一个新的 RDD。转换操作不会触发真正的计算
- 动作：Action，执行计算，指定输出形式，返回结果。只有进行动作操作时，spark才会根据RDD依赖关系生成DAG，从起点开始进行真正的运算

RDD 依赖关系分为：
- 窄依赖：父RDD的一个分区只被子RDD的一个分区使用（一对一，多对一）。例如 map, filter, union
- 宽依赖：父RDD的分区对应子RDD的多个分区（一对多）。例如 groupByKey, sortByKey

### DAG
有向无环图，反应 RDD 之间的依赖关系 （新的 RDD 依赖于旧 RDD，于是一段程序就形成了一个 DAG）

### Executor
运行在工作节点上的一个进程，负责运行任务，并未应用程序存储数据

### 任务（Task）
运行在 Executor 上的工作单元

###  作业（Job）
一个作业包含多个 RDD 及应用于 RDD 的各种操作

### 阶段（Stage）
作业的基本调度单位，一个作业分为多组任务，每组任务（任务集）被称为阶段

阶段的划分原则：

遇到宽依赖就断开；遇到窄依赖就将当前 RDD 加入到当前stage

###  应用（Application）
一个应用由一个任务控制节点，多个job组成，每个job包含多个stage，每个stage包含多个 task

## RDD 编程
创建 RDD：

```scala
//创建程序执行的上下文
val sc = new SparkContext(“server”, “AppName”, “Sparkhome”, “Appjar”)
//读数据源，参数可以是文件名/目录/压缩包
val rdd = sc.textFIle(“fileURI”)`
//执行转换操作
val filterRDD = rdd.filter(_.contains(“text”)_)
// 保存在内存中
filterRDD.cache()
//执行动作，触发真正的计算
filterRDD.count()

```

###  RDD 转换操作
- filter(func)：筛选满足函数 func 的元素，返回新的 RDD
- map(func)：将每个元素传递到 func 中
- mapValues(func)：仅对值应用函数 func
- flatMap(func)：每个元素可以映射多个输出结果
- groupByKey()：应用于(k,v)键值对的数据集时，返回一个新的(k,iterable)形式数据集。即对相同 key 的值进行分组

e.g. `("spark", 1), ("spark", 2), ("hadoop", 3), ("hadoop", 5)`，结果为 `("spark", (1,2)), ("hadoop", (3,5))`

- reduceByKey(func)：应用于(k,v)键值对的数据集时，返回一个新的(k,v)形式数据集，其中每个值是将每个 key 传递到 func 中进行聚合（合并具有相同 key 的值）

e.g. `reduceByKey((a,b) => a+b)`，有四个键值对 `("spark", 1), ("spark", 2), ("hadoop", 3), ("hadoop", 5)`，合并后的结果为 `("spark", 3), ("hadoop", 8)`

### RDD 行动操作
- count()：返回元素个数
- collect()：以数组形式返回数据集所有元素
- first()：返回第一个元素
- take(n)：返回前 n 个元素
- reduce(func)：聚合数据集中的元素，func 输入两个参数返回一个值
- foreach(func)：将每个元素传递到func中运行

统计文本中单行文本所包含的单词数最大值：

```scala
val lines = sc.textFile(“…”)
lines.map(line => line.split(“ ”).size).reduce(a,b) => if (a>b) a else b

```


### 综合实例

```scala
val rdd = sc.parallelize(Array((“spark”, 2), (“hadoop”, 6), (“hadoop”, 4), (“spark”, 6)))

rdd.mapValues(x => (x,1)).reduceByKey((x,y) => (x._1 + y._1, x._2 + y._2)).mapValues(x => (x._1 / x._2)).collect()

```

理解：

- 构建包含四对键值对的数组，调用 parallelize 方法生成 RDD
- 首先将每个键值对的 value 进行修改，转换为 (value, 1)。这里的 1 就表示该 key 出现了 1 次：`(“spark”, (2,1)), (“hadoop”, (6,1)), (“hadoop”, (4,1)), (“spark”, (6,1)) `
- x,y 是相同键对应的 value。e.g. x 为 (2,1)，y 为(6,1)，最后生成键值对( “spark”,(8,2))
- 最后相除得到平均值，新的键值对为 (“spark”, 4)


### 应用程序打包

将生成的 jar 包通过 spark-submit 提交到 spark 运行

```java
/usr/local/spark/bin/spark-submit —class “AppName” <jar包位置>

```

spark-submit 格式如下：

```java
./bin/spark-submit
  —-class <main-class>  //需要运行的程序主类，应用程序的入口
  —-master <master-url>  //local或者server或者集群
  —-deploy-mode <deploy-mode>  //部署模式
  <application-jar>  //应用程序jar包
  [application-arguments]  //传递给主类的主方法的参数

```