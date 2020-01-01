---
title: Spark Streaming
categories: 
  - Spark
tags: spark
---

Spark Streaming 能够实现实时的数据流的流式处理（秒级），并支持将处理完的数据推送到文件系统、数据库等

maven 依赖如下：
```java
<dependency>
    <groupId>org.apache.spark</groupId>
    <artifactId>spark-streaming_2.12</artifactId>
    <version>2.4.3</version>
    <scope>provided</scope>
</dependency>
```

## 创建 StreamingContext

StreamingContext 对象是 Spark Streaming 程序的主入口

scala ：
```scala
import org.apache.spark._
import org.apache.spark.streaming._
val conf = new SparkConf().setAppName("name").setMaster("local[2]") //本地运行模式，两个线程，一个监听，一个处理数据
val ssc = new StreamingContext(conf, Seconds(1)) // seconds 表示每隔1s就自动执行一次流计算
```

Java ：
```java
import org.apache.spark.*;
import org.apache.api.java.function.*;
import org.apache.spark.streaming.*;
import org.apache.spark.streaming.api.java.*;

SparkConf conf = new SparkConf().setMaster("local[2]").setAppName("WordCount");
JavaStreamContext jssc = new JavaStreamingContext(conf, Durations.seconds(1));
```

## 创建输入源
### 文件输入源

scala

```scala
val lines = ssc.textFileStream("file")
// 这里写一些 RDD 转换和动作
// 启动接收数据
ssc.start()
ssc.awaitTermination()
```

java:
```java
streamingContext.textFileStream(dataDirectory);
```

### Kafka 输入源

Kafka 为高级输入源，需要依赖独立的库（jar）：官网下载 `spark-streaming-kafka-0-10_2.12 `相关 jar 包，保存到 Spark 目录的 jars 目录下，并将 kafka 安装目录的 libs 下所有 jar 复制到 `spark/jars/kafka` 目录下

需要的 maven 依赖：
```java
<dependency>
  <groupId>org.apache.kafka</groupId>
  <artifactId>kafka-clients</artifactId>
  <version>2.12</version>
</dependency>
```

构建生产者：

```java
Properties props = new Properties();
// broker 地址 (集群地址)
props.put("bootstrap.servers", "ip:port");
// kafka消息key的序列化方式
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
// kafka消息value的序列化方式
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
KafkaProducer<String, String> kafkaProducer = new KafkaProducer<>(props);
```

创建消息：

ProducerRecord 对象即发送的信息对象，包括 topic, key(可选), value，由kafka决定分区

具有相同 key 的消息会被写到同一分区

```java
ProducerRecord<String, String> record = new ProducerRecord<>("ORDER-DETAIL", 
    // 将要发送的消息序列化为JSON
    JSON.toJSONString(new Order(para1, para2, para3)));
// 发送消息    
producer.send(record);
producer.close();
```

创建消费者：

```java
Properties props = new Properties();
props.put("boostrap.servers", "ip:port");
// 每个消费者有独立的组号
props.put("group.id", "group_name");
// key 的反序列化方式
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
// value 的反序列化方式
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
KafkaConsumer<String, String> kafkaConsumer = new KafkaConsumer<>(props);
```

消费者订阅topic并消费

```java
KafkaConsumer.subscribe(Lists.newArrayList("ORDER-DETAIL"));
try{
    while(true){
        // 消费者必须持续从kafka进行轮询，否则会被认为死亡
        // 从而导致它处理的分区被交给同一 consumer group 的其他消费者
        ConsumerRecords<String, String> records = kafkaConsumer.poll(1000);
        // 为防止消费者被认为死亡，需要尽可能确保处理消息工作尽快完成
        for(ConsumerRecord<String, String> record: records){
            System.out.println("message content:"+GSON.toJson(record));
            System.out.println("message value:"+record.value());
        }
        // 每次消费完后异步提交
        kafkaConsumer.commitAsync();
    }finally{
        // 消费者关闭之前同步提交
        kafkaConsumer.commitSync();
        kafkaConsumer.close();
    }
}
```

## DStream 输出

### 输出至文件

调用 `saveAsTextFile()` 方法

```scala
xDStream.saveAsTextFiles("file:///path")
```

### 输出至数据库

```scala
xDstream.foreachRDD(rdd => {
      //内部函数,接收 records并保存到数据库
      def func(records: Iterator[(String,Int)]) {
        var conn: Connection = null
        var stmt: PreparedStatement = null
        try {
          val url = "jdbc:mysql://localhost:3306/spark"
          val user = "root"
          val password = "hadoop"
          conn = DriverManager.getConnection(url, user, password)
          // 对 records 中每条记录 p 都插入数据库
          // p 的类型为 [String, Int]
          records.foreach(p => {
            val sql = "insert into wordcount(word,count) values (?,?)"
            stmt = conn.prepareStatement(sql)
            // 对应第一个问号
            stmt.setString(1, p._1.trim)
            // 对应第二个问号
            stmt.setInt(2,p._2.toInt)
            stmt.executeUpdate()
          })
        } catch {
          case e: Exception => e.printStackTrace()
        } finally {
          if (stmt != null) {
            stmt.close()
          }
          if (conn != null) {
            conn.close()
          }
        }
      }
      // 对 RDD 重新设置分区
      val repartitionedRDD = rdd.repartition(3)
      // 将每个分区的数据保存到数据库
      // 等价于 repartitionedRDD.foreachPartition(records => func(records))
      repartitionedRDD.foreachPartition(func)
    })
```

RDD 重新分区的原因：每次保存 RDD 至数据库都需要启动数据库连接，RDD分区数太大会带来多次连接数据库的开销