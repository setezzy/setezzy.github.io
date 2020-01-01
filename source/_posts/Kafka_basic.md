---
title: Kafka 基础
categories: 
  - Kafka
tags: 
  - kafka
---

## 核心概念

### Producer

消息生产者，Producer 发送消息需指定 topic，但不需指定 partition ；

kafka 通过 load balance自动分配 partition， 使得消息能够均匀分布

### Consumer

消息消费者，消息的状态被保存在 consumer 中；

读取消息为顺序读取 （只保证一个 partition 上的有序性，partition 间为无序）；

消费者通过定位自己的 offset 定位并读取消息

- ConsumerGroup：

分区的每条消息只能被group内的一个消费者消费；同一 group 的多个 consumer 可以消费同一个 topic，但不能同时消费一个 partition

consumer 数目最好不要大于 partition 数目 （partition不允许并发，会造成浪费）

### Broker

Kafka 节点 （用于引用服务器），多个 broker 组成一个 kafka 集群

节点间通信通过 zookeeper 管理

### Topic

消息源的不同分类，Kafka 集群能同时处理多个 topic 的分发

###  Partition

topic 的物理分区，每个分区为一个有序队列

分区内每条消息都会被分配一个有序的Id (offset)，消费者可以在请求中指定 offset，获取某特定位置开始的消息

每个 partition 可以在 kafka broker 节点上存储 replica 副本 (follower)，以便某一个 broker 宕机时不会影响整个 kafka 集Producer 写 kafka 时先写 partition leader，再由 leader push 给其他 follower。一旦某 broker 宕机，zookeeper 会选择一个 follower 变成 leader

### Message
消息

### Producers

向 kafka 的一个 topic 发送消息的过程

### Consumers

订阅 topic 并消费的过程

### 拓扑图：

![](Kafka/structure.png)


## 序列化和反序列化

### 序列化

生产者消息序列化后才能发送，对应的代码为：

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

`StringSerializer` 为内置的字符串序列化方式，通过 `getBytes("UTF8")` 来实现

若自定义序列化需实现 `Serializer` 接口

### 反序列化

消费者需要反序列化消息后才能得到真实内容，对应代码如下：

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

`StringDeserializer` 为内置的字符串反序列化方式，通过 `new String(data, "UTF8")` 实现

若自定义序列化需实现 `Deserializer` 接口

推荐使用 json 作为标准数据传输格式，使用内置的序列化和反序列化