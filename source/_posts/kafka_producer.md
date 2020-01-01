---
title: Kafka producer 消息发送问题
categories:
  - Kafka
tags:
  - kafka
---


在项目中采用了异步回调的方式发 Kafka 消息，结果程序执行后并没有打印任何信息

```java
if(record != null){
    kafkaProducer.send(record, (recordMetadata, e) -> {
        if(e != null){
            log.warn("fail to send message", e)
        }else{
            // 打印 metadata 信息
        }
    })
}
```


最后通过找资料找到了问题所在，我在这里对 producer 消息发送时可能存在的问题进行一些总结


### Kafka 消息发送流程

1. `new kafkaProducer()` 后会创建一个后台 IOThread，消息真正的发送与回调都是在该后台线程中执行的

2.  `kafkaProducer.send()` 实际上是将消息保存到 `RecordAccumulator` 中 （`ConcurrentMap<TopicPartition, Dequeue<ProducerBatch>>`），同一 topic 同一分区为一个批次，这批次的消息就被保存到了一个双向队列中

3. 后端线程会扫描 accumulator 资源池，将消息发送到 Kafka broker

4. 消息发送成功（成功写入 Kafka）会返回 `RecordMetaData` 对象，包含 topic 和分区信息、偏移量


### Kafka 消息发送方式

- 异步

   `kafkaProducer.send(record)`
   
   异步发送可指定回调函数，发送完成通过回调通知 producer
   
- 同步

  `kafkaProducer.send().get()`

   该方式是通过调用 `send` 方法返回的 Future 对象的 `get()` 方法，阻塞等待消息发送完成


### 问题及解决

同步方法相比于异步方法较为耗时，所以一般推荐采用异步回调来发送 Kafka 消息

这时可能出现以下问题：

主线程退出，但后台 IO 线程还没有来得及从队列中获取消息和发送给 broker，使得消息丢失

解决方案：

- 采用同步方式，即 `.send().get()`
- `Thread.sleep()`，让主线程睡眠一段时间，使后台线程有时间将消息发送出去


&nbsp;
&nbsp;

参考博客：
[Kafka producer 源码解析](http://generalthink.github.io/2019/03/07/kafka-producer-source-code-analysis/)
[一次Kafka producer 没有 close 引起的思考](https://blog.csdn.net/qyhuiiq/article/details/88757209)
