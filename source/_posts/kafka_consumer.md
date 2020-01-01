---
title: Kafka consumer 消费问题
categories:
  - Kafka
tags:
  - kafka
---

## pull or push

一般消息中间件存在推送和拉取两种方式获取服务器数据，而 kafka consumer 采用的是主动拉取 broker 数据进行消费

原因在于服务器推送速度可能与 consumer 消费速度不匹配，特别是 consumer 中执行比较复杂和耗时的业务处理时，consumer 可能会不堪重负导致系统故障

选择主动拉取的方式可以避免这类问题，consumer 可以根据自己的状态拉取数据，对数据进行延迟处理


## 消费精准性问题

### offset

关于 offset 有两个核心概念：

- Last committed offset：consumer group 最新一次提交的 offset，表明这之前的数据都已经消费成功
- Current Position：当前数据的 offset，即 last committed offset 到 current position 之间的消息拉取成果，可能正在处理，还未提交

消费者与服务器的消费确认非常重要，异步模式下 committed offset 落后于 current position，一旦 consumer 出现故障，下一次消费又只会从 committed offset 开始拉取数据，导致重复消费。

### Kafka 消息分发语义

- at-most-once：消息可能丢失，但不会重复消费
   
   读取消息 -- 保存位置 -- 处理消息
   
- at-least-once：消息不会丢失，但可能重复消费

   读取消息 -- 处理消息 -- 保存位置

- exactly-once：精确消费

   自己管理 commit offset；在重新启动时使用 seek(TopicPartition, long) 来恢复上次消费的位置

### Kafka 提交策略

- 自动提交

```java
props.put("enable.auto.commit", "true");
// 设置自动提交的时间间隔
props.put("auto.commit.interval.ms", "1000");
```

在  `consumer.poll()` 中提交完成提交：

consumer 加入 group；

获取所有已被消费的分区的offset信息；

异步提交offset

- 手动同步提交

```java
consumer.commitSync();
```

提交成功马上返回，否则抛出异常

- 手动异步提交

```java
consumer.commitAsync();
```

- 同步异步组合提交

可以在消息处理后异步提交，在 consumer close 前同步提交一次