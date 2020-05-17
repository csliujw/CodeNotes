# 0.引言

我们软件复杂度设计的核心往往是数据的不一致。我们要保证单一数据源，确保数据保持一致，避免数据不一致带来的各种问题。典型的是并发问题！

保持数据的一致性有很多种方法：加锁，数据只读，统一更改等【操作系统&组成原理中有相应策略】

一致性问题的解决办法

- 排队（例如：锁、互斥量、管程）
- 投票（例如：Paxos，Raft等）

初中级 了解基本概念、原理（再别人做好的基础上进行开发）

高级 应对不同场景、正确使用、到达预期结果（了解每种数据结构合适的使用场景，理解为什么要这样用）

高专 深度掌握原理、本质，对不满足自身需求的可以进行改进，扩展（为什么要有这种数据结构，为什么要这样实现）

# 一、开篇

ThreadLocal，线程局部变量，每个线程都绑定了自己的值，互不干扰。

场景介绍。【每个线程都会有自己的ThreadLocal，不同的线程访问的ThreadLocal是不一样的！】

- 资源持有
- 线程安全
- 线程一致
- 并发计算

# 二、API使用

- set
- get
- initialValue.在获取时，若发现没有数据，则会调用initalValue方法

# 三、使用场景

## 3.1 并发场景分析

工具 apache2-util

```java
static Integer c = 0;

@RequestMapping("/start")
public Integer stat(){
    retur c;
}

@RequestMapping("/add")
public Integer add(){
    c++;
    return c;
}
```

在并发场景中，数据不一致。

ThreadLocal可以保证数据安全的操作，但是不同线程的ThreadLocal是不一样的。因为一个线程可能有多个ThreadLocal变量，所以需要一个数据结构进行维护这些ThreadLocal，Map是一个不错的选择。每个ThreadLocal内部都持有一个ThreadLocalMap静态类，一个线程，ThreadLocalMap只会被实例化一次，所以可以做到维护多个ThreadLocal。

如何解决？

让每个线程进行安全的计算，在获取的时候，获取所有线程的值进行累加！我们用ThreadLocal安全的操作每个线程处理的变量，在返回值的时候，对所有ThreadLocal进行累加。可以采用静态的Set集合，存储所有ThreadLocal中的值，但是，在Set存入值的时候可能会出现并发问题，Set.add()方法需要进行加锁。

从大面积的加锁，转向了小面积的加锁！

- 完全避免同步（难）
- 缩小同步范围（简单）+ThreadLocal解决问题

## 3.2 Quartz

## 3.3 MyBatis

`MyBatis`的`SqlSessionFactor`是将连接存入到`ThreadLocal`中，获取也是直接从`ThreadLocal`中获取，从而保证拿到的是同一个连接。

