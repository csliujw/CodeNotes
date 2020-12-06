# MySQL InnoDB存储引擎

记录读书过程中感觉重要的部分内容。

<span style="color:green">**PS：非DBer，故不会太细致，主要学习原理。**</span>

# MySQL体系结构和存储引擎

## 体系结构

- MySQL 单进程多线程架构；MySQL数据库实例在系统上的表现就是一个进程
- MySQL区别于其他数据库的最重要的特点：
  - 插件式的表存储引擎
  - MySQL的存储引擎是基于表的，而不是数据库

## 存储引擎

### InnoDB存储引擎

-  支持事务、行锁设计、支持外键（从5.5.8开始，InnoDB为默认存储引擎）
- 提供插入缓冲（insert buffer）、二次写（double  write）、自适应哈希索引（adaptive hash index）、预读（read ahead）等高性能和高可用的功能
- 表中数据的存储，InnoDB采用聚集的方式【什么意思？】
- 1.2的InnoDB也支持全文索引了~

### MyISAM存储引擎

- 不支持事务、表锁设计、支持全文索引，主要面向一些OLAP数据库应用
- MyISAM存储引擎的缓冲池只缓存索引文件，不缓冲数据文件。
- 数据库应用类型如下：
  - 对于SQL开发人员来说，必须先要了解进行SQL编程的对象类型，即要开发的数据库应用是哪种类型。一般来说，可将数据库的应用类型分为OLTP（OnLine Transaction Processing，联机事务处理）和OLAP（OnLine Analysis Processing，联机分析处理）两种。
  - OLTP是传统关系型数据库的主要应用，其主要面向基本的、日常的事务处理，例如银行交易。
  - OLAP是数据仓库系统的主要应用，支持复杂的分析操作，侧重决策支持，并且提供直观易懂的查询结果。

### Memory存储引擎

- 表中数据存放在内存中，适用于存储临时数据的临时表。
- 速度快，但是限制也很多，只支持表锁，并发性能较差。

### 命令汇总

- 查看存储引擎  `show engines\G;`

- 连接mysql  `mysql -hip地址  -u用户名 -p密码`

  eg：`mysql -h27.0.0.1 -uroot -proot`

- 查看mysql权限视图

  eg：`use mysql`

  ​		`select host,user,password from user;`

## 行锁和表锁

# InnoDB存储引擎

## 概述

- 第一个完整支持ACID事务的MySQL存储引擎
- 行锁设计、支持MVCC、支持外键、提供一致性非锁定读
- 被设计用来最有效地利用以及使用内存和CPU

## InnoDB体系架构

有多个内存块，内存块组成一个大地内存池。

有后台线程，主要负责刷新内存池中的数据。

### 后台线程

> **Master Thread**

Master Thread是一个非常核心的后台线程，主要负责将缓冲池中的数据异步刷新到磁盘，保证数据的一致性，包括脏页的刷新、合并插入缓冲（INSERTBUFFER）、UNDO页的回收等。

> **IO Thread**

在InnoDB存储引擎中大量使用了AIO（Async IO）来处理写IO请求，这样可以极大提高数据库的性能。而IO Thread的工作主要是负责这些IO请求的回调（callback）处理。

> **Purge Thread**

事务被提交后，其所使用的`undolog`可能不再需要，因此需`Purge Thread`来回收已经使用并分配的`undo`页。在`InnoDB 1.1`版本之前，`purge`操作仅在`InnoDB`存储引擎的`Master Thread`中完成。而从`InnoDB 1.1`版本开始，`purge`操作可以独立到单独的线程中进行，以此来减轻`Master Thread`的工作，从而提高`CPU`的使用率以及提升存储引擎的性能。用户可以在`MySQL`数据库的配置文件中添加如下命令来启用独立的`Purge Thread：`

```shell
[mysqld]
innodb_purge_thrreads = 1
```

> **Page Cleaner Thread**

`Page Cleaner Thread`是在`InnoDB 1.2.x`版本中引入的。其作用是将之前版本中脏页的刷新操作都放入到单独的线程中来完成。而其目的是为了减轻原`MasterThread`的工作及对于用户查询线程的阻塞，进一步提高`InnoDB`存储引擎的性能

### 内存

#### 缓冲池

CPU速度与磁盘速度之间差距过大，基于磁盘的数据库系统通常用缓冲池技术来提高数据库的整体性能。

> **重要内容提及**

在数据库中读取页的操作：先将从磁盘读取到页放入缓冲池中，这个过程称为将页“FIX”在缓冲池。下一次读取时，先在缓冲池中找。（与OS的页面查找类似）

对数据库中页的修改，先改缓冲池中的页，再以一定的频率刷新到磁盘上。以一种称之为<span style="color:green">**`Checkpoint`**</span>的机制刷新回磁盘。

<span style="color:green">**缓存池中缓存的数据页类型：**</span>索引页、数据页、插入缓冲、自适应哈希索引、InnoDB存储的锁信息、数据字典信息等。不过缓存索引页和数据页占缓冲池很大的一部分~

<span style="color:green">**从InnoDB 1.0.x开始，允许多个缓冲池实例**</span>

#### LRU

<span style="color:green">MySQL InnoDB的LRU并非传统的LRU，新数据是存在midpoint位置而非首部。</span>

数据库中的缓冲池是通过LRU（Latest Recent Used，最近最少使用）算法来进行管理的。

MySQL InnoDB的LRU并非传统的LRU，新数据是存在midpoint位置。这个算法在InnoDB存储引擎下称为midpoint insertion strategy。默认配置下在LRU列表长度的5/8处。midpoint之后的列表称为old列表，之前的列表称为new列表。可简单理解为，midpoint之前的都是最为活跃的热点数据。

当前数据可能只是这次要用到，如果直接把数据放到链表的首部，那么可能会让真正的热点数据丢失，因此引入了midpoint机制。

用户可以对midpoint的值进行调整，比如用户预估自己的热点数据高达90%，则可以调整midpoint的值。

```shell
# 数据插入到链表尾部10%的位置
set global innodb_old_blocks_pct = 10 
```

## Checkpoin技术

挖坑，后面填

## InnoDB关键特性

### 概述

- 插入缓冲（Insert Buffer）
- 两次写（Double Write）
- 自适应哈希索引（Adaptive Hash Index）
- 异步IO（Async IO）
- 刷新邻接页（Flush Neighbor Page）

