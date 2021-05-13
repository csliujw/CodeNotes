周阳Redis (https://www.bilibili.com/video/BV1Rv41177Af?from=search&seid=4054525143288945156)

https://blog.csdn.net/weixin_48013460/article/details/111885274?spm=1001.2014.3001.5502#MySQL_124

# NoSQL简介

## 技术发展

> 技术的分类

1、解决功能性的问题：Java、Jsp、RDBMS、Tomcat、HTML、Linux、JDBC、SVN

2、解决扩展性的问题：Struts、Spring、SpringMVC、Hibernate、Mybatis

3、解决性能的问题：NoSQL、Java线程、Hadoop、Nginx、MQ、ElasticSearch

----

Web1.0的时代，数据访问量很有限，用一夫当关的高性能的单点服务器可以解决大部分问题。

随着Web2.0的时代的到来，用户访问量大幅度提升，同时产生了大量的用户数据。加上后来的智能移动设备的普及，所有的互联网平台都面临了巨大的性能挑战。

> 解决CPU及内存压力

集群+分布式：请求分担到各个服务器上，减轻单台服务器的压力。

问题：session该如何处理？如何存储？

- session存储到客户端；不安全！
- session复制，把session复制到各个服务器上。（浪费空间）
- 所有session复制到一个地方。存储到NoSQL数据库（NoSQL无需IO）

<img src="..\pics\redis\image-20210513191632352.png">

> 解决IO压力

<img src="..\pics\redis\1.png">

## NoSQL数据库

### NoSQL数据库概述

NoSQL(NoSQL = Not Only SQL)，意即“不仅仅是SQL”，泛指非关系型的数据库。 

NoSQL 不依赖业务逻辑方式存储，而以简单的key-value模式存储。因此大大的增加了数据库的扩展能力。

l 不遵循SQL标准。

l 不支持ACID。

l 远超于SQL的性能。

### NoSQL适用场景

l 对数据高并发的读写

l 海量数据的读写

l 对数据高可扩展性的

### NoSQL不适用场景

l 需要事务支持

l 基于sql的结构化查询存储，处理复杂的关系,需要即席查询。

`l （用不着sql的和用了sql也不行的情况，请考虑用NoSql）`

### 常见NoSQL数据库

> Memcache

- 很早出现的NoSql数据库
- 数据都在内存中，一般不持久化
- 支持简单的key-value模式，支持类型单一
- `一般是作为缓存数据库辅助持久化的数据库`

> Redis

- 几乎覆盖了Memcached的绝大部分功能
- 数据都在内存中，支持持久化，主要用作备份恢复
- 除了支持简单的key-value模式，还支持多种数据结构的存储，比如 list、set、hash、zset等。
- `一般是作为缓存数据库辅助持久化的数据库`

> MongoDB：文档型数据库

- 高性能、开源、模式自由(schema  free)的文档型数据库
- 数据都在内存中， 如果内存不足，把不常用的数据保存到硬盘
- 虽然是key-value模式，但是对value（尤其是json）提供了丰富的查询功能
- 支持二进制数据及大型对象
- `可以根据数据的特点替代RDBMS，成为独立的数据库。或者配合RDBMS，存储特定的数据。`

## 行式存储数据库

### 行式数据库

| id   | name | city | age  |
| ---- | ---- | ---- | ---- |
| 1    | 张三 | 北京 | 20   |
| 2    | 李四 | 南昌 | 56   |

数据按行存储的。查询id=3的，查询速度快。统计年龄平均值，速度慢。

### 列式数据库

| id   | name | city | age  |
| ---- | ---- | ---- | ---- |
| 1    | 张三 | 北京 | 20   |
| 2    | 李四 | 南昌 | 56   |

数据按列存储的。查询id=3的，查询速度慢。统计年龄平均值，速度快。

#### Hbase

HBase是Hadoop项目中的数据库。它用于需要对大量的数据进行随机、实时的读写操作的场景中。

HBase的目标就是处理数据量非常庞大的表，可以用普通的计算机处理超过10亿行数据，还可处理有数百万列元素的数据表。

#### Cassandra

Apache Cassandra是一款免费的开源NoSQL数据库，其设计目的在于管理由大量商用服务器构建起来的庞大集群上的`海量数据集(数据量通常达到PB级别)`。在众多显著特性当中，`Cassandra最为卓越的长处是对写入及读取操作进行规模调整，而且其不强调主集群的设计思路能够以相对直观的方式简化各集群的创建与扩展流程。`

## 图关系型数据库

主要应用：社会关系，公共交通网络，地图及网络拓谱`(n*(n-1)/2)`

# Redis概述安装

Ø Redis是一个`开源`的`key-value`存储系统。

Ø 和Memcached类似，它支持存储的value类型相对更多，包括`string(字符串)、list(链表)、set(集合)、zset(sorted set --有序集合)和hash（哈希类型）`。

Ø 这些数据类型都支持push/pop、add/remove及取交集并集和差集及更丰富的操作，而且这些操作都是`原子性`的。

Ø 在此基础上，Redis支持各种不同方式的`排序`。

Ø 与memcached一样，为了保证效率，数据都是`缓存在内存中`。

Ø 区别的是Redis会`周期性`的把更新的`数据写入磁盘`或者把修改操作写入追加的记录文件。

Ø 并且在此基础上实现了`master-slave(主从)`同步。

## 应用场景

### 配合关系型数据库做高速缓存

Ø 高频次，热门访问的数据，降低数据库IO

Ø 分布式架构，做session共享

### 多样的数据结构存储持久化数据

最新N个数据 ------- 通过List实现按自然时间排序的数据

排行榜，Top N ------- 利用zset（有序集合）

时效性的数据，如手机验证码 -------  Expire 过期

计数器，秒杀 ------- 原子性，自增方法INCR，DECR

去除大量数据中的重复数据 -------  利用Set集合

构建队列 -------  利用list集合

发布订阅消息系统 -------  pub/sub模式 

### redis的安装

下载redis-6.2.1.tar.gz放/opt目录

解压命令：tar -zxvf redis-6.2.1.tar.gz

解压完成后进入目录：cd redis-6.2.1

在redis-6.2.1目录下再次执行make命令（只是编译好）

如果没有准备好C语言编译环境，make会报错—Jemalloc/jemalloc.h：没有那个文件

解决方案：运行make distclean

在redis-6.2.1目录下再次执行make命令（只是编译好）

跳过maketest 继续执行: make install

### 安装目录

`/usr/local/bin`

查看默认安装目录：

redis-benchmark:性能测试工具，可以在自己本子运行，看看自己本子性能如何

redis-check-aof：修复有问题的AOF文件，rdb和aof后面讲

redis-check-dump：修复有问题的dump.rdb文件

redis-sentinel：Redis集群使用

redis-server：Redis服务器启动命令

redis-cli：客户端，操作入口

### 前台启动

redis-server

前台启动，命令行窗口不能关闭，否则服务器停止

### 后台启动

备份redis.conf：拷贝一份redis.conf到其他目录 cp  /opt/redis-3.2.5/redis.conf  /myredis

修改配置文件：修改redis.conf(128行)文件将里面的daemonize no 改成 yes，让服务在后台启动

Redis启动：redis-server/myredis/redis.conf

客户端访问：redis-cli，多个端口可以redis-cli-p6379

测试验证：ping

Redis关闭：redis-cli shutdown，也可以进入终端后再关闭。多实例关闭：redis-cli -p 6379 shutdown

### 相关知识介绍

端口6379从何而来：Alessia `M（6）e（3）r（7）z（9）`

redis默认16个数据库，类似数组下标从0开始，初始默认使用0号库使用命令 select  <dbid>来切换数据库。如: select 8 统一密码管理，所有库同样密码。

- `dbsize`查看当前数据库的key的数量
- `flushdb清空当前库`
- `flushall通杀全部库`

Redis是单线程+多路IO复用技术

多路复用是指使用一个线程来检查多个文件描述符（Socket）的就绪状态，比如调用select和poll函数，传入多个文件描述符，如果有一个文件描述符就绪，则返回，否则阻塞直到超时。得到就绪状态后进行真正的操作可以在同一个线程里执行，也可以启动线程执行（比如使用线程池）

----

> 串行  vs  多线程+锁（memcached） vs  `单线程+多路IO复用(Redis)`

多路IO复用。发出需要票的请求，然后去做其他事，黄牛买到票了再通知线程过来拿数据。期间CPU一直在运行，没有停。

<img src="..\pics\redis\3.png">

（与Memcache三点不同: 支持多数据类型，支持持久化，单线程+多路IO复用）

<img src="..\pics\redis\2.png">

# 常用五大数据类型

哪里去获得redis常见数据类型操作命令http://www.redis.cn/commands.html

## key

> 存储

set key value

expire key 10  10秒钟：为给定的key设置过期时间

> 查询

keys *查看当前库所有key   (匹配：keys *1)

get k1 --- 拿到对应的value

exists k1 --- 是否存在这个key

ttl key 查看还有多少秒过期，-1表示永不过期，-2表示已过期

type key 查看你的key是什么类型

dbsize查看当前数据库的key的数量

> 删除

del key    删除指定的key数据

`unlink key  根据value选择非阻塞删除：`仅将keys从keyspace元数据中删除（就是让你查询不到，但是可能没有真的删除），真正的删除会在后续异步操作。

> 其他操作

select命令切换数据库

flushdb清空当前库

flushall通杀全部库

## 字符串

### 简介

String是Redis最基本的类型，你可以理解成与Memcached一模一样的类型，一个key对应一个value。

String类型是`二进制安全`的。意味着Redis的string可以包含任何数据。比如jpg图片或者序列化的对象。

String类型是Redis最基本的数据类型，一个Redis中字符串value最多可以是`512M`

### 常用命令

set <key><value>：添加键值对

*NX：当数据库中key不存在时，可以将key-value添加数据库

*XX：当数据库中key存在时，可以将key-value添加数据库，与NX参数互斥

*EX：key的超时秒数

*PX：key的超时毫秒数，与EX互斥

 

get  <key>：查询对应键值

append  <key><value>：将给定的<value> 追加到原值的末尾

strlen  <key>：获得值的长度

setnx  <key><value>：只有在 key 不存在时   设置 key 的值

 

incr  <key> ：原子性操作。将 key 中储存的数字值增1，只能对数字值操作，如果为空，新增值为1

decr  <key>：原子性操作。将 key 中储存的数字值减1，只能对数字值操作，如果为空，新增值为-1

incrby / decrby  <key><步长>：将 key 中储存的数字值增减。自定义步长。

mset  <key1><value1><key2><value2>：同时设置一个或多个 key-value对  

mget  <key1><key2><key3> ：同时获取一个或多个 value  

msetnx <key1><value1><key2><value2> ：同时设置一个或多个 key-value 对，当且仅当所有给定 key 都不存在。

getrange  <key><起始位置><结束位置>：获得值的范围，类似java中的substring，前包，后包

setrange  <key><起始位置><value>：用 <value>  覆写<key>所储存的字符串值，从<起始位置>开始(索引从0开始)。

setex  <key><过期时间><value>：设置键值的同时，设置过期时间，单位秒。

getset <key><value>：以新换旧，设置了新值同时获得旧值。

### 数据结构

String的数据结构为简单动态字符串(Simple Dynamic String,缩写SDS)。是可以修改的字符串，内部结构实现上类似于Java的ArrayList，采用预分配冗余空间的方式来减少内存的频繁分配.

<img src="..\pics\redis\4.png">

如图中所示，内部为当前字符串实际分配的空间capacity一般要高于实际字符串长度len。当字符串长度小于1M时，扩容都是加倍现有的空间，如果超过1M，扩容时一次只会多扩1M的空间。需要注意的是字符串最大长度为512M。

> SDS的定义

```c
struct sdshdr{
    int len; // 记录已经使用的长度
    int free; // 记录还可使用的长度 总长度 = len + free。不够会扩容。多余的采用惰性空间释放。
    char buff[]; // 存放字符串
}
```

## 列表

### 简介

单键多值

Redis 列表是简单的字符串列表，按照插入顺序排序。你可以添加一个元素到列表的头部（左边）或者尾部（右边）。

它的底层实际是个`双向链表`，对两端的操作性能很高，通过索引下标的操作中间的节点性能会较差。

<img src="..\pics\redis\5.png">

### 常用命令

lpush/rpush  <key><value1><value2><value3>：从左边/右边插入一个或多个值。

- 假设原始链表 head-->1-->2-->3 加入4

- lpush：头插法。head`-->4`-->1-->2-->3 
- rpush：尾插法。head-->1-->2-->3`-->4` 

lpop/rpop  <key>：从左边/右边吐出一个值。值在键在，值光键亡。

rpop/lpush  <key1><key2>从<key1>：列表右边吐出一个值，插到<key2>列表左边。

lrange <key><start><stop>：按照索引下标获得元素(从左到右)；lrange mylist 0 -1  0左边第一个，-1右边第一个，（0-1表示获取所有）

- lrange key1 0 -1：从索引为0开始，拿到尾部。

lindex <key><index>：按照索引下标获得元素(从左到右)

- lindex key 0

llen <key>：获得列表长度 

linsert <key>  before <value><newvalue>在<value>的后面插入<newvalue>插入值

- linsert key before "v11" "v100"：在v11前面加入v100

lrem <key><n><value>：从左边删除n个value(从左到右)

lset<key><index><value>：将列表key下标为index的值替换成value

### 数据结构

List的数据结构为快速链表quickList。

首先在列表元素较少的情况下会使用一块连续的内存存储，这个结构是ziplist，也即是压缩列表。

它将所有的元素紧挨着一起存储，分配的是一块连续的内存。

当数据量比较多的时候才会改成quicklist。

因为普通的链表需要的附加指针空间太大，会比较浪费空间。比如这个列表里存的只是int类型的数据，结构上还需要两个额外的指针prev和next。

<img src="..\pics\redis\6.png">

Redis将链表和ziplist结合起来组成了quicklist。也就是将多个ziplist使用双向指针串起来使用。这样既满足了快速的插入删除性能，又不会出现太大的空间冗余。

