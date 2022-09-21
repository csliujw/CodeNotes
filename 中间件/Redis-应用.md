# 快速回顾

[黑马程序员Redis入门到实战教程，全面透析redis底层原理+redis分布式锁+企业解决方案+redis实战_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1cr4y1671t?p=24)

## 特点

| -        | SQL                                                        | NoSQL                                                        |
| -------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| 数据结构 | 结构化（Structured）                                       | 非结构化                                                     |
| 数据关联 | 关联的（Relational）                                       | 无关联的                                                     |
| 查询方式 | SQL 查询                                                   | 非 SQL                                                       |
| 事务特性 | ACID                                                       | BASE                                                         |
| 存储方式 | 磁盘                                                       | 内存                                                         |
| 扩展性   | 垂直                                                       | 水平                                                         |
| 使用场景 | 1) 数据结构固定<br>2) 相关业务对数据安全性、一致性要求较高 | 1) 数据结构不固定<br>2) 对一致性、安全性要求不高<br>3) 对性能要求高 |

NoSQL 具体有 

- 键值类型（Redis）
- 文档类型（MongoBD）
- 列类型（HBase）
- Graph 类型（Neo4j）

Redis 介绍：诞生于 2009 年，全称是 Remote Dictionary Server，远程字典服务器，是一个基于内存的键值型 NoSQL 数据库。

特点

- key-value 类型，value 支持多种不同的数据结构，功能丰富。
- 单线程，每个命令具备原子性（类似 6.0 的多线程仅仅是在请求处理这块，核心的命令执行仍是当线程的）
- 低延迟，速度快（基于内存、IO 多路复用、良好的编码）
- 支持数据持久化
- 支持主从集群、分片集群
    - 主从集群：从节点可以去备份主节点
    - 分片集群：进行数据拆分，比如把 1T 的数据切分成多分存储在不同的服务器上
    - 支持多语言客户端

## 安装

在 docker 中安装 redis。

docker 中进入后台运行的 redis：docker exec -it ae7c /bin/bash。然后再输入 redis-cli 启动交互模式。

## 数据结构

Redis 是一个 key-value 的数据库，key 一般是 String 类型，不过 value 的类型多种多样。

五大基本类型

- String 类型 =\=\> "hello"
- Hash 类型 =\=\> {name: "Jack", age: 21}
- List 类型 =\=\> [A -> B -> C -> C]
- Set 类型 =\=\> {A, B, C}
- SortedSet 类型 =\=\> {A: 1, B: 2, C: 3}

特殊类型

- GEO =\=\> {A:（120.3， 30.5）}
- BitMap =\=\> 0110110101110101011
- HyperLog =\=\> 0110110101110101011

## 命令

Redis 的命令可以在<a href="https://redis.io/commands">官网</a>查询到，也可以通过在 Redis 客户端输入 `help @命令类型` 如 `help @generic` 查询通用命令的使用方式。

### 通用命令

通用指令是部分数据类型的，都可以使用的指令，常见的有：

- KEYS：查看符合模板的所有 key，<span style="color:blue">不建议在生产环境设备上使用</span>

- DEL：删除一个指定的 key

- EXISTS：判断 key 是否存在
- EXPIRE：给一个 key 设置有效期，有效期到期时该 key 会被自动删除，不设置时间的话默认是永久有效
- TTL：查看一个KEY的剩余有效期（它怎么统计有效期的呢？）

通过 help [command] 可以查看一个命令的具体用法，例如：

<div align="center"><img src="img/image-20220321162710086.png"></div>

### String类型

String 类型，也就是字符串类型，是 Redis 中最简单的存储类型。其 value 是字符串，不过根据字符串的格式不同，又可以分为3类：

- string：普通字符串
- int：整数类型，可以做自增、自减操作
- float：浮点类型，可以做自增、自减操作

不管是哪种格式，底层都是字节数组形式存储，只不过是编码方式不同。字符串类型的最大空间不能超过 512m。

| key   | value       |
| ----- | ----------- |
| msg   | hello world |
| num   | 10          |
| score | 92.6        |

String 的常见命令有：

- SET：添加或者修改已经存在的一个 String 类型的键值对
- GET：根据 key 获取 String 类型的 value
- MSET：批量添加多个 String 类型的键值对
- MGET：根据多个 key 获取多个 String 类型的 value
- INCR：让一个整型的 key 自增1
- INCRBY：让一个整型的 key 自增并指定步长，例如：incrby num 2 让 num 值自增 2
- INCRBYFLOAT：让一个浮点类型的数字自增并指定步长
- SETNX：添加一个 String 类型的键值对，前提是这个 key 不存在，否则不执行（返回 0 表示存在了，未添加成功）
- SETEX：添加一个 String 类型的键值对，并且指定有效期

### Hash类型

Hash 类型，也叫散列，其 value 是一个无序字典，类似于 Java 中的HashMap 结构。

String 结构是将对象序列化为 JSON 字符串后存储，当需要修改对象某个字段时很不方便，而 Hash 结构可以将对象中的每个字段独立存储，可以针对单个字段做 CRUD。

<div align="center"><img src="img/image-20220321163814619.png"></div>

Hash 的常见命令有：

- HSET key field value：添加或者修改 hash 类型 key 的 field 的值
- HGET key field：获取一个 hash 类型 key 的 field 的值
- HMSET：批量添加多个 hash 类型 key 的 field 的值
- HMGET：批量获取多个 hash 类型 key 的 field 的值
- HGETALL：获取一个 hash 类型的 key 中的所有的 field 和 value
- HKEYS：获取一个 hash 类型的 key 中的所有的 field
- HVALS：获取一个 hash 类型的 key 中的所有的 value
- HINCRBY：让一个 hash 类型 key 的字段值自增并指定步长
- HSETNX：添加一个 hash 类型的 key 的 field 值，前提是这个 field 不存在，否则不执行

### List类型

Redis 中的 List 类型与 Java 中的 LinkedList 类似，可以看做是一个双向链表结构。既可以支持正向检索和也可以支持反向检索。

特征也与 LinkedList 类似：

- 有序
- 元素可以重复
- 插入和删除快
- 查询速度一般

常用来存储一个有序数据，例如：朋友圈点赞列表，评论列表等。

List的常见命令有：

- LPUSH key element ... ：向列表左侧插入一个或多个元素
- LPOP key：移除并返回列表左侧的第一个元素，没有则返回nil
- RPUSH key element ... ：向列表右侧插入一个或多个元素
- RPOP key：移除并返回列表右侧的第一个元素
- LRANGE key star end：返回一段角标范围内的所有元素
- <span style="color:green">BLPOP 和 BRPOP：与 LPOP 和 RPOP 类似，只不过在没有元素时会等待指定的时间，而不是直接返回 nil，有点类似于阻塞队列。Timeout 时间是以秒为单位</span>

<div align="center"><img src="img/image-20220321164231779.png"></div>

可以用 List 结构模拟 stack、queue 和 blockingkqueue。

### Set类型

Redis 的 Set 结构与 Java 中的 HashSet 类似，可以看做是一个 value 为 null 的 HashMap。因为也是一个 hash 表，因此具备与 HashSet 类似的特征：

- 无序
- 元素不可重复
- 查找快
- 支持交集、并集、差集等功能（寻找共同好友）

String 的常见命令有：

- SADD key member ... ：向 set 中添加一个或多个元素
- SREM key member ... : 移除 set 中的指定元素
- SCARD key： 返回 set 中元素的个数
- SISMEMBER key member：判断一个元素是否存在于 set 中
- SMEMBERS：获取 set 中的所有元素
- lSINTER key1 key2 ... ：求 key1 与 key2 的交集 S1，S2 的交（B、C）
- lSDIFF key1 key2 ... ：求 key1 与 key2 的差集，S1，S2 的差（A）
- lSUNION key1 key2 ..：求 key1 和 key2 的并集，S1，S2 的并（A、B、C、D）

<div align="center"><img src="img/image-20220321164621649.png"></div>

> 练习

将下列数据用 Redis 的 Set 集合来存储：

- 张三的好友有：李四、王五、赵六
- 李四的好友有：王五、麻子、二狗

利用Set的命令实现下列功能：

- 计算张三的好友有几人
- 计算张三和李四有哪些共同好友
- 查询哪些人是张三的好友却不是李四的好友
- 查询张三和李四的好友总共有哪些人
- 判断李四是否是张三的好友
- 判断张三是否是李四的好友
- 将李四从张三的好友列表中移除

### SortedSet类型

Redis 的 SortedSet 是一个可排序的 set 集合，与 Java 中的 TreeSet 有些类似，但底层数据结构却差别很大。SortedSet 中的每一个元素都带有一个 score 属性，可以基于 score 属性对元素排序，底层的实现是一个跳表（SkipList，用来排序的）加 hash 表。

SortedSet 具备下列特性：

- 可排序
- 元素不重复
- 查询速度快

因为 SortedSet 的可排序特性，经常被用来实现排行榜这样的功能。

SortedSet 的常见命令有：默认都是升序排名。

- ZADD key score member：添加一个或多个元素到 sorted set ，如果已经存在则更新其 score 值

- ZREM key member：删除 sorted set 中的一个指定元素

- ZSCORE key member : 获取 sorted set 中的指定元素的 score 值

- ZRANK key member：获取 sorted set 中的指定元素的排名
- ZCARD key：获取 sorted set 中的元素个数
- ZCOUNT key min max：统计 score 值在给定范围内的所有元素的个数
- ZINCRBY key increment member：让 sorted set 中的指定元素自增，步长为指定的 increment 值，比如自增 increment（自增1）。
- ZRANGE key min max：按照 score 排序后，获取指定排名范围内的元素 （zrange age 0 9 ，取排名前 10 的元素）
- ZRANGEBYSCORE key min max：按照 score 排序后，获取指定 score 范围内的元素
- ZDIFF、ZINTER、ZUNION：求差集、交集、并集

注意：所有的排名默认都是升序，如果要降序则在命令的 Z 后面添加REV 即可：ZREVRange 就是降序了

> 练习

将班级的下列学生得分存入 Redis 的 SortedSet 中：

- Jack 85, Lucy 89, Rose 82, Tom 95, Jerry 78, Amy 92, Miles 76

并实现下列功能：

- 删除 Tom 同学
- 获取 Amy 同学的分数
- 获取 Rose 同学的排名
- 查询 80 分以下有几个学生
- 给Amy同学加 2 分
- 查出成绩前 3 名的同学
- 查出成绩 80 分以下的所有同学

## 设计KEY

Redis 没有类似 MySQL 中的 Table 的概念，我们该如何区分不同类型的key呢？例如，需要存储用户、商品信息到 redis，有一个用户 id 是1，有一个商品 id 恰好也是 1。

<span style="color:orange">可以采用 `项目名:业务名:类型:id` 这种多个单词形成层级结构，单词间用 ':' 隔开的方式设计 key。</span>

- <span style="color:orange">user 相关的 key：`project_name:user:1`</span>

- <span style="color:orange">product 相关的 key：`project_name:product:1`</span>

在 redis 内部，最后会以层级关系显示这些数据。

如果 Value 是一个 Java 对象，如一个 User 对象，则可以将对象序列化为 JSON 字符串后存储。

| key       | value                                     |
| --------- | ----------------------------------------- |
| user:1    | {"id":1, "name": "Jack", "age": 21}       |
| product:1 | {"id":1, "name": "小米11", "price": 4999} |

`Redis 的 key 的格式=\=\>[项目名]:[业务名]:[类型]:[id]`

## Java客户端

在 Redis 官网中提供了各种语言的客户端，地址：https://redis.io/clients

<div align="center"><img src="img/image-20220321165204979.png"></div>

### Jedis

① 引入依赖

```xml
<!-- jedis 客户端-->
<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
</dependency>
```

② 建立连接

```java
jedis = new Jedis("192.168.160.44", 6379);
```

③ 使用 Jedis

```java
jedis.set("name", "hahah");
```

④ 释放资源

```java
jedis.close();
```

完整代码

```java
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.springframework.boot.test.context.SpringBootTest;
import redis.clients.jedis.Jedis;

@SpringBootTest
public class TestJedis {
    private Jedis jedis;

    @Before
    public void setUp() {
        jedis = new Jedis("192.168.160.44", 6379);
    }

    @Test
    public void test() {
        jedis.set("name", "hahah");
    }

    @After
    public void close() {
        jedis.close();
    }
}
```

Jedis 本身是线程不安全的，并且频繁的创建和销毁连接会有性能损耗，因此推荐使用 Jedis 连接池代替 Jedis 的直连方式。

```java
// Jedis 连接池
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

public class JedisUtils {
    private static final JedisPool jedisPool;

    static {
        JedisPoolConfig config = new JedisPoolConfig();
        config.setMaxIdle(10);
        config.setMinIdle(5);
        config.setMaxTotal(20);
        config.setMaxWaitMillis(1000);
        jedisPool = new JedisPool(config, "192.168.2.240");
    }

    public static Jedis getJedis() {
        return jedisPool.getResource();
    }

    public static void close(Jedis jedis){
        jedis.close(); // 有连接池则归还到池中，没有则直接关闭
    }
}
```

```java
@SpringBootTest
public class TestJedis {
    private Jedis jedis;

    @Before
    public void setUp() {
        jedis = JedisUtils.getJedis();
    }

    @Test
    public void test() {
        jedis.set("name", "hahah");
        String name = jedis.get("name");
        System.out.println(name);
    }

    @After
    public void close() {
        jedis.close();
    }
}
```

这里解释下为什么说 Jedis 是线程不安全的。 

[jedis非线程安全 - 简书 (jianshu.com)](https://www.jianshu.com/p/5e4a1f92c88f)

简单说就是，Jedis 共享了 Socket。Jedis 类中有 RedisInputStream和 RedisOutputStream 两个属性，而发送命令和获取返回值都是使用这两个成员变量，这很容易引发多线程问题。Jedis 执行命令前需要创建一个 conneciton 连接：

```java
public void connect() {
    // 如果 socket 连接关闭了的话，创建一个新的连接。
    if (!this.isConnected()) {
        try {
            this.socket = this.jedisSocketFactory.createSocket();
            this.outputStream = new RedisOutputStream(this.socket.getOutputStream());
            this.inputStream = new RedisInputStream(this.socket.getInputStream());
        } catch (IOException var2) {
            this.broken = true;
            throw new JedisConnectionException("Failed connecting to " + this.jedisSocketFactory.getDescription(), var2);
        }
    }
}
```

可能两个线程出现这种情况：

```mermaid
sequenceDiagram
participant T1 as 线程1
participant T2 as 线程2
participant S as Socket
T1->>S:进入了if (!this.isConnected())花括号里
T1-->>T2:线程上下文切换,线程2开始执行
T2->>S:也进入了if (!this.isConnected())花括号里
T2-->>T1:线程上下文切换,线程1继续执行
T1->>S:创建了 socket,准备执行InputStream方法,接收redis的响应了
T1-->>T2:线程上下文切换,线程2开始执行
T2->>S:创建了socket,但是还没有初始化连接
T2-->>T1:线程上下文切换,线程1继续执行
T1-->S:线程1用未connection的连接获取InputStream出错
```

### SpringDataRedis

SpringData 是 Spring 中数据操作的模块，包含对各种数据库的集成，其中对 Redis 的集成模块就叫做 SpringDataRedis

官网地址：https://spring.io/projects/spring-data-redis

- 提供了对不同 Redis 客户端的整合（Lettuce 和 Jedis）
- 提供了 RedisTemplate 统一 API 来操作 Redis
- 支持 Redis 的发布订阅模型
- 支持 Redis 哨兵和 Redis 集群
- 支持基于 Lettuce 的响应式编程
- 支持基于 JDK、JSON、字符串、Spring 对象的数据序列化及反序列化
- 支持基于 Redis 的 JDKCollection 实现

> 快速入门

SpringDataRedis 中提供了 RedisTemplate 工具类，其中封装了各种对 Redis 的操作。并且将不同数据类型的操作 API 封装到了不同的类型中：

| API                                | 返回值类型      | 说明                    |
| ---------------------------------- | --------------- | ----------------------- |
| <b>redisTemplate</b>.opsForValue() | ValueOperations | 操作 String 类型数据    |
| <b>redisTemplate</b>.opsForHash()  | HashOperations  | 操作 Hash 类型数据      |
| <b>redisTemplate</b>.opsForList()  | ListOperations  | 操作 List 类型数据      |
| <b>redisTemplate</b>.opsForSet()   | SetOperations   | 操作 Set 类型数据       |
| <b>redisTemplate</b>.opsForZSet()  | ZSetOperations  | 操作 SortedSet 类型数据 |
| <b>redisTemplate</b>               |                 | 通用的命令              |

① 引入一来就，redis 启动的坐标和连接池依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
<dependency>
    <groupId>org.apache.commons</groupId>
    <artifactId>commons-pool2</artifactId>
</dependency>
```

② redis 相关配置文件

```yaml
spring:
  redis:
    host: 192.168.160.44
    port: 6379
    lettuce:
      pool:
        max-active: 10
        max-idle: 10
        min-idle: 1
        time-between-eviction-runs: 10s
```

③ 注入 RedisTemplate

```java
@Autowired
RedisTemplate redisTemplate;
```

④ 测试

```java
@SpringBootTest(classes = HmDianPingApplication.class)
@SuppressWarnings("all")
@RunWith(SpringRunner.class) // JUnit4 需要加这句
public class RedisTemplateTest {
    @Autowired
    RedisTemplate redisTemplate;

    @Test
    public void test() {
        // 插入一条string类型数据，如果采用默认的序列化机制的话，
        // name 会被当成一个 Java 对象进行序列化，
        // 默认采用 JDK 序列化，序列化为字节形式。
        redisTemplate.opsForValue().set("name", "hello-world");
        redisTemplate.opsForValue().setIfAbsent("name", "hello");
    }
}
```

>序列化方式

RedisTemplate 可以接收任意 Object 作为值写入 Redis，只不过写入前会把 Object 序列化为字节形式，默认是采用 JDK 序列化，得到的结果是这样的

<div align="center"><img src="img/image-20220321221236720.png"></div>

Spring 默认提供了一个 StringRedisTemplate 类，它的 key 和 value 的序列化方式默认就是 String 方式。省去了我们自定义 RedisTemplate 的过程，我们不用自己定义序列化反序列化的配置参数了。

```java
@SpringBootTest
@RunWith(SpringRunner.class) // JUnit4 需要加这句
public class RedisTemplateTest {
    @Autowired
    StringRedisTemplate redisTemplate;

    @Test
    public void test() {
        redisTemplate.opsForValue().set("name", "hello-world");
        redisTemplate.opsForValue().setIfAbsent("name", "hello");
    }
}
```

# 场景实战

[黑马程序员Redis入门到实战教程，全面透析redis底层原理+redis分布式锁+企业解决方案+redis实战_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1cr4y1671t?p=24)

## 内容概述

- 短信登录：Redis 共享 session 的应用
- 商品缓存查询：缓存使用技巧；缓存雪崩、穿透等问题
- 优惠券秒杀：Redis 计数器、Lua 脚本实现 Redis 分布式锁、Redis 的三种消息队列
- 达人探店：基于 List 的点赞列表、基于 SortedSet 的点赞排行榜
- 好友关注：基于 Set 集合的关注、取关、共同关注、消息推送等功能
- 附件的商户：Redis 的 GenHash 的应用
- 用户签到：Redis 的 BitMap 数据统计功能
- UV 统计：Redis 的 HyperLogLog 的统计功能（一般大数据量下的话不会用这个）

导入 SQL 文件

<div align="center"><img src="img/image-20220507205857248.png"></div>

SQL 中包含如下的表

- tb_user：用户表
- tb_user_info：用户详情表
- tb_shop：商户信息表
- tb_shop_type：商户类型表
- tb_blog：用户日记表（达人探店日记）
- tb_follow：用户关注表
- tb_voucher：优惠券表
- tb_voucher_order：优惠券的订单表

项目的运行流程如下：

这是一个前后端分离的项目，用 ngnix 进行方向代理，访问 tomcat 集群。Redis 和 MySQL 也有相应的集群。 

<div align="center"><img src="img/image-20220507210207656.png"></div>

项目的访问路径是：http://localhost:8081/shop-type/list，如果可以看到数据则证明运行没有问题。

PS：需要修改 application.yaml 文件中的 MySQL、Redis 的地址信息。资料中提供了一个 nginx，复制到任意目录，在控制台输入 `start nginx.exe` 启动 nginx。然后访问: [http://127.0.0.1:8080](http://127.0.0.1:8080/) ，即可看到页面。

## 短信登录

### 基于Session实现登录

具体的流程如下。

<div align="center"><img src="img/image-20220507210751764.png"></div>

我们需要在一些操作上判断用户是否登录。一种方式是在特定的操作上加判断，另一种方式是写一个过滤器，当访问特定的 url 路径时，判断是否登录。显然，用过滤器的方式更好。我们需要把拦截器拦截到的用户信息传递到 Controller 里去，并且要保证线程安全（一条完整的 HTTP 请求对应一个线程，如果共享数据，如用同一个 ArrayList 存储，会出现并发安全问题）。我们可以使用 ThreadLocal，可以确保线程之间安全的使用自己线程中的数据。

拦截器的两个方法

- preHandler 在业务处理器处理请求之前被调用。
- afterCompletion 完全处理请求之后被调用，可用于清理资源。

### 集群的session共享问题

<b>session 共享问题</b>：多台 Tomcat 并不共享 session 存储空间，当请求切换到不同 Tomcat 服务时导致数据丢失的问题。Tomcat 后面虽然提供了 session 拷贝的解决办法，但是太浪费内存空间了，数据拷贝是有延迟的。

session的替代方案应该满足：

- 数据共享
- 内存存储
- key、value 结构
- ---> redis

这里，我们使用 Redis 充当缓存，解决 Session 共享问题。

### 基于Redis实现共享session登录

<div align="center"><img src="img/image-20220507215112295.png"></div>



<div align="center"><img src="img/image-20220507215201598.png"></div>

注意，拦截器类是我们手动 new 出来放到 IOC 里面的，所以拦截器里注入 StringRedisTemplate 需要用构造函数手动给对象赋值。然后添加自定义的拦截器又是在 Configurate 类中做的， StringRedisTemplate 可以通过依赖注入实例化。

保存登录的用户信息，可以用 String 结构，以 JSON 字符串来保存，比较直观。

| key         | value                   |
| ----------- | ----------------------- |
| shop:user:1 | {name: "Jack", age: 21} |
| shop:user:2 | {name: "Pose", age: 18} |

哈希结构可以将对象中的每个字段独立存储，可以针对单个字段做 CRUD，并且内存占用更少

<div align="center"><img src="img/image-20220507215602774.png"></div>

### 总结

Redis 代替 session 需要考虑的问题：

- 选择合适的数据结构
- 选择合适的 key
- 选择合适的存储粒度

登录拦截器的优化

拦截器的执行顺序有个先后关系，可以通过 Order 来设置。Order 的值越小，执行计划等级越高。

<div align="center"><img src="img/image-20220507215858667.png"></div>

## 商品查询缓存

- 添加 Redis 缓存
- 缓存更新策略
- 缓存穿透
- 缓存雪崩
- 缓存击穿
- 缓存工具封装

### 缓存

<b>缓存</b>就是数据交换的缓冲区（称作Cache [ kæʃ ] ），是存贮数据的临时地方，一般读写性能较高。

```mermaid
graph LR
浏览器缓存-->应用层缓存-->数据库缓存-->CPU缓存
```

- 缓存的作用：
    - 降低后端负载
    - 提高读写效率，降低响应时间
- 缓存的成本
    - 数据一致性成本
    - 代码维护成本
    - 运维成本

### 添加Redis缓存

<div align="center"><img src="img/image-20220507220302647.png"></div>


<div align="center"><img src="img/image-20220507220326256.png"></div>

修改 ShopTypeController 中的 queryTypeList 方法，添加查询缓存

<div align="center"><img src="img/image-20220507220401217.png"></div>

### 缓存更新策略

|    -     | 内存淘汰                                                     | 超时剔除                                                     | 主动更新                                     |
| :------: | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
|   说明   | 不用自己维护，利用Redis的内存淘汰机制，当内存不足时自动淘汰部分数据。下次查询时更新缓存。 | 给缓存数据添加TTL时间，到期后自动删除缓存。下次查询时更新缓存。 | 编写业务逻辑，在修改数据库的同时，更新缓存。 |
|  一致性  | 差                                                           | 一般                                                         | 好                                           |
| 维护成本 | 无                                                           | 低                                                           | 高                                           |

业务场景：

- 低一致性需求：使用内存淘汰机制。例如店铺类型的查询缓存
- 高一致性需求：主动更新，并以超时剔除作为兜底方案。例如店铺详情查询的缓存

#### 主动更新策略

- <b>Cache Aside Pattern</b>：由缓存的调用这，在更新数据库的同时更新缓存。【推荐这个】
- <b>Read/Write Through Pattern</b>：缓存与数据库整合为一个服务，由服务来维护一致性。调用者调用该服务，无需关系缓存一致性问题。
- <b>Write Behind Caching Pattern</b>：调用者只操作缓存，由其他线程异步的将缓存数据持久化到数据库，保证最终一致性。

操作缓存和数据库时有三个问题需要考虑：

操作缓存和数据库时有三个问题需要考虑：

1.删除缓存还是更新缓存？

- 更新缓存：每次更新数据库都更新缓存，无效写操作较多
- 删除缓存：更新数据库时让缓存失效，查询时再更新缓存

2.如何保证缓存与数据库的操作的同时成功或失败？

- 单体系统，将缓存与数据库操作放在一个事务
- 分布式系统，利用 TCC 等分布式事务方案

3.先操作缓存还是先操作数据库？

- 先删除缓存，再操作数据库
- 先操作数据库，再删除缓存

先操作数据库，再删除缓存。√

先删除缓存，再操作数据库。×

缓存更新策略的最佳实践方案：

1.低一致性需求：使用Redis自带的内存淘汰机制

2.高一致性需求：主动更新，并以超时剔除作为兜底方案

- 读操作：
    - 缓存命中则直接返回
    - 缓存未命中则查询数据库，并写入缓存，设定超时时间

- 写操作：
    - 先写数据库，然后再删除缓存
    - 要确保数据库与缓存操作的原子性

> 给查询商铺的缓存添加超时剔除和主动更新的策略

修改 ShopController 中的业务逻辑，满足下面的需求：

① 根据 id 查询店铺时，如果缓存未命中，则查询数据库，将数据库结果写入缓存，并设置超时时间

② 根据 id 修改店铺时，先修改数据库，再删除缓存

### 缓存穿透

<b>缓存穿透是指客户端请求的数据在缓存中和数据库中都不存在，这样缓存永远不会生效，这些请求都会打到数据库。</b>

常见的解决方案有两种：

1️⃣缓存空对象

- 优点：实现简单，维护方便
- 缺点：额外的内存消耗；可能造成短期的不一致

2️⃣布隆过滤

- 优点：内存占用较少，没有多余 key
- 缺点：实现复杂；存在误判可能

<div align="center"><img src="img/image-20220607115533210.png"></div>

> 缓存穿透产生的原因是什么？

用户请求的数据在缓存中和数据库中都不存在，不断发起这样的请求，给数据库带来巨大压力

> 缓存穿透的解决方案有哪些？

<div align="center"><img src="img/image-20220607115909900.png"></div>

- 缓存 null 值：会带来额外的内存存储压力和短暂的不一致性。
- 布隆过滤：基于布隆算法，准确性无法保证，实现复杂。
- 增强 id 的复杂度，避免被猜测 id 规律
- 做好数据的基础格式校验
- 加强用户权限校验
- <b>做好热点参数的限流</b>

### 缓存雪崩

缓存雪崩是指在同一时段大量的缓存 key 同时失效或者 Redis 服务宕机，导致大量请求到达数据库，带来巨大压力。

<b>常见的解决方案有：</b>

1️⃣给不同的 Key 的 TTL 添加随机值，比如原先设置的 20，那么我们再加一个随机的 1~10 的值

2️⃣利用 Redis 集群提高服务的可用性（很严重），尽可能的避免宕机。宕机的话，可以用 redis 的哨兵机制，监控服务。搭建 Redis 服务形成主从，如果有机器 宕机（如主宕机了，会随机从一个从机器里选一个替代主）

3️⃣给缓存业务添加降级限流策略

4️⃣给业务添加多级缓存

### 缓存穿透

<span style="color:red">缓存击穿问题也叫热点 Key 问题，就是一个被高并发访问并且缓存重建业务较复杂的 key 突然失效了，无数的请求访问会在瞬间给数据库带来巨大的冲击。</span>

假定有多个线程（1，2，3，4）查询到了一个失效的缓存，线程 1 发现缓存失效了，开始走重建缓存的流程。此时缓存还未重建完毕，线程 2，3，4 也发现缓存失效了，也走了重建缓存的流程（没有必要）。

<div align="center"><img src="img/image-20220607145640360.png"></div>

常见的解决方案有：互斥锁和逻辑过期。利用互斥锁，让其中一个线程重建缓存即可了。

<div align="center"><img src="img/image-20220607150022962.png"></div>

互斥锁需要互相等待，1000 个线程来了，一个负责重建，其他的只能等待。如果重建时间长的话，系统性能会很差。可以为数据设置一个逻辑过期，解决等待的问题。

<div align="center"><img src="img/image-20220607150345908.png"></div>

| 解决方案        | 优点                                         | 缺点                                         |
| --------------- | -------------------------------------------- | -------------------------------------------- |
| <b>互斥锁</b>   | 没有额外的内存消耗<br>保证一致性<br>实现简单 | 线程需要等待，性能受影响<br>可能有死锁风险   |
| <b>逻辑过期</b> | 线程无需等待，性能较好                       | 不保证一致性<br>有额外的内存消耗<br>实现复杂 |

> 基于互斥锁解决缓存击穿问题

需求：修改根据 id 查询商铺的业务，基于互斥锁（redis 的 setnx）方式来解决缓存击穿问题。

<div align="center"><img src="img/image-20220607151014185.png"></div>

```java
private boolean tryLock(String key){
    Boolean flag = stringRedisTemplate.opsForValue().setIfAbsent(key,"1",10,TimeUnit.SECONDS);
    return flag==null?false:true;
}

private void unlock(String key){
    stringRedisTemplate.delete(key);
}
```

获取锁成功后，应该再次检测 redis 缓存是否存在，做 doubleCheck，如果存在则无需重建缓存。

> 基于逻辑过期方式解决缓存击穿问题

需求：修改根据 id 查询商铺的业务，基于互斥锁方式来解决缓存击穿问题。

<div align="center"><img src="img/image-20220607151228929.png"></div>

### 缓存工具封装

<b>基于 StringRedisTemplate 封装一个缓存工具类，满足下列需求：</b>

方法 1：将任意 Java 对象序列化为 json 并存储在 string 类型的 key 中，并且可以设置 TTL 过期时间

方法 2：将任意 Java 对象序列化为 json 并存储在 string 类型的 key 中，并且可以设置逻辑过期时间，用于处理缓存击穿问题

方法 3：根据指定的 key 查询缓存，并反序列化为指定类型，利用缓存空值的方式解决缓存穿透问题

方法 4：根据指定的 key 查询缓存，并反序列化为指定类型，需要利用逻辑过期解决缓存击穿问题

在更新缓存的时候，不知道数据库查询的具体操作，采用 lambda 由用户传入具体的操作。

## 优惠券秒杀

- 全局唯一 ID
- 实现优惠券秒杀下单
- 超卖问题
- 一人一单
- 分布式锁
- Redis 优化秒杀
- Redis 消息队列实现异步秒杀

### 全局唯一 ID

每个店铺都可以发布优惠券

当用户抢购时，就会生成订单并保存到 tb_voucher_order 这张表中，而订单表如果使用数据库自增ID就存在一些问题：

- id 的规律性太明显
- 受单表数据量的限制

全局 ID 生成器，是一种在分布式系统下用来生成全局唯一ID的工具，一般要满足下列特性：唯一性、高可用、高性能、递增性、安全性

为了增加 ID 的安全性，我们不直接使用 Redis 自增的数值，而是拼接一些其它信息

```mermaid
graph
符号位-1bit
时间戳-31bit
序列化-32bit
```

ID 的组成部分

- 符号位：1bit，永远为 0
- 时间戳：31bit，以秒为单位，可以使用 69 年
- 序列号：32bit，秒内的计数器，支持每秒产生 $2^{32}$ 个不同 ID `A << COUNT_BIT | count`

> 全局唯一 ID 生成策略

- UUID
- Redis 自增
- snowflake 算法
- 数据库自增

> Redis 自增 ID 策略：

- 每天一个 key，方便统计订单量
- ID 构造是时间戳 + 计数器

> 案例：生成全局唯一 ID

```java
private static final int COUNT_BIT = 32;
public long nextId(String keyPrefix){
    // 1.生成时间戳
    LocalDateTime now = LocalDateTime.now();
    long nowSecond = new.toEpochSecond(ZoneOffset.UTC);
    long timestamp = nowSecond - BEGIN_TIMESTAMP;
    
    // 2.生成序列号
    // 2.1 获取当前日期，精确到天
    String date = now.format(DateTimeFormatter.ofPattern("yyy:MM:dd"));
    // 2.2 自增长（不会存在空指针的，如果key不存在会自动给你重建一个，然后+1返回1给你）
    Long count = stringRedisTemplate.opsForValue().increment("icr:"+keyPrefix+":"+date);
    
    // 拼接返回
   	// 符号位 时间戳 序列化
    // 用位运算。
    return (timestamp<<COUNT_BIT)|count
}
```

### 优惠券秒杀下单

每个店铺都可以发布优惠券，分为平价券和特价券。平价券可以任意购买，而特价券需要秒杀抢购：

tb_voucher：优惠券的基本信息，优惠金额、使用规则等
tb_seckill_voucher：优惠券的库存、开始抢购时间，结束抢购时间。特价优惠券才需要填写这些信息

下单时需要判断两点：

- 秒杀是否开始或结束，如果尚未开始或已经结束则无法下单
- 库存是否充足，不足则无法下单

<div align="center"><img src="img/image-20220607153644701.png"></div>

### 超卖问题

超卖问题是典型的多线程安全问题，针对这一问题的常见解决方案就是加锁。

认为线程安全问题一定会发生，因此在操作数据之前先获取锁，确保线程串行执行。例如 Synchronized、Lock 都属于悲观锁

认为线程安全问题不一定会发生，因此不加锁，只是在更新数据时去判断有没有其它线程对数据做了修改。如果没有修改则认为是安全的，自己才更新数据。如果已经被其它线程修改说明发生了安全问题，此时可以重试或异常。

> 乐观锁

乐观锁的关键是判断之前查询得到的数据是否有被修改过，常见的方式有两种，版本号法和 CAS。

版本号法，加一个版本字段，先查下数据的库存和版本，修改时判断版本是否一致，不一致说明被改过，重新查询库存，修改库存。可以直接用库存作为版本号。直接判断版本号是否一致的话，商品卖出的成功率太低，直接判断库存是否大于零更合适。

有时候只能通过数据是否变化来判断是否可以卖出，这种时候为了提高并发度可以采用分段锁的思想。将数据分为多个段，让多个线程可以并发处理。

> 悲观锁

加重量级锁

如果觉得并发粒度不够，可以采用分段锁的思想。

| 方案                                             | 优点     | 缺点               |
| ------------------------------------------------ | -------- | ------------------ |
| 悲观锁，添加同步锁，让线程串行执行               | 简单粗暴 | 性能一般           |
| 乐观锁，不加锁，在更新时判断是否有其它线程在修改 | 性能好   | 存在成功率低的问题 |

### 一人一单

修改秒杀业务，要求同一个优惠券，一个用户只能下一单。可以直接用联合主键完成，也可以代码里写逻辑。

- 查询订单
- 判断是否存在，不存在则扣减库存，创建订单。

查询订单和扣减库存这块会出现并发问题，因为查询和创建不是原子性的。在加锁的时候需要注意锁的范围和事务的范围。

<div align="center"><img src="img/image-20220607154427229.png"></div>

```java
@Transactional
public Result createVoucherOrder(Long voucherId){
    Long userId = UserHolder.getUser().getId();
    synchronized(userId.toString().intern()){
        // 业务代码
    }
    // 会出现，锁已经释放了，但是事务还没有提交。数据没有刷新到数据库中，导致其他事务查询到的还是未更新的数据，导致数据不一致。
    // 为了避免这种问题，需要加大锁的范围
}
```

加大锁的范围

```java
@Transactional
public Result createVoucherOrder(Long voucherId){
    Long userId = UserHolder.getUser().getId();
    // 业务代码
}

// 加大锁的范围,让锁锁住事务 Spring 事务失效的集中情况
public XX KK(){
    // some code
    Long userId = UserHolder.getUser().getId();
    // Spring 的事务是通过代理对象实现的，直接调用 createVoucherOrder 是不会走代理对象的，所以我们需要改成用代理对象调用
    synchronized(userId.toString().intern()){
    	IVoucherOrderService proxy = AopContext.currentProxy();
        return proxy.createVoucherOrder(userId); // createVoucherOrder 是接口 IVoucherOrderService 中的方法
    }
}
```

通过加锁可以解决在单机情况下的一人一单安全问题，但是在集群模式下就不行了。假定下面两个线程访问的是不同的服务器：

<div align="center"><img src="img/image-20220607154606585.png"></div>

模拟集群模型：

1.将服务启动两份，端口分别为 8081 和 8082

<div align="center"><img src="img/image-20220607203735703.png"></div>

2.修改 nginx 的 conf 目录下的 nginx.conf 文件，配置反向代理和负载均衡。

<div align="center"><img src="img/image-20220607203729101.png"></div>

3.jmeter 并发访问，出现并发安全问题，并未实现一人一单。

分布式锁可以解决上述问题。当然也可以直接用数据库的联合主键解决一人一单的问题。

### 分布式锁

分布式锁是满足分布式系统或集群模式下多进程可见并且互斥的锁。

分布式锁的核心是实现多进程之间互斥，而满足这一点的方式有很多，常见的有三种

|        | **MySQL**                   | **Redis**                 | **Zookeeper**                    |
| ------ | --------------------------- | ------------------------- | -------------------------------- |
| 互斥   | 利用 mysql 本身的互斥锁机制 | 利用 setnx 这样的互斥命令 | 利用节点的唯一性和有序性实现互斥 |
| 高可用 | 好                          | 好                        | 好                               |
| 高性能 | 一般                        | 好                        | 一般                             |
| 安全性 | 断开连接，自动释放锁        | 利用锁超时时间，到期释放  | 临时节点，断开连接自动释放       |

> 基于 Redis 的分布式锁

在代码中，我们为了实现分布式锁时定义了两个基本方法：

- 获取锁方法：

    - 互斥，确保只能有一个线程获取锁

        `setnx lock thread1`，并通过 `expire lock 10` 添加锁过期时间，避免服务器宕机引起死锁

    - 非阻塞：尝试一次，成功返回 true，失败返回 false


- 释放锁方法：

    - 手动释放
    - 超时释放：获取锁时添加一个超时时间 `del key`


```java
public interface ILock{
    // timeoutsec 表示锁的持有时间，过期后自动释放。
    boolean tryLock(long timeoutsec){};
    void unlock();
}
```

<div align="center"><img src="img/image-20220607155325168.png"></div>

上述的方案存在一个问题，如果两个线程争抢锁，线程 1 抢到了，但是由于执行业务的时间太长了，致使锁超时释放。此时线程2 拿到了锁执行业务。在线程 2 执行业务的时候，线程 1 业务执行完毕了，释放了锁（释放了线程 2 加的锁），线程 3 在线程 2 为完成业务，且锁未超时的情况下拿到了锁。

<div align="center"><img src="img/image-20220613171618902.png"></div>

为了解决这个问题，我们需要在释放锁之前判断一下，是不是自己加的锁，是自己加的锁才要释放。

> 改进 Redis 的分布式锁

需求：修改之前的分布式锁实现，满足：

- 在获取锁时存入线程标示（可以用 UUID 表示）
- 在释放锁时先获取锁中的线程标示，判断是否与当前线程标示一致
    - 如果一致则释放锁
    - 如果不一致则不释放锁


<span style="color:red">但是这种方案仍然存在问题</span>

判断锁标识和释放锁不是原子性的，会有并发问题。线程 1 判断锁标识，发现一致，在释放锁时发送了阻塞（如 GC）。在阻塞过程中，其他服务器的线程 2 获取到了锁，并开始执行业务。在线程 2 执行业务时，线程 1 把线程 2 的锁释放了。如果有其他线程也来抢锁，是可以拿到锁的。

<div align="center"><img src="img/image-20220613172756337.png"></div>

### Lua 脚本

<span style="color:orange">前面的核心问题在与，判断锁和释放锁不是原子性的操作，而 Lua 脚本可以解决这种问题。</span>

Redis 提供了 Lua 脚本功能，在一个脚本中编写多条 Redis 命令，确保多条命令执行时的原子性。Lua 是一种编程语言，它的基本语法可以参考网站：https://www.runoob.com/lua/lua-tutorial.html
这里重点介绍 Redis 提供的调用函数，语法如下：

```shell
# 执行redis命令
redis.call('命令名称', 'key', '其它参数', ...)
```

例如，我们要执行 set name jack，则脚本是这样

```shell
# 执行 set name jack
redis.call('set', 'name', 'jack')
```

例如，我们要先执行 set name Rose，再执行 get name，则脚本如下

```shell
# 先执行 set name jack
redis.call('set', 'name', 'jack')
# 再执行 get name
local name = redis.call('get', 'name')
# 返回
return name
```

写好脚本以后，需要用 Redis 命令来调用脚本，调用脚本的常见命令如下

<div align="center"><img src="img/image-20220611203037898.png"></div>

例如，我们要执行 redis.call('set', 'name', 'jack') 这个脚本，语法如下

<div align="center"><img src="img/image-20220611203428604.png"></div>

如果脚本中的 key、value 不想写死，可以作为参数传递。key 类型参数会放入 KEYS 数组，其它参数会放入 ARGV 数组，在脚本中可以从 KEYS 和 ARGV 数组获取这些参数：

```shell
eval "return redis.call('set', KEYS[1],ARGV[1])" 1 name rose
```

### Lua 改进分布式锁

释放锁的业务流程是这样的：

- 1.获取锁中的线程标示
- 2.判断是否与指定的标示（当前线程标示）一致
- 3.如果一致则释放锁（删除）
- 4.如果不一致则什么都不做
- 5.如果用 Lua 脚本来表示则是这样的：

```mysql
-- 这里的 KEYS[1] 就是锁的key，这里的ARGV[1] 就是当前线程标示
-- 获取锁中的标示，判断是否与当前线程标示一致
if (redis.call('GET', KEYS[1]) == ARGV[1]) then
  -- 一致，则删除锁
  return redis.call('DEL', KEYS[1])
end
-- 不一致，则直接返回
return 0
```

需求：基于 Lua 脚本实现分布式锁的释放锁逻辑
提示：RedisTemplate 调用 Lua 脚本的 API 如下：

```java
/**
 script：脚本
 keys：对应 KEYG
 args：对应 ARGV
*/
public <T> execute(RedisScript<T> script, List<K> kyes, Object... args){
    return scriptExecutor.execute(script,keys,args);
}
```

假定，我们将 lua 脚本放在 resources 根目录下

```java
private static final DefaultRedisScript<Long> UNLOCK_SCRIPT;
statIC{
    UNLOCK_SCRIPT = new DefaultRedisScript<>();
	UNLOCK_SCRIPT.setLocation(new ClassPathResource("unlock.lua"));
    UNLOCK_SCRIPT.setResultType(Long.class);
}

public static unlock(){
    stringRedisTemplate.execute(
    	UNLOCK_SCRIPT,
        Collections.singletionList(KEY_PREFIX+name),
        ID_PREFIX+Thread.currentThread().getId()
    );
}
```

基于 Redis 的分布式锁实现思路：

- 利用 set nx ex 获取锁，并设置过期时间，保存线程标示
- 释放锁时先判断线程标示是否与自己一致，一致则删除锁

特性：

- 利用 set nx 满足互斥性
- 利用 set ex 保证故障时锁依然能释放，避免死锁，提高安全性
- 利用 Redis 集群保证高可用和高并发特性

> 基于 setnx 实现的分布式锁存在下面的问题：

- 不可重入：同一个线程无法多次获取同一把锁
- 不可重试：获取锁只尝试一次就返回 false，没有重试机制
- 超时释放：锁超时释放虽然可以避免死锁，但如果是业务执行耗时较长，也会导致锁释放，存在安全隐患
- 主从一致性：如果 Redis 提供了主从集群，主从同步存在延迟，当主宕机时，如果从没有同步主中的锁数据，则会出现锁失效。

### Redisson

Redisson 是一个在 Redis 的基础上实现的 Java 驻内存数据网格（In-Memory Data Grid）。它不仅提供了一系列的分布式的 Java 常用对象，还提供了许多分布式服务，其中就包含了各种分布式锁的实现。

<div align="center"><img src="img/image-20220611222819150.png"></div>

官网地址： https://redisson.org
GitHub 地址： https://github.com/redisson/redisson

为了避免 Redission 里的配置把 SpringBoot 里的覆盖了，这里就采用映入 redisson，自己配置 Redission 的方式。

> 引入依赖

```xml
<dependency>
	<groupId>org.redisson</groupId>
    <artifactId>redisson</artifactId>
    <version>3.13.6</version>
</dependency>
```

> 配置客户端

```java
@Configuration
public class RedisConfig{
    @Bean
    public RedissonClient redissonClient(){
        Config config = new Config();
        // 添加redis地址，这里添加了单点的地址，也可以使用config.useClusterServers()添加集群地址 
        config.useSingleServer()
            .setAddress("redis://192.168.1.101:6379")
            .setPassword("123");
        return Redission.create(config);
    }
}
```

> 使用 Redisson 分布式锁

```java
@Resource
private RedissonClient redissonClient;
@Test
void testRedisson() throws InterruptedException {
    // 获取锁（可重入），指定锁的名称
    RLock lock = redissonClient.getLock("anyLock");
    // 尝试获取锁，参数分别是：获取锁的最大等待时间（期间会重试），锁自动释放时间，时间单位
    boolean isLock = lock.tryLock(1, 10, TimeUnit.SECONDS);
    // 判断释放获取成功
    if(isLock){
        try {
            System.out.println("执行业务");
        }finally {
            // 释放锁
            lock.unlock();
        }
    }
}
```

> 原理

<div align="center"><img src="img/image-20220611225200680.png"></div>

释放锁的 lua 脚本

```lua
local key = KEYS[1]; 
-- 锁的key
local threadId = ARGV[1]; 
-- 线程唯一标识
local releaseTime = ARGV[2]; 
-- 锁的自动释放时间
-- 判断当前锁是否还是被自己持有
if (redis.call('HEXISTS', key, threadId) == 0) then
    return nil; 
    -- 如果已经不是自己，则直接返回
end;
-- 是自己的锁，则重入次数-1
local count = redis.call('HINCRBY', key, threadId, -1);
-- 判断是否重入次数是否已经为0 
if (count > 0) then
    -- 大于0说明不能释放锁，重置有效期然后返回
    redis.call('EXPIRE', key, releaseTime);
    return nil;
else  -- 等于0说明可以释放锁，直接删除
    redis.call('DEL', key);
    return nil;
end;
```

Redisson 分布式锁原理：

- 可重入：利用 hash 结构记录线程 id 和重入次数
- 可重试：利用信号量和 PubSub 功能实现等待、唤醒，获取锁失败的重试机制
- 超时续约：利用 watchDog，每隔一段时间（releaseTime / 3），重置超时时间

<div align="center"><img src="img/image-20220611225523290.png"></div>

### Redisson分布式锁主从一致问题

再刷下视频

1）不可重入 Redis 分布式锁：
原理：利用 setnx 的互斥性；利用 ex 避免死锁；释放锁时判断线程标示
缺陷：不可重入、无法重试、锁超时失效
2）可重入的 Redis 分布式锁：
原理：利用 hash 结构，记录线程标示和重入次数；利用 watchDog 延续锁时间；利用信号量控制锁重试等待
缺陷：redis 宕机引起锁失效问题
3）Redisson 的 multiLock：
原理：多个独立的 Redis 节点，必须在所有节点都获取重入锁，才算获取锁成功
缺陷：运维成本高、实现复杂

## Redis 优化秒杀

<div align="center"><h5>原始架构</h5></div>
<div align="center"><img src="img/image-20220611230518024.png"></div>

<div align="center"><h5>Redis优化后</h5></div>
<div align="center"><img src="img/image-20220611230740275.png"></div>



> 改进秒杀业务，提高并发性能

- 新增秒杀优惠券的同时，将优惠券信息保存到 Redis 中
- 基于 Lua 脚本，判断秒杀库存、一人一单，决定用户是否抢购成功
- 如果抢购成功，将优惠券 id 和用户 id 封装后存入阻塞队列
- 开启线程任务，不断从阻塞队列中获取信息，实现异步下单功能

> 秒杀优化思路

- 先利用 Redis 完成库存余量、一人一单判断，完成抢单业务
- 再将下单业务放入阻塞队列，利用独立线程异步下单
- 基于阻塞队列的异步秒杀存在哪些问题？
    - 内存限制问题
    - 数据安全问题

## Redis消息队列实现异步秒杀

用 RabbitMQ 这些更好

消息队列（Message Queue），字面意思就是存放消息的队列。最简单的消息队列模型包括 3 个角色：

- 消息队列：存储和管理消息，也被称为消息代理（Message Broker）
- 生产者：发送消息到消息队列
- 消费者：从消息队列获取消息并处理消息

<div align="center"><img src="img/image-20220611231720897.png"></div>



Redis 提供了三种不同的方式来实现消息队列：

- list 结构：基于 List 结构模拟消息队列
- PubSub：基本的点对点消息模型
- Stream：比较完善的消息队列模型

### 基于List的消息队列

消息队列（Message Queue），字面意思就是存放消息的队列。而 Redis 的 list 数据结构是一个双向链表，很容易模拟出队列效果。队列是入口和出口不在一边，因此我们可以利用：LPUSH 结合 RPOP、或者  RPUSH 结合 LPOP 来实现。不过要注意的是，当队列中没有消息时 RPOP 或 LPOP 操作会返回null，并不像 JVM 的阻塞队列那样会阻塞并等待消息。因此这里应该使用 BRPOP 或者 BLPOP 来实现阻塞效果。

> 基于 List 的消息队列有哪些优缺点？

优点：利用 Redis 存储，不受限于 JVM 内存上限；基于 Redis 的持久化机制，数据安全性有保证；可以满足消息有序性

缺点：无法避免消息丢失；只支持单消费者

### 基于PubSub的消息队列

PubSub（发布订阅）是 Redis2.0 版本引入的消息传递模型。顾名思义，消费者可以订阅一个或多个 channel，生产者向对应 channel 发送消息后，所有订阅者都能收到相关消息。

- SUBSCRIBE channel [channel] ：订阅一个或多个频道
- PUBLISH channel msg ：向一个频道发送消息
- PSUBSCRIBE pattern[pattern] ：订阅与 pattern 格式匹配的所有频道

<div align="center"><img src="img/image-20220611232158788.png"></div>

> 基于 PubSub 的消息队列

优点：采用发布订阅模型，支持多生产、多消费

缺点：不支持数据持久化；无法避免消息丢失；消息堆积有上限，超出时数据丢失

### 基于Stream的消息队列

Stream 是 Redis 5.0 引入的一种新数据类型，可以实现一个功能非常完善的消息队列。

发送消息的命令：

<div align="center">
    <img src="img/image-20220612151307548.png">
    <img src="img/image-20220612204433469.png">
    <img src="img/image-20220612204007529.png">
</div>

读取消息的方式之一：XREAD

<div align="center"><img src="img/image-20220612204645909.png"></div>

例如，使用 XREAD 读取第一个消息

<div align="center"><img src="img/image-20220612204656329.png"></div>

XREAD 阻塞方式，读取最新的消息

<div align="center"><img src="img/image-20220612204903295.png"></div>

在业务开发中，我们可以循环的调用 XREAD 阻塞方式来查询最新消息，从而实现持续监听队列的效果，伪代码如下：

<div align="center"><img src="img/image-20220612204922366.png"></div>

当我们指定起始 ID 为 $ 时，代表读取最新的消息，如果我们处理一条消息的过程中，又有超过 1 条以上的消息到达队列，则下次获取时也只能获取到最新的一条，会出现漏读消息的问题。

> Stream 消息队列的 XREAD 命令特点
>
> - 消息可回溯
> - 一个消息可以被多个消费者读取
> - 可以阻塞读取
> - 有消息漏读的风险

### 基于Stream的消息队列-消费者组

消费者组（Consumer Group）：将多个消费者划分到一个组中，监听同一个队列。具备下列特点：

- 消息分流：队列中的消息会分流给组内的不同消费者，而不是重复消费，从而加快消息处理的速度
- 消息标示：消费者组会维护一个标示，记录最后一个被处理的消息，哪怕消费者宕机重启，还会从标示之后读取消息。确保每一个消息都会被消费
- 消息确认：消费者获取消息后，消息处于 pending 状态，并存入一个 pending-list。当处理完成后需要通过 XACK 来确认消息，标记消息为已处理，才会从 pending-list 移除。

创建消费者组：

```bash
XGROUP CREATE  key groupName ID [MKSTREAM]
```

- key：队列名称
- groupName：消费者组名称
- ID：起始 ID 标示，$ 代表队列中最后一个消息，0 则代表队列中第一个消息
- MKSTREAM：队列不存在时自动创建队列

其他常见命令

```shell
# 删除指定的消费者组
XGROUP DESTORY key groupName

# 给指定的消费者组添加消费者
XGROUP CREATECONSUMER key groupname consumername

# 删除消费者组中的指定消费者
XGROUP DELCONSUMER key groupname consumername
```

从消费者组读消息

```shell
XREADGROUP GROUP group consumer [COUNT count] [BLOCK milliseconds] [NOACK] STREAMS key [key ...] ID [ID ...]
```

- group：消费组名称
- consumer：消费者名称，如果消费者不存在，会自动创建一个消费者
- count：本次查询的最大数量
- BLOCK milliseconds：当没有消息时最长等待时间
- NOACK：无需手动 ACK，获取到消息后自动确认
- STREAMS key：指定队列名称
- ID：获取消息的起始 ID：
    - ">"：从下一个未消费的消息开始
    - 其它：根据指定 id 从 pending-list 中获取已消费但未确认的消息，例如 0，是从 pending-list 中的第一个消息开始

消费者监听消息的基本思路：

<div align="center"><img src="img/image-20220612212045478.png"></div>

> Stream 消息队列 XREADGROUP 命令特点

- 消息可回溯
- 可以多消费者争抢消息，加快消费速度
- 可以阻塞读取
- 没有消息漏读的风险
- 有消息确认机制，保证消息至少被消费一次

### Redis 消息队列总结

|                     | List                                     | PubSub             | Stream                                                 |
| ------------------- | ---------------------------------------- | ------------------ | ------------------------------------------------------ |
| <b>消息持久化</b>   | 支持                                     | 不支持             | 支持                                                   |
| <b>阻塞读取</b>     | 支持                                     | 支持               | 支持                                                   |
| <b>消息堆积处理</b> | 受限于内存空间，可以利用多消费者加快处理 | 受限于消费者缓冲区 | 受限于队列长度，可以利用消费者组提高消费速度，减少堆积 |
| <b>消息确认机制</b> | 不支持                                   | 不支持             | 支持                                                   |
| <b>消息回溯</b>     | 不支持                                   | 不支持             | 支持                                                   |

### 案例练习

基于 Redis 的 Stream 结构作为消息队列，实现异步秒杀下单

1. 创建一个 Stream 类型的消息队列，名为 stream.orders
2. 修改之前的秒杀下单 Lua 脚本，在认定有抢购资格后，直接向 stream.orders 中添加消息，内容包含 voucherId、userId、orderId
3. 项目启动时，开启一个线程任务，尝试获取 stream.orders 中的消息，完成下单

## 达人探店

这部分主要练习业务。

- 发布笔记
- 点赞
- 点赞排行

### 发布笔记

探店笔记类似点评网站的评价，往往是图文结合。对应的表有两个：

- tb_blog：探店笔记表，包含笔记中的标题、文字、图片等
- tb_blog_comments：其他用户对探店笔记的评价

<div align="center"><img src="img/image-20220612214430543.png"></div>

需要注意部分是文件上传的设置。文件上传的路径要设置好，此处需要设置为 Nginx 目录

```java
public class SystemConstants{
    // 设置为 nginx 的目录
    public static final String IMAGE_UPLOAD_DIR = "D:\\xx\\nginx-1.18.0\\html\\xx\imgs\\";
    public static final String USER_NICK_NAME_PREFIX = "user_";
    public static final int DEFAULT_PAGE_SIZE = 5;
    public static final int MAX_PAGE_SIZE = 10;
}
```

### 点赞

需求如下：

- 同一个用户只能点赞一次，再次点击则取消点赞
- 如果当前用户已经点赞，则点赞按钮高亮显示（前端已实现，判断字段 Blog 类的 isLike 属性）

实现步骤：

- 给 Blog 类中添加一个 isLike 字段，标示是否被当前用户点赞
- 修改点赞功能，利用 Redis 的 set 集合判断是否点赞过，未点赞过则点赞数 +1，已点赞过则点赞数 -1
- 修改根据 id 查询 Blog 的业务，判断当前登录用户是否点赞过，赋值给 isLike字段
- 修改分页查询 Blog 业务，判断当前登录用户是否点赞过，赋值给 isLike 字段

### 点赞排行榜

点赞排行榜/游戏排行榜 etc...

需求：按照点赞时间先后排序，返回 Top5 的用户

|                 | List                 | Set          | SortedSet       |
| --------------- | -------------------- | ------------ | --------------- |
| <b>排序方式</b> | 按添加顺序排序       | 无法排序     | 根据score值排序 |
| <b>唯一性</b>   | 不唯一               | 唯一         | 唯一            |
| <b>查找方式</b> | 按索引查找或首尾查找 | 根据元素查找 | 根据元素查找    |

## 好友关注

- 关注和取关
- 共同关注
- 关注推送

### 关注和取关

关注是 User 之间的关系，是博主与粉丝的关系，在数据库中可以用一张 tb_follow 表来标示

<div align="center"><img src="img/image-20220612225906719.png"></div>

### 共同关注

利用 Redis 中恰当的数据结构（如 Set 集合），实现共同关注功能。在博主个人页面展示出当前用户与博主的共同好友。

### 关注推送

关注推送也叫做 Feed 流，直译为投喂。为用户持续的提供“沉浸式”的体验，通过无限下拉刷新获取新的信息。传统模式是用户寻找内容，而 Feed 模式是内容匹配用户

```mermaid
graph LR
用户---->|寻找|内容
```

```mermaid
graph LR
内容--->|匹配|用户
```

#### Feed 流的模式

Feed 流产品有两种常见模式

- Timeline：不做内容筛选，简单的按照内容发布时间排序，常用于好友或关注。例如朋友圈
    - 优点：信息全面，不会有缺失。并且实现也相对简单
    - 缺点：信息噪音较多，用户不一定感兴趣，内容获取效率低
- 智能排序：利用智能算法屏蔽掉违规的、用户不感兴趣的内容。推送用户感兴趣信息来吸引用户
    - 优点：投喂用户感兴趣信息，用户粘度很高，容易沉迷
    - 缺点：如果算法不精准，可能起到反作用

<b style="color:orange">本例中的个人页面，是基于关注的好友来做 Feed 流，因此采用 Timeline 的模式。该模式的实现方案有三种：</b>

- 拉模式：带上时间戳，按时间戳排序显示消息。假定 p4 关注了p1 和 p2，因此 p4 会把关注人的消息拉去到自己的收件箱。收件箱读完后就清理掉，因此比较节省内存（只保存一份消息）。但是每次读取的时候都重新去拉起消息做排序，比较耗时。如果关注的人多，这个耗时就更久了。

    ```mermaid
    graph LR
    p1-->发件箱1[发件箱 msg-16666]
    p2-->发件箱2[发件箱 msg-16667]
    p3-->发件箱3[发件箱 msg-16668]
    p4---|手动拉取|收件箱4[msg-16666/16667]
    ```

- 推模式：主动把消息推送到粉丝的收件箱。内存占用比较大，每个消息好发送给好多人。

    ```mermaid
    graph LR
    张三-->|主动推送|粉丝1-->收件箱1[收件箱: msg-1666]
    李四-->|主动推送|粉丝2-->收件箱2[收件箱: msg-1669]
    ```

- 推拉结合：普通人粉丝少，用推模式。大 V 活跃粉丝人少，可以用推模式，人少，内存耗费不是很高，速度也快；普通粉丝人少，不适合用推模式，采用拉模式。

|                  | **拉模式** | **推模式**                             | **推拉结合**           |
| ---------------- | ---------- | -------------------------------------- | ---------------------- |
| **写比例**       | 低         | 高                                     | 中                     |
| **读比例**       | 高         | 低                                     | 中                     |
| **用户读取延迟** | 高         | 低                                     | 低                     |
| **实现难度**     | 复杂       | 简单                                   | 很复杂                 |
| **使用场景**     | 很少使用   | 用户量少、没有大 V（用户量在千万以下） | 过千万的用户量，有大 V |

#### 案例

基于推模式实现关注推送功能

① 修改探店笔记的业务，在保存 blog 到数据库的同时，推送到粉丝的收件箱

② 收件箱满足可以根据时间戳排序（List、Sorted），必须用 Redis 的数据结构实现

③ 查询收件箱数据时，可以实现分页查询

Feed 流中的数据会不断更新，所以数据的角标也在变化，因此不能采用传统的分页模式。可以采用滚动分页的模式：每次记录当前查询的最后一条记录，下一页就从最后一条记录开始查询。

<div align="center"><img src="img/image-20220627124355521.png"></div>

部分代码

```java
/**
保存笔记
查询笔记作者所有粉丝
推送笔记 id 给粉丝
返回 id
主要是推送的代码
*/
for(Follow follow : follows){
    Long userId = follow.getUserId();
    String key = "feed:"+userId;
    stringRedisTemplate.opsForZSet().add(key,blog.getId().toString(),System.currentTimeMillis());
}
```

拉取代码逻辑

```java
/**
zreverangebyscore z1 5 0 withscores limit 1 3
滚动分页查询参数
max：当前时间戳 | 上一次查询的最小时间戳
min：0
offset：0 | 在上一次的结果中，与最小值一样的元素个数
count：3
*/
public Result queryBlogOfFollow(Long max,Integer offset){
    Long userId = xxx;
    String key = "feed:"+userId;
    Set<ZSetOperations.TypedTuple<String>> typedTuples = stringRedisTemplate.opsForZSet()
        .reverseRangeByScoreWithScores(key,0,max,offset,2);
    //...
    // 解析数据
    long minTime = 0;
    int os = 1;
    for(ZSetOperations.TypedTule<String> tuple : typedTuples){
        ids.add(Long.valueOf(tuple.getValue()));
        long time = tuple.getScore().longValue();
        if(time == minTime){
            os++;
        }else{
            minTime = time;
            os = 1;
        }
    }
    // 返回数据
}
```

## 附件商户

### GEO数据结构

GEO 就是 Geolocation 的简写形式，代表地理坐标。Redis 在 3.2 版本中加入了对 GEO 的支持，允许存储地理坐标信息，帮助我们根据经纬度来检索数据。常见的命令有：

- GEOADD：添加一个地理空间信息，包含：经度（longitude）、纬度（latitude）、值（member）
- GEODIST：计算指定的两个点之间的距离并返回
- GEOHASH：将指定 member 的坐标转为 hash 字符串形式并返回
- GEOPOS：返回指定 member 的坐标
- GEORADIUS：指定圆心、半径，找到该圆内包含的所有 member，并按照与圆心之间的距离排序后返回。6.2 以后已废弃
- GEOSEARCH：在指定范围内搜索 member，并按照与指定点之间的距离排序后返回。范围可以是圆形或矩形。6.2. 新功能
- GEOSEARCHSTORE：与 GEOSEARCH 功能一致，不过可以把结果存储到一个指定的 key。 6.2. 新功能

> 练习

添加下面几条数据：

- 北京南站（ 116.378248 39.865275 ）
- 北京站（ 116.42803 39.903738 ）
- 北京西站（ 116.322287 39.893729 ）

计算北京西站到北京站的距离

搜索天安门（ 116.397904 39.909005 ）附近 10km 内的所有火车站，并按照距离升序排序

```bash
geoadd g1 116.378248 39.865275 bjn 116.42803 39.903738  116.322287 39.893729 bjz bjx

geodist g1 bjn bjx

geosearch g1 fromlonlat 116.397904 39.909005 byradius 10 km withdist # 默认是升序的
"""
123
"""
```

### 附件商户搜索

假定：在首页中点击某个频道，即可看到频道下的商户。

我们可以按照商户类型做分组，类型相同的商户作为同一组，以 typeId 为 key 存入同一个 GEO 集合中即可

| key           | value | score |
| ------------- | ----- | ----- |
| shop:geo:food | xxx   | xxx   |
|               | xxx   | xxx   |
| shop:geo:tax  | qqq   | www   |

SpringDataRedis 的 2.3.9 版本并不支持 Redis 6.2 提供的 GEOSEARCH 命令，因此我们需要提示其版本，修改自己的 POM 文件，内容如下：

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
    <exclusions>
        <exclusion>
            <groupId>org.springframework.data</groupId>
            <artifactId>spring-data-redis</artifactId>
        </exclusion>
        <exclusion>
            <artifactId>lettuce-core</artifactId>
            <groupId>io.lettuce</groupId>
        </exclusion>
    </exclusions>
</dependency>
<dependency>
    <groupId>org.springframework.data</groupId>
    <artifactId>spring-data-redis</artifactId>
    <version>2.6.2</version>
</dependency>
<dependency>
    <artifactId>lettuce-core</artifactId>
    <groupId>io.lettuce</groupId>
    <version>6.1.6.RELEASE</version>
</dependency>
```

## 用户签到

- BitMap 用法
- 签到功能
- 签到统计

### BitMap用法

假如我们用一张表来存储用户签到信息，其结构应该如下：

<div align="center"><img src="img/image-20220612231356449.png"></div>

假如有 1000 万用户，平均每人每年签到次数为 10 次，则这张表一年的数据量为 1 亿条

每签到一次需要使用（8 + 8 + 1 + 1 + 3 + 1）共 22 字节的内存，一个月则最多需要 600 多字节

我们按月来统计用户签到信息，签到记录为 1，未签到则记录为 0.

把每一个 bit 位对应当月的每一天，形成了映射关系。用 0 和 1 标示业务状态，这种思路就称为位图（BitMap）。

Redis 中是利用 string 类型数据结构实现 BitMap，因此最大上限是 512M，转换为bit则是 $2^{32}$个 bit 位。

BitMap 的操作命令有：

- SETBIT：向指定位置（offset）存入一个 0 或 1
- GETBIT ：获取指定位置（offset）的 bit 值
- BITCOUNT ：统计 BitMap 中值为 1 的 bit 位的数量
- BITFIELD ：操作（查询、修改、自增）BitMap 中 bit 数组中的指定位置（offset）的值
- BITFIELD_RO ：获取 BitMap 中 bit 数组，并以十进制形式返回
- BITOP ：将多个 BitMap 的结果做位运算（与 、或、异或）
- BITPOS ：查找 bit 数组中指定范围内第一个 0 或 1 出现的位置

### 签到功能

需求：实现签到接口，将当前用户当天签到信息保存到Redis中。因为BitMap底层是基于String数据结构，因此其操作也都封装在字符串相关操作中了。

### 签到统计

> 什么叫连续签到天数？

从最后一次签到开始向前统计，直到遇到第一次未签到为止，计算总的签到次数，就是连续签到天数。我们可以用 redis 的 bitmap 来实现连续签到功能。

<div align="center"><img src="img/image-20220613164010361.png"></div>

需求：用 redis 命令实现一个月签到的功能。

 ```shell
 # 初始化签到数（设置一个数据，所有bit位初始化为0）
 setbit count 0 0
 
 # 签到（第一天就将offset=0的bit位设置为1）
 setbit count 0 1
 
 # 查看你第一天(offset=0)是否签到
 getbit count 0
 
 # 统计签到的总天数
 bitcount count
 
 # 统计指定范围内的签到天数（0字节到~0字节的签到天数，就是统计了8个bit位）
 bitcount count 0 0
 ```

> 如何得到本月到今天为止的所有签到数据?

BITFIELD key GET u[dayOfMonth] 0

> 如何从后向前遍历每个 bit 位？

与 1 做与运算，就能得到最后一个 bit 位。随后右移 1 位，下一个 bit 位就成为了最后一个 bit 位。

## UV统计

- HyperLogLog 用法
- 实现 UV 统计

### HyperLogLog用法

首先我们搞懂两个概念：

- UV：全称 Unique Visitor，也叫独立访客量，是指通过互联网访问、浏览这个网页的自然人。1 天内同一个用户多次访问该网站，只记录 1 次。
- PV：全称 Page View，也叫页面访问量或点击量，用户每访问网站的一个页面，记录 1 次 PV，用户多次打开页面，则记录多次 PV。往往用来衡量网站的流量。

UV 统计在服务端做会比较麻烦，因为要判断该用户是否已经统计过了，需要将统计过的用户信息保存。但是如果每个访问的用户都保存到 Redis 中，数据量会非常恐怖。

Hyperloglog(HLL) 是从 Loglog 算法派生的概率算法，用于确定非常大的集合的基数，而不需要存储其所有值。相关算法原理可以参考：https://juejin.cn/post/6844903785744056333#heading-0

Redis 中的 HLL 是基于 string 结构实现的，单个 HLL 的内存永远小于 16kb，内存占用低的令人发指！作为代价，其测量结果是概率性的，有小于 0.81％ 的误差。不过对于 UV 统计来说，这完全可以忽略。

<div align="center"><img src="img/image-20220613170117858.png"></div>

### 实现UV统计

我们直接利用单元测试，向 HyperLogLog 中添加 100 万条数据，看看内存占用和统计效果如何

```java
@Test
void testHyperLogLog() {
    // 准备数组，装用户数据
    String[] users = new String[1000];
    // 数组角标
    int index = 0;
    for (int i = 1; i <= 1000000; i++) {
        // 赋值
        users[index++] = "user_" + i;
        // 每1000条发送一次
        if (i % 1000 == 0) {
            index = 0;
            stringRedisTemplate.opsForHyperLogLog().add("hll1", users);
        }
    }
    // 统计数量
    Long size = stringRedisTemplate.opsForHyperLogLog().size("hll1");
    System.out.println("size = " + size); // 997593
}
```

> HyperLogLog 的作用
>
> - 做海量数据的统计工作
> - HyperLogLog 的优点：内存占用极低；性能非常好
> - HyperLogLog 的缺点：有一定的误差

> Pipeline 导入数据

如果要导入大量数据到 Redis 中，可以有多种方式：

- 每次一条，for 循环写入
- 每次多条，批量写入

# 分布式缓存

单点 Redis 的问题：

- 数据丢失问题，服务器重启可能会丢失数据；Redis 数据持久化可以解决该问题
- 并发能力问题，单节点的 Redis 无法满足高并发场景；搭建主从集群，实现读写分离
- 故障回复问题，如果 Redis 宕机，则服务不可用，需要一种自动的故障恢复手段；利用 Redis 哨兵，实现健康检测和自动恢复
- 存储能力问题，Redis 基于内存，单节点的存储难以满足海里数据需求；搭建分片集群，利用插槽机制实现动态扩容

## Redis的持久化

- RDB 持久化
- AOF 持久化

### RDB持久化

RDB 全称 Redis Database Backup file（Redis 数据备份文件），也被叫做 Redis 数据快照。简单来说就是把内存中的所有数据都记录到磁盘中。当 Redis 实例故障重启后，从磁盘读取快照文件，恢复数据。快照文件称为 RDB 文件，默认是保存在当前运行目录。

#### 执行时机

RDB 持久化在四种情况下会执行：

- 执行 save 命令
- 执行 bgsave 命令
- Redis 停机时
- 触发 RDB 条件时

<b>1）save命令</b>

执行下面的命令，可以立即执行一次 RDB：

<div align="center"><img src="img/image-20210725144536958.png"></div>

save 命令会导致主进程执行 RDB，这个过程中其它所有命令都会被阻塞。适合用在 Redis 即将停止时，比如在数据迁移时可能用到。

<b>2）bgsave 命令</b>

下面的命令可以异步执行 RDB：

<div align="center"><img src="img/image-20210725144725943.png"></div>

这个命令执行后会开启独立进程完成 RDB，主进程可以持续处理用户请求，不受影响。

<b>3）停机时</b>

Redis 停机时会执行一次 save 命令，实现 RDB 持久化。

<b>4）触发 RDB 条件</b>

Redis 内部有触发 RDB 的机制，可以在 redis.conf 文件中找到，格式如下：

```properties
# 900秒内，如果至少有1个key被修改，则执行bgsave ， 如果是save "" 则表示禁用RDB
save 900 1  
save 300 10  
save 60 10000 
```

RDB 的其它配置也可以在 redis.conf 文件中设置：

```properties
# 是否压缩 ,建议不开启，压缩也会消耗cpu，磁盘的话不值钱
rdbcompression yes

# RDB文件名称
dbfilename dump.rdb  

# 文件保存的路径目录
dir ./ 
```

RDB 的频率不要太高，频率太高会一直处于写入数据的状态，影响性能，一般用默认的就好。

#### RDB原理

bgsave 开始时会 fork 主进程得到子进程，子进程共享主进程的内存数据。完成 fork 后读取内存数据并写入 RDB 文件。注意：fork 这个操作过程是阻塞的。

fork 采用的是 copy-on-write 技术：

- 当主进程执行读操作时，访问共享内存；
- 当主进程执行写操作时，则会拷贝一份数据，执行写操作。

<div align="center"><img src="img/image-20210725151319695.png"></div>

Linux 中，所有的进程都没办法直接操作物理内存而是由操作系统给每个进程分配一个虚拟内存，主进程操作虚拟内存，操作系统维护一个虚拟内存与物理内存直接的映射关系（页表）。fork 主进程实际上是 fork 页表（页表中保存了物理内存与虚拟内存的映射关系）的过程，让子进程和主进程拥有一样的映射关系。这样就实现了子进程和主进程一样的内存共享。这样就无需拷贝内存中的数据，直接实现数据共享。

但这样会有一个问题，就是一个读一个写，会有并发问题。如果子进程在拷贝数据的时候，主进程还在写怎么办？fork 底层会采用 copy-on-write 的技术。然源数据只读，如果需要修改就复制一份数据，在复制的数据中进行修改（后面好像是等持久化结束后，在写入源数据。MySQL 也有一个类似的操作，查下 MySQL 的笔记）

#### 小结

RDB 方式 bgsave 的基本流程？

- fork 主进程得到一个子进程，共享内存空间
- 子进程读取内存数据并写入新的 RDB 文件
- 用新 RDB 文件替换旧的 RDB 文件

RDB 会在什么时候执行？save 60 1000 代表什么含义？

- 默认是服务停止时才会执行
- 代表 60 秒内至少执行 1000 次修改则触发 RDB

RDB 的缺点？

- RDB 执行间隔时间长，两次 RDB 之间写入数据有丢失的风险（要速度快的话就牺牲数据的一致性）
- fork 子进程、压缩、写出 RDB 文件都比较耗时

### AOF持久化

AOF 全称为 Append Only File（追加文件）。Redis 处理的每一个写命令都会记录在 AOF 文件，可以看做是命令日志文件。

<div align="center"><img src="img/image-20210725151543640.png"></div>

#### AOF配置

AOF 默认是关闭的，需要修改 redis.conf 配置文件来开启 AOF：

```properties
# 是否开启AOF功能，默认是no
appendonly yes
# AOF文件的名称
appendfilename "appendonly.aof"
```

AOF 的命令记录的频率也可以通过 redis.conf 文件来配：

```properties
# 表示每执行一次写命令，立即记录到AOF文件，Redis 主进程完成磁盘写入操作。
appendfsync always 
# 写命令执行完先放入AOF缓冲区，然后表示每隔1秒将缓冲区数据写到AOF文件，是默认方案，子进程完成磁盘写入操作
appendfsync everysec 
# 写命令执行完先放入AOF缓冲区，由操作系统决定何时将缓冲区内容写回磁盘
appendfsync no
```

三种策略对比：

<div align="center"><img src="img/image-20210725151654046.png"></div>

#### AOF文件重写

因为是记录命令，AOF 文件会比 RDB 文件大的多。而且 AOF 会记录对同一个 key 的多次写操作，但只有最后一次写操作才有意义。通过执行 bgrewriteaof 命令，可以让 AOF 文件执行重写功能，用最少的命令达到相同效果。

<div align="center"><img src="img/image-20210725151729118.png"></div>

如图，AOF 原本有三个命令，但是 `set num 123 和 set num 666` 都是对 num 的操作，第二次会覆盖第一次的值，因此第一个命令记录下来没有意义。

所以重写命令后，AOF 文件内容就是：`mset name jack num 666`

Redis 也会在触发阈值时自动去重写 AOF 文件。阈值也可以在 redis.conf 中配置：

```properties
# AOF文件比上次文件增长超过 100%（翻了一倍）则触发重写
auto-aof-rewrite-percentage 100
# AOF文件体积超过 64mb 就触发重写 
auto-aof-rewrite-min-size 64mb 
```

### 混合RDB和AOF

Redis 4.0 中提出了一个混合使用 AOF 日志和内存快照的方法。内存快照以一定的频率执行，在两次快照之间，使用 AOF 日志记录这期间的所有命令操作。这样，不用频繁执行快照，避免了频繁 fork 对主线程的影响。且，AOF 日志也只用记录两次快照间的操作，无需记录所有操作了，不会出现文件过大的情况，也可以避免重写开销。如下图所示，T1 和 T2 时刻的修改，用 AOF 日志记录，等到第二次做全量快照时，就可以清空 AOF 日志，因为此时的修改都已经记录到快照中了，恢复时就不再用日志了。

<div align="center"><img src="img/AOF_RDB.webp" width="80%"></div>

### RDB与AOF对比

RDB 和 AOF 各有自己的优缺点，如果对数据安全性要求较高，在实际开发中往往会<b>结合</b>两者来使用。

<div align="center"><img src="img/image-20210725151940515.png"></div>

## Redis主从

### 搭建主从架构

单节点 Redis 的并发能力是有上限的，要进一步提高 Redis 的并发能力，就需要搭建主从集群，实现读写分离。

<div align="center"><img src="img/image-20210725152037611.png"></div>

多个从结点承担读的请求，Redis 读取数据的能力可以得到极大的提升。

### 主从同步原理

#### 全量同步

主从第一次建立连接时，会执行<b>全量同步</b>，将 master 节点的所有数据都拷贝给 slave 节点，流程：

<div align="center"><img src="img/image-20210725152222497.png"></div>

这里有一个问题，master 如何得知 salve 是第一次来连接呢？？

有几个概念，可以作为判断依据：

- <b>Replication Id</b>：简称 replid，是数据集的标记，id 一致则说明是同一数据集。每一个 master 都有唯一的 replid，slave 则会继承 master 节点的 replid
- <b>offset</b>：偏移量，随着记录在 repl_baklog 中的数据增多而逐渐增大。slave 完成同步时也会记录当前同步的 offset。如果 slave 的 offset 小于 master 的 offset，说明 slave 数据落后于 master，需要更新。

因此 slave 做数据同步，必须向 master 声明自己的 replication id  和 offset，master 才可以判断到底需要同步哪些数据。

因为 slave 原本也是一个 master，有自己的 replid 和 offset，当第一次变成 slave，与 master 建立连接时，发送的 replid 和 offset 是自己的 replid 和 offset。

master 判断发现 slave 发送来的 replid 与自己的不一致，说明这是一个全新的 slave，就知道要做全量同步了。

master 会将自己的 replid 和 offset 都发送给这个 slave，slave 保存这些信息。以后 slave 的 replid 就与 master 一致了。

因此，<b>master 判断一个节点是否是第一次同步的依据，就是看 replid 是否一致</b>。

<div align="center"><img src="img/image-20210725152700914.png"></div>

完整流程描述：

- slave 节点请求增量同步
- master 节点判断 replid，发现不一致，拒绝增量同步
- master 将完整内存数据生成 RDB，发送 RDB 到 slave
- slave 清空本地数据，加载 master 的 RDB
- master 将 RDB 期间的命令记录在 repl_baklog，并持续将 log 中的命令发送给 slave
- slave 执行接收到的命令，保持与 master 之间的同步

#### 增量同步

全量同步需要先做 RDB，然后将 RDB 文件通过网络传输个 slave，成本太高了。因此除了第一次做全量同步，其它大多数时候 slave 与 master 都是做<b>增量同步</b>。

增量同步就是只更新 slave 与 master 存在差异的部分数据。如图：

<div align="center"><img src="img/image-20210725153201086.png"></div>

那么 master 怎么知道 slave 与自己的数据差异在哪里呢？简单来说是根据 master 和 slave 的 offset 的差值来判断的，如果 master 和 slave 的 offset 不一样，则说明主从需要进行同步。如果 master 的 offset 覆盖了未同步的数据，就得进行全增量同步了。具体原理请看 “repl_backlog 原理”

#### repl_backlog原理

master 怎么知道 slave 与自己的数据差异在哪里呢？这就要靠全量同步时的 repl_baklog 文件了。

这个文件是一个固定大小的数组，只不过数组是环形，也就是说<b>角标到达数组末尾后，会再次从 0 开始读写</b>，这样数组头部的数据就会被覆盖。

repl_baklog 中会记录 Redis 处理过的命令日志及 offset，包括 master 当前的 offset，和 slave 已经拷贝到的 offset：

<div align="center"><img src="img/image-20210725153359022.png"></div>

slave 与 master 的 offset 之间的差异，就是 salve 需要增量拷贝的数据了。随着不断有数据写入，master 的 offset 逐渐变大， slave 也不断的拷贝，追赶 master 的 offset

<div align="center"><img src="img/image-20210725153524190.png"></div>

直到数组被填满

<div align="center"><img src="img/image-20210725153715910.png"></div>

此时，如果有新的数据写入，就会覆盖数组中的旧数据。不过，旧的数据只要是绿色的，说明是已经被同步到slave的数据，即便被覆盖了也没什么影响。因为未同步的仅仅是红色部分。

但是，如果 slave 出现网络阻塞，导致 master 的 offset 远远超过了 slave 的 offset： 

<div align="center"><img src="img/image-20210725153937031.png">
</div>

如果 master 继续写入新数据，其 offset 就会覆盖旧的数据，直到将 slave 现在的 offset 也覆盖：

<div align="center"><img src="img/image-20210725154155984.png"></div>

棕色框中的红色部分，就是尚未同步，但是却已经被覆盖的数据。此时如果 slave 恢复，需要同步，却发现自己的 offset 都没有了，无法完成增量同步了。只能做全量同步。

<div align="center"><img src="img/image-20210725154216392.png"></div>

### 主从同步优化

主从同步可以保证主从数据的一致性，非常重要。

可以从以下几个方面来优化 Redis 主从就集群：

- 在 master 中配置 repl-diskless-sync yes 启用无磁盘复制，<span style="color:orange">（即，不是先在磁盘中生成 RDB 然后再通过网络发送出去，而是直接通过网络发送，不再经过磁盘了。适合磁盘 IO 速度慢，网络速度快。）</span>，避免全量同步时的磁盘 IO。
- Redis 单节点上的内存占用不要太大，减少 RDB 导致的过多磁盘 IO

上面两个都是在提高全量同步的性能，下面两点是从减少全量同步出发的。

- 适当提高 repl_baklog 的大小，允许主从数据的差异更大，就可以减少全量同步发生的几率了。发现 slave 宕机时尽快实现故障恢复，尽可能避免全量同步
- 限制一个 master 上的 slave 节点数量，如果实在是太多 slave，则可以采用主-从-从链式结构，减少 master 压力<span style="color:orange">（后面的 slave 同步中间的 slave 的数据）</span>

<div align="center"><img src="img/image-20210725154405899.png"></div>

### 小结

<b>简述全量同步和增量同步区别？</b>

- 全量同步：master 将完整内存数据生成 RDB，发送 RDB 到slave。后续命令则记录在 repl_baklog，逐个发送给 slave。
- 增量同步：slave 提交自己的 offset 到 master，master 获取 repl_baklog 中从 offset 之后的命令给 slave

<b>什么时候执行全量同步？</b>

- slave 节点第一次连接 master 节点时
- slave 节点断开时间太久，repl_baklog 中的 offset 已经被覆盖时

<b>什么时候执行增量同步？</b>

- slave 节点断开又恢复，并且在 repl_baklog 中能找到 offset 时

<b>实际使用是全量同步+增量同步一起使用。</b>

## Redis哨兵

slave 节点宕机恢复后可以找 master 节点同步数据，那 master 节点宕机该如何处理？

Redis 提供了哨兵（Sentinel）机制来实现主从集群的自动故障恢复。哨兵是用于监控整个集群做故障恢复的。

### 哨兵原理

#### 集群的结构和作用

哨兵的结构如图：

<div align="center"><img src="img/image-20210725154528072.png"></div>

哨兵的作用如下：

- <b>监控</b>：Sentinel 会不断检查您的 master 和 slave 是否按预期工作。
- <b>自动故障恢复</b>：如果 master 故障，Sentinel 会将一个 slave 提升为 master。当故障实例恢复后也以新的 master 为主。
- <b>通知</b>：Sentinel 充当 Redis 客户端的服务发现来源，当集群发生故障转移时，会将最新信息推送给 Redis 的客户端。<span style="color:orange">（Redis 客户端找主从服务的时候，是从 Sentinel 中找的，由 Sentinel 告诉客户端主的地址在哪里，从的地址在哪里；此时 Sentinel 就充当了 Redis 客户端服务发现的来源了。）</span>

#### 服务状态监控

Sentinel 基于心跳机制监测服务状态，每隔 1 秒向集群的每个实例发送 ping 命令：

- 主观下线：如果某 sentinel 节点发现某实例未在规定时间响应，则认为该实例<b>主观下线</b>，sentinel 认为你下线了，所以是主观下线。
- 客观下线：若超过指定数量（quorum）的 sentinel 都认为该实例主观下线，则该实例<b>客观下线</b>。 quorum 值最好超过 Sentinel 实例数量的一半。

<div align="center"><img src="img/image-20210725154632354.png"></div>

#### 故障恢复原理

一旦发现 master 故障，sentinel 需要在 salve 中选择一个作为新的 master，选择依据是这样的：

- 首先会判断 slave 节点与 master 节点断开时间长短，如果超过指定值（down-after-milliseconds * 10）则会排除该 slave 节点<span style="color:orange">（断开时间越长，未同步的数据就越多，这样的节点就不具备选举的资格）</span>
- 然后判断 slave 节点的 slave-priority 值（默认都是 1），越小优先级越高，如果是 0 则永不参与选举
- 如果 slave-prority 一样，则判断 slave 节点的 offset 值，越大说明数据越新，优先级越高
- 最后是判断 slave 节点的运行 id 大小，越小优先级越高。（是为了避免 offset 都一样，难以抉择，因此依靠 id 随便选一个）

当选出一个新的 master 后，该如何实现切换呢？流程如下：

- sentinel 给备选的 slave1 节点发送 slaveof no one 命令，让该节点成为 master
- sentinel 给所有其它 slave 发送 slaveof 192.168.150.101 7002 命令，让这些 slave 成为新 master 的从节点，开始从新的 master 上同步数据。
- 最后，sentinel 将故障节点标记为 slave，当故障节点恢复后会自动成为新的 master 的 slave 节点

<div align="center"><img src="img/image-20210725154816841.png"></div>

#### 小结

Sentinel 的三个作用是什么？

- 监控
- 故障转移
- 通知

Sentinel 如何判断一个 redis 实例是否健康？

- 每隔 1 秒发送一次 ping 命令，如果超过一定时间没有相向则认为是主观下线
- 如果大多数 sentinel 都认为实例主观下线，则判定服务下线

故障转移步骤有哪些？

- 首先选定一个 slave 作为新的 master，执行 slaveof no one（自己不再是 slave，要变成 master）
- 然后让所有节点都执行 slaveof 新 master
- 修改故障节点配置，添加 slaveof 新 master

### 搭建哨兵集群

~ 这部分没有实践过，先记个笔记，后面再说。

### RedisTemplate

在 Sentinel 集群监管下的 Redis 主从集群，其节点会因为自动故障转移而发生变化，Redis 的客户端必须感知这种变化，及时更新连接信息。Spring 的 RedisTemplate 底层利用 lettuce 实现了节点的感知和自动切换。

#### 导入Demo工程

redis-demo 这个文件夹

#### 引入依赖

在项目的 pom 文件中引入依赖：

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
```

#### 配置Redis地址

在配置文件 application.yml 中指定 redis 的 sentinel 相关信息：

```yaml
spring:
  redis:
    sentinel:
      master: mymaster
      nodes:
        - 192.168.150.101:27001
        - 192.168.150.101:27002
        - 192.168.150.101:27003
```

#### 配置读写分离

在项目的启动类中，添加一个新的 bean：

```java
@Bean
public LettuceClientConfigurationBuilderCustomizer clientConfigurationBuilderCustomizer(){
    return clientConfigurationBuilder -> clientConfigurationBuilder.readFrom(ReadFrom.REPLICA_PREFERRED);
}
```

这个 bean 中配置的就是读写策略，包括四种：

- MASTER：从主节点读取
- MASTER_PREFERRED：优先从 master 节点读取，master 不可用才读取replica
- REPLICA：从 slave（replica）节点读取
- REPLICA _PREFERRED：优先从 slave（replica）节点读取，所有的 slave 都不可用才读取 master

## Redis分片集群

### 搭建分片集群

主从和哨兵可以解决高可用、高并发读的问题。但是依然有两个问题没有解决：

- 海量数据存储问题

- 高并发写的问题

使用分片集群可以解决上述问题，如图：

<div align="center"><img src="img/image-20210725155747294.png"></div>



分片集群特征：

- 集群中有多个 master，每个 master 保存不同数据

- 每个 master 都可以有多个 slave 节点

- master 之间通过 ping 监测彼此健康状态

- 客户端请求可以访问集群任意节点，最终都会被转发到正确节点

### 散列插槽

#### 插槽原理

Redis 会把每一个 master 节点映射到 0~16383 共 16384 个插槽（hash slot）上，查看集群信息时就能看到：

<div align="center"><img src="img/image-20210725155820320.png"></div>

数据 key 不是与节点绑定，而是与插槽绑定。redis 会根据 key 的有效部分计算插槽值，分两种情况：

- key 中包含 "{}"，且 “{}” 中至少包含 1 个字符，“{}” 中的部分是有效部分
- key 中不包含 “{}”，整个 key 都是有效部分

例如：key 是 num，那么就根据 num 计算，如果是 {itcast} num，则根据 itcast 计算。计算方式是利用 CRC16 算法得到一个 hash 值，然后对 16384 取余，得到的结果就是 slot 值。

<div align="center"><img src="img/image-20210725155850200.png"></div>

如图，在 7001 这个节点执行 set a 1 时，对 a 做 hash 运算，对 16384 取余，得到的结果是 15495，因此要存储到 103 节点。

到了 7003 后，执行 `get num` 时，对 num 做 hash 运算，对 16384 取余，得到的结果是 2765，因此需要切换到 7001 节点。

#### 小结

Redis 如何判断某个 key 应该在哪个实例？

- 将 16384 个插槽分配到不同的实例
- 根据 key 的有效部分计算哈希值，对 16384 取余
- 余数作为插槽，寻找插槽所在实例即可

如何将同一类数据固定的保存在同一个 Redis 实例？

- 这一类数据使用相同的有效部分，例如 key 都以 {typeId} 为前缀

### 集群伸缩

redis-cli --cluster 提供了很多操作集群的命令，可以通过下面方式查看：

<div align="center"><img src="img/image-20210725160138290.png"></div>

添加节点的命令

<div align="center"><img src="img/image-20210725160448139.png"></div>

#### 需求分析

需求：向集群中添加一个新的 master 节点，并向其中存储 num = 10

- 启动一个新的 redis 实例，端口为 7004
- 添加 7004 到之前的集群，并作为一个 master 节点
- 给 7004 节点分配插槽，使得 num 这个 key 可以存储到 7004 实例

这里需要两个新的功能：

- 添加一个节点到集群中
- 将部分插槽分配到新插槽

<b>创建 Redis 实例</b>

创建一个文件夹：

```sh
mkdir 7004
```

拷贝配置文件：

```sh
cp redis.conf /7004
```

修改配置文件：

```sh
sed /s/6379/7004/g 7004/redis.conf
```

启动

```sh
redis-server 7004/redis.conf
```

<b>添加新节点到 redis</b>

<div align="center"><img src="img/image-20210725160448139.png"></div>

执行命令：

```sh
redis-cli --cluster add-node  192.168.150.101:7004 192.168.150.101:7001
```

通过命令查看集群状态：

```sh
redis-cli -p 7001 cluster nodes
```

如图，7004 加入了集群，并且默认是一个 master 节点：

<div align="center"><img src="img/image-20210725161007099.png"></div>

但是，可以看到 7004 节点的插槽数量为 0，因此没有任何数据可以存储到 7004 上

<b>转移插槽</b>

我们要将 num 存储到 7004 节点，因此需要先看看 num 的插槽是多少：

<div align="center"><img src="img/image-20210725161241793-16637601446392.png"></div>

如上图所示，num 的插槽为 2765.

我们可以将 0~3000 的插槽从 7001 转移到 7004，命令格式如下：

<div align="center"><img src="img/image-20210725161401925-16637601446381.png"></div>

具体命令如下：

建立连接：

<div align="center"><img src="img/image-20210725161506241-16637601446393.png"></div>

得到下面的反馈：

<div align="center"><img src="img/image-20210725161540841-16637601446394.png"></div>

询问要移动多少个插槽，我们计划是 3000 个：

新的问题来了：

<div align="center"><img src="img/image-20210725161637152-16637601446395.png"></div>

那个 node 来接收这些插槽？？

显然是 7004，那么 7004 节点的 id 是多少呢？

<div align="center"><img src="img/image-20210725161731738-16637601446396.png"></div>

复制这个 id，然后拷贝到刚才的控制台后：

<div align="center"><img src="img/image-20210725161817642-16637601446397.png"></div>

这里询问，你的插槽是从哪里移动过来的？

- all：代表全部，也就是三个节点各转移一部分
- 具体的 id：目标节点的 id
- done：没有了

这里我们要从 7001 获取，因此填写 7001 的 id：

<div align="center"><img src="img/image-20210725162030478-166376014464011.png"></div>

填完后，点击 done，这样插槽转移就准备好了：

<div align="center"><img src="img/image-20210725162101228-16637601446398.png"></div>

确认要转移吗？输入 yes：

然后，通过命令查看结果：

<div align="center"><img src="img/image-20210725162145497-16637601446409.png"></div>

可以看到： 

<div align="center"><img src="img/image-20210725162224058-166376014464010.png"></div>

目的达成。

### 故障转移

集群初识状态是这样的：

<div align="center"><img src="img/image-20210727161152065-166376053323329.png"></div>

其中 7001、7002、7003 都是 master，我们计划让 7002 宕机。

#### 自动故障转移

当集群中有一个master宕机会发生什么呢？比如直接停止一个 redis 实例，例如 7002：

```sh
redis-cli -p 7002 shutdown
```

1）首先是该实例与其它实例失去连接

2）然后是疑似宕机：

<div align="center"><img src="img/image-20210725162319490-166376045993423.png"></div>

3）最后是确定下线，自动提升一个 slave 为新的 master：

<div align="center"><img src="img/image-20210725162408979-166376045993424.png"></div>

4）当 7002 再次启动，就会变为一个 slave 节点了：

<div align="center"><img src="img/image-20210727160803386-166376045993425.png"></div>

#### 手动故障转移

利用 cluster failover 命令可以手动让集群中的某个 master 宕机，切换到执行 cluster failover 命令的这个 slave 节点，实现无感知的数据迁移。其流程如下：

<div align="center"><img src="img/image-20210725162441407.png"></div>

这种 failover 命令可以指定三种模式：

- 缺省：默认的流程，如图 1~6 歩
- force：省略了对 offset 的一致性校验
- takeover：直接执行第 5 歩，忽略数据一致性、忽略 master 状态和其它 master 的意见

<b>案例需求</b>：在 7002 这个 slave 节点执行手动故障转移，重新夺回 master 地位。

步骤如下：

1）利用 redis-cli 连接 7002 这个节点

2）执行 cluster failover 命令

如图：

<div align="center"><img src="img/image-20210727160037766.png"></div>

效果：

<div align="center"><img src="img/image-20210727161152065.png"></div>

### RedisTemplate访问分片集群

RedisTemplate 底层同样基于 lettuce 实现了分片集群的支持，而使用的步骤与哨兵模式基本一致：

1）引入 redis 的 starter 依赖

2）配置分片集群地址

3）配置读写分离

与哨兵模式相比，其中只有分片集群的配置方式略有差异，如下：

```yaml
spring:
  redis:
    cluster:
      nodes:
        - 192.168.150.101:7001
        - 192.168.150.101:7002
        - 192.168.150.101:7003
        - 192.168.150.101:8001
        - 192.168.150.101:8002
        - 192.168.150.101:8003
```

# 最佳实践

- Redis 键值设计
- 批处理优化
- 服务端优化
- 集群

## 键值设计

### 优雅的Key设计

Redis 的 key 虽然可以自定义，但最好遵循下面的几个最佳实践约定：

- 遵循基本格式：`[业务名称]:[数据名]:[id]`
- 长度不超过 44 字节，节省内存。
- 不包含特殊字符

例如：登录业务，保存用户信息，key 可以这样设计 `login:user:10`

<b>优点</b>

- 可读性强、
- 避免 key 冲突、
- 方便管理
- 更节省内存：key 是 string 类型，底层编码包含 int、embstr 和 raw 三种。embstr 在小于 44 字节使用，采用连续内存空间，内存占用更小。

```shell
# 示例 如果是 4.0 版本以下的 redis embstr 的长度限制是 39 字节
set name 123
type num # string
object encoding num # "int" 类型

set name Jack
object encoding name # "embstr"
type name # string

set name aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa # 44 字节
object encoding name # "embstr"

set name aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa # 45 字节
object encoding name # "raw"
```

### 慎用BigKey

BigKey 通常以 Key 的大小和 key 中的成员的数量来综合判断的，如：

- Key 本身的数据量过大：一个 String 类型的 key，它的值为 5MB
- Key 中的成员数过多：一个 ZSET 类型的 Key，它的成员数量为 10,000 个
- Key 中成员的数据量过大：一个 Hash 类型的 key，它的成员数量虽然只有 1000 个，但这些成员的 value 总大小为 100MB

推荐值：

- 单个 key 的 value 小于 10KB
- 对于集合类型的 key，建议元素数量小于 1000

```shell
memory usage name # 衡量 key 占用的字节大小，不推荐使用，耗费 CPU，实践上预估一下就行。
# 字符串看长度，集合看大小，大致估计
```

<b>BigKey 的危害</b>

- 网络阻塞：对 BigKey 进行请求时，少量的 QPS 就可能导致带宽使用率被占满，导致 Redis 实例，甚至是物理机变慢
- 数据倾斜：BigKey 所在的 Redis 实例内存使用率远超其他实例，无法使数据分片的内存资源达到平衡
- Redis 阻塞：对元素较多的 hash、list、zset 等做运算会耗时较久，使主线程被阻塞
- CPU 压力：对 BigKey 的数据序列化和反序列化会导致 CPU 的使用率飙升，影像 Redis 实例和本机其他应用

<b>如何发现 BigKey</b>

- `redis-cli --bigkeys` 利用 redis-cli 提供的 --bigkeys 参数，可以遍历分析所有 key，并返回 key 的整体统计信息与每个数据的 Top1 的 big key
- `scan 扫描` 自行编程，利用 scan 扫描 Redis 中的所有 key，利用 strlen、hlen 等命令判断 key 的长度（不推荐使用 MEMORY USAGE，十分耗费 CPU）
- `第三方工具` 利用三方工具，如 Redis-Rdb-Tools 分析 RDB 快照文件，全面分析内存使用情况
- `网络监控` 自定义工具，监控进出 Redis 的网络数据，超出预警值时主动告警

<b>删除 BigKey</b>

BigKey 内存占用较多，删除这些 key 也需要耗费很长的时间，导致 Redis 主线程阻塞，引发一系列问题。

- redis 3.0 及以下版本，如果是集合类型，则遍历 BigKey 的元素，先逐个删除子元素，最后删除 BigKey
- redis 4.0 以后可以使用 unlink 异步删除 `unlink key`

### 恰当的数据类型

BigKey 往往都是业务设计不恰当导致的，选择更合适的数据类型，避免 BigKey。

<b>比如，存储一个 User 对象，我们有三种存储方式</b>

- json 字符串，实现简单，但是数据耦合强，不灵活，需要修改或获取部分字段的话需要传输所有的数据。

    |user:1|{"name":"Jack", "age":21}|

- 字段打散，可以灵活访问，但是原本一个 key 可以解决的，现在却需要多个 key，占用空间大（key 占用的空间）

    |user:1:name|Jack|	|user:1:age|21|

- hash，只需要一个 key，并且内部的 value 也是哈希结构的，并且内部的 value 采用的也是压缩链表，空间占用小，可以灵活访问对象的任意字段，缺点是代码相对复杂。

每一次存储 key，value 的时候，在 redis 内部是有很多元信息要保存的，原本一个 key 可以解决的，你用多个 key，元信息的内存消耗就上来了。

<b>假如有 hash 类型的 key，其中有 100 万对 field 和 value，field 是自增 id，这个 key 存在什么问题？如何优化？</b>

方案一：修改 hash entry 的数量上限

- 存在的问题：hash 的 entry 数量超过 500 时，会使用哈希表而不是 ZipList，内存占用较多。
- 可以通过 hash-max-ziplist-entries 配置 entry 上限。但是如果 entry 过多就会导致 BigKey。

```shell
config get hash-max-ziplist-entries # 获取配置的最大值

config set hash-max-ziplist-entries 1000 # 设置最大值为 1000
```

方案二：拆分为 string 类型

- 存在的问题：string 底层没有太多内存优化，内存占用较多（大量的 key 会产生大量与数据无关的元学习，占用内存空间）
- 想要批量获取数据比较麻烦

方案三：把一个大的 hash 拆分为小的 hash，分开存储<span style="color:orange">（数据分片）</span>。

- 比如，将 id/100 作为 key，这样数据就会分散到 100 个哈希中，将 id % 100 作为 field，这样每 100 个元素作为一个 Hash

hash、set、hashset 这些的 key 有内存优化（ziplist），推荐使用。临时存储信息，占用存储空间小的用 string 还是挺合适的，比如短信验证码。

### 总结

key 的最佳实践

- 固定格式：`[业务名]:[数据名]:[id]`
- 足够简短：不超过 44 字节
- 不包含特殊字符

value 的最佳实践

- 合理的拆分数据，拒绝 BigKey
- 选择合适的数据结构
- Hash 结构的 entry 数量不要超过 1000（默认是 500，可进行配置）
- 设置合理的超时时间

## 批处理优化

- pipeline
- 集群下的批处理

### Pipeline

大数据量的导入。

<b>单个命令的执行流程</b>

一次命令的响应时间 = 1 次往返的网络传输耗时 + 1 次 Redis 执行命令耗时

<b>N 条命令批量执行</b>

N 次命令的响应时间 = 1 次往返的网络传输耗时 + N 次 Redis 执行命令耗时

<b>批处理方案</b>

较少网络耗时，使用 mset、hmset 这些命令，如利用 mset 批量插入 10 万条数据，每次插入 1k 条。但是，不要在一次批处理中传输太多命令，否则单次命令占用带宽过多，会导致网络阻塞。

MSET 这些命令虽然可以进行批处理操作，但是只能操作部分数据类型，因此如果有对复杂数据类型的批处理需求，可以使用 Pipeline。

```java
void test(){
    Pipeline pipeline = jedis.pipelined();
    for(int i=1; i<= 100000; i++){
        pipeline.set("test:key_"+i,"value_"+i);
        // pipelien.zaddXX
        if(i%1000 == 0){
            // 每放入 1000 条命令，批量执行
            pipeline.sync();
        }
    }
}
```

M 操作比 Pipeline 快，因为 M 操作是 Redis 内置的操作，Redis 会把 M 操作的多组 key 和 value 作为一个原子性操作，一次性执行完；而 Pipeline 是把所有命令一起发过去，但未必是一起执行。

<b>总结</b>

批处理方案

- 原生 M 操作
- Pipeline 批处理操作

注意事项

- 批处理时不建议一次携带太多命令
- Pipeline 的多个命令直接不具备原子性

### 集群下的批处理

如 MSET 或 Pipeline 这样的批处理需要在一次请求中携带多条命令，而此时如果 Redis 是一个集群，那批处理命令的多个 key 必须落在一个插槽中，否则会导致执行失败。

| -        | 串行命令                       | 串行 slot                                                    | 并行 slot                                                    | hash_tag                                                     |
| -------- | ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 实现思路 | for 循环遍历，依次执行每个命令 | 在客户端计算每个 key 的 slot，将 slot 一致分为一组，每组都利用 Pipelien 批处理。串行执行各组命令。 | 在客户端计算每个 key 的 slot，将 slot 一致分为一组，每组都利用 Pipelien 批处理。并行执行各组命令。 | 将所有 key 设置相同的 hash_tag，则所有 key 的 slot 一定相同。 |
| 耗时     | N 次网络耗时+N 次命令耗时      | m 次网络耗时 + N 次命令耗时 <br>m = key 的 slot 个数         | 1 次网络耗时+N 次命令耗时                                    | 1 次网络耗时+N 次命令耗时                                    |
| 优点     | 实现简单                       | 耗时较短                                                     | 耗时非常短                                                   | 耗时非常短，实现简单                                         |
| 缺点     | 耗时非常久                     | 实现稍复杂 <br> slot 越多，耗时越久                          | 实现复杂                                                     | 容易出现数据倾斜                                             |







