# 初识MQ

[Spring AMQP](https://docs.spring.io/spring-amqp/docs/current/reference/html/#message-converters)

## 同步和异步通讯

微服务间通讯有同步和异步两种方式：

- 同步通讯：就像打电话，需要实时响应。

- 异步通讯：就像发邮件，不需要马上回复。

两种方式各有优劣，打电话可以立即得到响应，但是你却不能跟多个人同时通话。发送邮件可以同时与多个人收发邮件，但是往往响应会有延迟。

### 同步通讯

> <b>之前学习的 Feign 调用就属于同步方式，虽然调用可以实时得到结果，但存在下面的问题</b>

1️⃣<b>耦合度高：</b>每次加入新的需求，都要修改原来的代码。

2️⃣<b>性能下降：</b>调用者需要等待服务提供者响应，如果调用链过长则响应时间等于每次调用的时间之和。

3️⃣<b>资源浪费：</b>调用链中的每个服务在等待响应过程中，不能释放请求占用的资源，高并发场景下会极度浪费系统资源。

4️⃣<b>级联失败：</b>如果服务提供者出现问题，所有调用方都会跟着出问题，迅速导致整个微服务群故障。

> <b>同步调用的优点</b>

- 时效性较强，可以立即得到结果

> <b>同步调用的问题</b>

- 耦合度高
- 性能和吞吐能力下降
- 有额外的资源消耗
- 有级联失败问题

### 异步通讯

<b style="color:green">异步调用可以避免上述问题，异步调用常见实现是事件驱动模型</b>

以购买商品为例，用户支付后需要调用订单服务完成订单状态修改，调用物流服务，从仓库分配响应的库存并准备发货。

<div align="center"><img src="assets/image-20211029100941563.png"></div>

在事件驱动模型中用两种角色：<span style="color:red">一种是事件发布者，一种是事件订阅者。</span>

- 支付服务是事件发布者（publisher），在支付完成后只需要发布一个支付成功的事件（event），事件中带上订单 id。
- 订单服务、仓储服务、物流服务是事件订阅者（Consumer），订阅支付成功的事件，监听到事件后完成自己业务即可。

为了解除事件发布者与订阅者之间的耦合，两者并不是直接通信，而是有一个中间人（Broker）。发布者发布事件到 Broker，不关心谁来订阅事件。订阅者从 Broker 订阅事件，不关心谁发来的消息。



<div align="center"><img src="assets/image-20210422095356088.png"></div>

Broker 是一个像数据总线一样的东西，所有的服务要接收数据和发送数据都发到这个总线上，这个总线就像协议一样，让服务间的通讯变得标准和可控。类似于一个有协议的队列，生产者向里面放数据，消费者从里面拿数据。

> <b>好处</b>

- 提升吞吐量：无需等待订阅者处理完成，响应更快速

- 故障隔离：服务没有直接调用，不存在级联失败问题
- 调用间没有阻塞，不会造成无效的资源占用
- 耦合度极低，每个服务都可以灵活插拔，可替换
- 流量削峰：不管发布事件的流量波动多大，都由 Broker 接收，订阅者可以按照自己的速度去处理事件

> <b>缺点</b>

- 架构复杂了，业务没有明显的流程线，不好管理
- 需要依赖于 Broker 的可靠、安全、性能

好在现在开源软件或云平台上 Broker 的软件是非常成熟的，比较常见的一种就是 MQ 技术。

## 技术对比

MQ，中文是消息队列（MessageQueue），字面来看就是存放消息的队列。也就是事件驱动架构中的 Broker。

> <b>比较常见的 MQ 实现</b>

- ActiveMQ
- RabbitMQ
- RocketMQ
- <div align="center"><img src="assets/image-20210410103322874.png"></div>

> <b>几种常见 MQ 的对比</b>

| 说明 | RabbitMQ            | ActiveMQ                 | RocketMQ | Kafka  |
| ---------- | ----------------------- | ------------------------------ | ------------ | ---------- |
| 公司/社区  | Rabbit                  | Apache                         | 阿里         | Apache     |
| 开发语言   | Erlang                  | Java                           | Java         | Scala&Java |
| 协议支持   | AMQP，XMPP，SMTP，STOMP | OpenWire,STOMP，REST,XMPP,AMQP | 自定义协议   | 自定义协议 |
| 可用性     | 高                      | 一般                           | 高           | 高         |
| 单机吞吐量 | 一般                    | 差                             | 高           | 非常高     |
| 消息延迟   | 微秒级                  | 毫秒级                         | 毫秒级       | 毫秒以内   |
| 消息可靠性 | 高                      | 一般                           | 高           | 一般       |

<span style="color:blue">追求可用性：Kafka、 RocketMQ 、RabbitMQ</span>

<span style="color:blue">追求可靠性：RabbitMQ、RocketMQ</span>

<span style="color:blue">追求吞吐能力：RocketMQ、Kafka</span>

<span style="color:blue">追求消息低延迟：RabbitMQ、Kafka</span>

# RabbitMQ简介

erlang 开发的 AMQP 的开源实现

## MQ基本概念

### 概述

MQ 全称 Message Queue（消息队列），是在消息的传输过程中保存消息的容器。多用于分布式系统之间进行通信。

传统的调用方式是，A 系统向 B 系统发起远程调用，然后等待 B 系统的调用结果。

```mermaid
graph LR
A系统===>|远程调用|B系统
```

而加入消息队列后，调用方式变成了：A 系统发送消息给 MQ ，B 系统从 MQ 中取出消息进行消费。

```mermaid
graph LR
A系统/生产者===>中间件/消息队列===>B系统/消费者
```

<b>小结</b>

- MQ，消息队列，存储消息的中间件
- 分布式系统通信两种方式：直接远程调用 和 借助第三方 完成间接通信
- 发送方称为生产者，接收方称为消费者

### 优势

优势：应用解耦、异步提速、削峰填谷

劣势：系统可用性降低、系统复杂度提高、会存在数据一致性问题

> <b>应用解耦</b>

如果库存系统出现问题，那么调用库存系统的订单系统也可能会出现问题，会导致后面无法正常调用支付系统和物流系统。系统耦合度高，一处错误可能导致后面无法正常执行。如果需要增加系统的话，如增加一个 X 系统，那么需要修改订单系统的代码，订单系统的可维护性低。

<div align="center"><img src="assets/image-20221010152735753.png"></div>

使用 MQ 使得应用间解耦，提升容错性和可维护性。订单系统发送消息给 MQ，其他系统订阅 MQ 的消息，拿到消息后就执行。即便库存呢系统执行出错了，也不会影响其他系统的正常执行。而且，库存系统可能只是某几十秒内或几分钟内有问题，后面好了，可以继续从 MQ 中拿到那个未正常消费的消息，重新执行。如果需要增加 X 系统的话，只需要 X 系统从 MQ 中拿消息进行消费即可。

<div align="center"><img src="assets/image-20221010153203967.png"></div>

> <b>异步提速</b>

一个下单操作耗时：20 + 300 + 300 + 300 = 920ms。用户点击完下单按钮后，需要等待 920ms 才能得到下单响应，太慢！

<div align="center"><img src="assets/image-20221010153525277.png"></div>

加入消息队列后，用户点击完下单按钮后，只需等待 25ms 就能得到下单响应 (20+5=25ms)。提升用户体验和系统吞吐量（单位时间内处理请求的数目）。

<div align="center"><img src="assets/image-20221010153740378.png"></div>

> <b>削峰填谷</b>

在没有消息队列的情况下，如果请求瞬间增大，系统来不及处理可能会崩溃。

<div align="center"><img src="assets/image-20221010153929448.png"></div>

加入消息队列后，请求可以先打在消息队列中，然后系统在逐渐从 MQ 中拉取请求逐个处理。

<div align="center"><img src="assets/image-20221010154111483.png"></div>

使用了 MQ 之后，限制消费消息的速度为 1000，这样一来，高峰期产生的数据势必会被积压在 MQ 中，高峰就被“削”掉了，但是因为消息积压，在高峰期过后的一段时间内，消费消息的速度还是会维持在1000，直到消费完积压的消息，这就叫做“填谷”。

使用 MQ 后，可以提高系统稳定性。

### 劣势

<b>系统可用性降低</b>

系统引入的外部依赖越多，系统稳定性越差。一旦 MQ 宕机，就会对业务造成影响。如何保证 MQ 的高可用？(集群容错？定期将数据 IO 到磁盘，参考 Redis 的操作？)

<b>系统复杂度提高</b>

MQ 的加入大大增加了系统的复杂度，以前系统间是同步的远程调用，现在是通过 MQ 进行异步调用。如何保证消息没有被重复消费？怎么处理消息丢失情况？那么保证消息传递的顺序性？这块的设计完全可以参考 TCP 协议。

<b>一致性问题</b>

A 系统处理完业务，通过 MQ 给 B、C、D 三个系统发消息数据，如果 B 系统、C 系统处理成功，D 系统处理失败。如何保证消息数据处理的一致性？

### 使用MQ的条件

①生产者不需要从消费者处获得反馈。引入消息队列之前的直接调用，其接口的返回值应该为空，这才让明明下层的动作还没做，上层却当成动作做完了继续往后走，即所谓异步成为了可能。

②容许短暂的不一致性。

③确实是用了有效果。即解耦、提速、削峰这些方面的收益，超过加入 MQ，管理 MQ 这些成本。

## RabbitMQ

### AMQP协议

AMQP，即 Advanced Message Queuing Protocol（高级消息队列协议），是一个网络协议，是应用层协议的一个开放标准，为面向消息的中间件设计。基于此协议的客户端与消息中间件可传递消息，并不受客户端/中间件不同产品，不同的开发语言等条件的限制。2006 年，AMQP 规范发布。类比 HTTP。 

<div align="center"><img src="assets/image-20221010155442820.png"></div>

### RabbitMQ简介

2007年，Rabbit 技术公司基于 AMQP 标准开发的 RabbitMQ 1.0 发布。RabbitMQ 采用 Erlang 语言开发。Erlang 语言由 Ericson 设计，专门为开发高并发和分布式系统的一种语言，在电信领域使用广泛。RabbitMQ 基础架构如下图

<div align="center"><img src="assets/image-20221010155703332.png"></div>

- <b>Broker：</b>接收和分发消息的应用，RabbitMQ Server 就是 Message Broker
- <b>Virtual host：</b>出于多租户和安全因素设计的，把 AMQP 的基本组件划分到一个虚拟的分组中，类似于网络中的 namespace 概念。当多个不同的用户使用同一个 RabbitMQ server 提供的服务时，可以划分出多个 vhost，每个用户在自己的 vhost 创建 exchange／queue 等 
- <b>Connection：</b>publisher／consumer 和 broker 之间的 TCP 连接
- <b>Channel：</b>如果每一次访问 RabbitMQ 都建立一个 Connection，在消息量大的时候建立 TCP Connection 的开销将是巨大的，效率也较低。<span style="color:orange">Channel 是在 connection 内部建立的逻辑连接，如果应用程序支持多线程，通常每个 thread 创建单独的 channel 进行通讯，</span>AMQP method 包含了channel id 帮助客户端和 message broker 识别 channel，所以 channel 之间是完全隔离的。<span style="color:orange">Channel 作为轻量级的 Connection  极大减少了操作系统建立 TCP connection 的开销</span>
- <b>Exchange：</b>message 到达 broker 的第一站，根据分发规则，匹配查询表中的 routing key，分发消息到 queue 中去。常用的类型有：direct (point-to-point), topic (publish-subscribe) and fanout (multicast)
- <b>Queue：</b>消息最终被送到这里等待 consumer 取走
- <b>Binding：</b>exchange 和 queue 之间的虚拟连接，binding 中可以包含 routing key。Binding 信息被保存到 exchange 中的查询表中，用于 message 的分发依据。

### RabbitMQ工作模式

RabbitMQ 提供了 6 种工作模式：简单模式、work queues、Publish/Subscribe 发布与订阅模式、Routing 路由模式、Topics 主题模式、RPC 远程调用模式（远程调用，不太算 MQ；暂不作介绍）

官网对应模式介绍：https://www.rabbitmq.com/getstarted.html

### 安装RabbitMQ

直接使用 docker 安装 RabbitMQ

```shell
# wsl 启动 docker
sudo service docker start
# 拉取 mq
docker pull rabbitmq:3-management
# 启动 mq
docker run \
 -e RABBITMQ_DEFAULT_USER=payphone \ # 设置管理员账号
 -e RABBITMQ_DEFAULT_PASS=123321 \ # 设置管理员密码
 -v mq-plugins:/plugins \ # 设置 mq 数据卷
 --name mq \
 --hostname mq1 \
 -p 15672:15672 \ # 管理信息页面
 -p 5672:5672 \  # 通信端口
 -d \
 rabbitmq:3-management
```

方便执行的命令

```shell
docker run -e RABBITMQ_DEFAULT_USER=payphone -e RABBITMQ_DEFAULT_PASS=123321 -v mq-plugins:/plugins --name mq --hostname mq1 -p 15672:15672 -p 5672:5672 -d rabbitmq:3-management
```

MQ 的基本结构如下

<div align="center"><img src="assets/image-20210717162752376.png"></div>

> <b>RabbitMQ 中的一些角色</b>

- publisher：生产者
- consumer：消费者
- exchange：交换机，负责消息路由，即消息发送到那几个 queue 中。
- queue：队列，存储消息
- virtualHost：虚拟主机，隔离不同租户的 exchange、queue、消息的隔离

### JMS

- JMS 即 Java 消息服务（JavaMessage Service）应用程序接口，是一个 Java 平台中关于面向消息中间件的 API
- JMS 是 JavaEE 规范中的一种，类比 JDBC
- 很多消息中间件都实现了JMS 规范，例如：ActiveMQ。RabbitMQ 官方没有提供 JMS 的实现包，但是开源社区有

## 核心概念

- Message：消息由消息头和消息体组成。消息体是不透明的，而消息头则由一系列的可选属性组成，这些属性包括 routing-key（路由键）、priority、相对于其他消息的优先权、delivery-mode 等
- Publisher：消息的生产者，也是一个向交换器发布消息的客户端应用程序
- Exchange：交换器，用来接收生产者发送的消息并将这些消息路由给服务器中的队列。
    - 四种类型：direct（默认）、fanout、topic、headers
    - 不同类型的 Exchange 转发消息的策略有所区别
- Queue：消息队列
    - 用来保存消息，直到发送给消费者。他是消息的容器，也是消息的终点，一个消息可投入一个或多个队列。消息一直在队列里面，等待消费者连接到这个队列将其取走。
- Binding：绑定
    - 用于消息队列和交换器之间的关联。一个绑定就是基于路由键将交换器和消息队列连接起来的路由规则。可以将交换器理解成一个由绑定构成的路由表。
    - Exchange 和 Queue 的绑定可以是多对多的关系
- Connection
    - 网络连接，如一个 TCP 连接
- Channel：信道【解决多路复用】
    - 多路复用连接中的一条独立的双向数据流通道。信道是建立在真实的 TCP 连接内的虚拟连接。AMQP 命令都是通过信道发送出去的。不管是发布消息、订阅队列还是接收消息，这些动作都是通过信道完成。因为对于操作系统来说，建立和销毁 TCP 都是非常昂贵的开销，所有引入了信道的概念，以复用一条 TCP 连接。
- Consumer：消息的消费者
    - 表示一个从消息队列中取得消息的客户端应用程序。
- Virtual Host：虚拟主机
    - 表示一批交换器、消息队列和相关对象。虚拟主机是共享相同的身份认证和加密环境的独立服务器域。每个 vhost 本质是一个 mini 版的 RabbitMQ 服务器，拥有自己的队列、交换器、绑定和权限机制。vhost 是 AMQP 概念的基础，必须在连接时指定，RabbitMQ 默认的 vhost 是 /
- Broker：消息队列服务器实体
    - Pulisher --> Exchange ===> Queue --> Connection(Channel) -->Consumer

### 基本流程

<b>AMQP</b> 中消息的路由过程和 Java 的 <b>JMS</b> 存在一些差别，<b>AMQP</b> 中增加了 <b>Exchange</b> 和 <b>Binding</b> 的角色。生产者把消息发布到 <b>Exchange</b> 上，消息最终到达队列并被消费者接收，而 <b>Binding </b>决定交换器的消息应该发送到那个队列。交换器不同，绑定规则不同，那么消息的结果就不一样。

## 消息工作模式

RabbitMQ 官方提供了 5 个不同的 Demo 示例，对应了不同的消息模型

<div align="center"><img src="assets/image-20210717163332646.png"></div>

### BasicQueue

```mermaid
graph LR
Producer===>queue===>Consumer
```

- Producer：生产者，发送消息的程序
- Consumer：消费者，消息的接收者，会一直等待消息到来
- queue：消息队列，用于缓存消息；生产者向其中投递消息，消费者从其中取出消息

### WorkQueue

```mermaid
graph LR
Producer===>Queue===>Consumer1
Queue===>Consumer2
```

- <b>Work Queues：</b>与入门程序的简单模式相比，多了一个或一些消费端，多个消费端共同消费同一个队列中的消息。在一个队列中如果有多个消费者，那么消费者之间对于同一个消息的关系是竞争的关系。(消息获取的线程安全性由消息队列自身保证吗？)
- <b>应用场景：</b>对于任务过重或任务较多情况使用工作队列可以提高任务处理的速度。

### Public/Sub订阅模式

```mermaid
graph LR
Producer===>X===>Queue1===>C1
X===>Queue2===>C2
```

在发布订阅模型中，会分发一个消息给多个消费者，多了一个 Exchange 角色，而且过程略有变化：

- P：生产者，也就是要发送消息的程序，但是不再发送到队列中，而是发给 X（交换机）
- C：消费者，消息的接收者，会一直等待消息到来
- Queue：消息队列，接收消息、缓存消息
- Exchange：交换机（X）。一方面，接收生产者发送的消息。另一方面，知道如何处理消息，例如递交给某个特别队列、递交给所有队列、或是将消息丢弃。到底如何操作，取决于 Exchange 的类型。Exchange 有常见以下 3 种类型：
    - Fanout：广播，将消息交给所有绑定到交换机的队列，routing key 为空。
    - Direct：定向，把消息交给符合指定 routing key 的队列
    - Topic：通配符，把消息交给符合 routing pattern（路由模式） 的队列
- Exchange（交换机）只负责转发消息，不具备存储消息的能力，因此如果没有任何队列与 Exchange 绑定，或者没有符合路由规则的队列，那么消息会丢失！

<b>编写逻辑</b>

创建交换机、创建队列，然后将队列和交换机进行绑定。

<b>总结</b>

- 交换机需要与队列进行绑定，绑定之后；一个消息可以被多个消费者都收到。
- 发布订阅模式与工作队列模式的区别
    - 工作队列模式不用定义交换机，而发布/订阅模式需要定义交换机
    - 发布/订阅模式的生产方是面向交换机发送消息，工作队列模式的生产方是面向队列发送消息 (底层使用默认交换机)
    - 发布/订阅模式需要设置队列和交换机的绑定，工作队列模式不需要设置，实际上工作队列模式会将队列绑定到默认的交换机

###  Routing路由模式

发布订阅模式是将一个消息发送给多个消费者，而 Routing 路由模式则是可以进行更为细致的发送，如将 error 消息发送给两个消费者，info、warning 这种消息则只发送给一个消费者。

<div align="center"><img src="assets/image-20221010173810605.png"></div>

- <span style="color:orange">队列与交换机的绑定，不能是任意绑定了，而是要指定一个 RoutingKey（路由 key）</span>
- 消息的发送方在向 Exchange 发送消息时，也必须指定消息的 RoutingKey
- Exchange 不再把消息交给每一个绑定的队列，而是根据消息的 Routing Key 进行判断，只有队列的 Routingkey 与消息的 Routing key 完全一致，才会接收到消息

<b>小结</b>

<b>Routing</b> 模式要求队列在绑定交换机时要指定 <b>routing key</b>，消息会转发到符合 routing key 的队列。

### Topics通配符模式

<b>模式说明</b>

- Topic 类型与 Direct 相比，都是可以根据 RoutingKey 把消息路由到不同的队列。只不过 Topic 类型 Exchange 可以让队列在绑定 Routing key 的时候使用<span style="color:red">通配符！</span>
- Routingkey 一般都是有一个或多个单词组成，多个单词之间以 `.` 分割，例如：item.insert 
- 通配符规则：`#` 匹配一个或多个词，`*` 匹配不多不少恰好 1 个词，例如：`item.#` 能够匹配 item.insert.abc 或者 item.insert，`item.*` 只能匹配 item.insert

<div align="center"><img src="assets/image-20221010174957440.png"></div>

- 红色 Queue：绑定的是 `usa.#` ，因此凡是以 `usa.` 开头的 routing key 都会被匹配到

- 黄色 Queue：绑定的是 `#.news` ，因此凡是以 `.news` 结尾的 routing key 都会被匹配

<b>总结</b>

Topic 主题模式可以实现 Pub/Sub 发布与订阅模式和 Routing 路由模式的功能，只是 Topic 在配置 routing key 的时候可以使用通配符，显得更加灵活。

### 工作模式总结

- 简单模式 HelloWorld：一个生产者、一个消费者，不需要设置交换机（使用默认的交换机）。
- 工作队列模式 Work Queue：一个生产者、多个消费者（竞争关系），不需要设置交换机（使用默认的交换机）。
- 发布订阅模式 Publish/subscribe：需要设置类型为 fanout 的交换机，并且交换机和队列进行绑定，当发送消息到交换机后，交换机会将消息发送到绑定的队列。
- 路由模式 Routing：需要设置类型为 direct 的交换机，交换机和队列进行绑定，并且指定 routing key，当发送消息到交换机后，交换机会根据 routing key 将消息发送到对应的队列。
- 通配符模式 Topic：需要设置类型为 topic 的交换机，交换机和队列进行绑定，并且指定通配符方式的 routing key，当发送消息到交换机后，交换机会根据 routing key 将消息发送到对应的队列。

## 入门案例

简单队列模式的模型图

```mermaid
graph LR
publisher===>queue===>consumer
```

官方的 HelloWorld 是基于最基础的消息队列模型来实现的，只包括三个角色：

- publisher：消息发布者，将消息发送到队列 queue
- queue：消息队列，负责接受并缓存消息
- consumer：订阅队列，处理队列中的消息

导入依赖

```xml
<dependency>
    <groupId>com.rabbitmq</groupId>
    <artifactId>amqp-client</artifactId>
    <version>5.16.0</version>
</dependency>
```

### publisher实现

思路：

- 建立连接
- 创建 Channel
- 声明队列
- 发送消息
- 关闭连接和 Channel

```java
package com.ex.rabbitmq.quick_test;

import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;

public class PublisherTest {
    static Logger log = LoggerFactory.getLogger("publisher");
    static String queueName = "simple.queue";

    public static void main(String[] args) throws Exception {
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("127.0.0.1");
        factory.setPort(5672);
        factory.setUsername("payphone");
        factory.setVirtualHost("/");
        factory.setPassword("123321");
        Connection connection = null;
        Channel channel = null;
        try {
            connection = factory.newConnection();
            log.debug(String.valueOf(connection));
            channel = connection.createChannel();
            // 创建队列（debug 执行完此语句后，队列 simple.queue 队列成功创建）
            channel.queueDeclare(queueName, false, false, false, null);
            for (int i = 0; i < 10; i++) {
                channel.basicPublish("", queueName, null, "hello rabbitmq".getBytes(StandardCharsets.UTF_8));
                log.debug("消息{}发送成功", "hello rabbitmq");
            }
        } finally {
            if (channel != null) {
                channel.close();
            }
            if (connection != null) {
                connection.close();
            }
        }
    }
}
```

### consumer实现

- 建立连接
- 创建 Channel
- 声明队列
- 订阅消息

```java
package com.ex.rabbitmq.quick_test;

import com.rabbitmq.client.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class ConsumerTest {
    static Logger log = LoggerFactory.getLogger("consumerTest");
    static String queueName = "simple.queue";

    public static void main(String[] args) throws IOException, TimeoutException, InterruptedException {
        // 设置消息队列参数
        ConnectionFactory factory = new ConnectionFactory();
        factory.setHost("127.0.0.1");
        factory.setPort(5672);
        factory.setUsername("payphone");
        factory.setVirtualHost("/");
        factory.setPassword("123321");

        Connection connection = null;
        Channel channel = null;
        try {
            // 建立连接并创建通道
            connection = factory.newConnection();
            channel = connection.createChannel();
            channel.queueDeclare(queueName, false, false, false, null);

            Channel finalChannel = channel;
            // 消费消息
            channel.basicConsume(queueName, new DefaultConsumer(finalChannel) {
                @Override
                public void handleDelivery(String consumerTag, Envelope envelope, AMQP.BasicProperties properties, byte[] body) throws IOException {
                    String message = new String(body);
                    log.debug("接收到消息:{}", message);
                    // 设置 ack 回复，表示接收到了消息
                    finalChannel.basicAck(envelope.getDeliveryTag(), false);
                }
            });
            log.debug("waiting for message");

        } finally {
            TimeUnit.SECONDS.sleep(50);
            channel.close();
            connection.close();
        }
    }
}

```

## 总结

> 基本消息队列的消息发送流程

1. 建立 connection

2. 创建 channel

3. 利用 channel 声明队列

4. 利用 channel 向队列发送消息

> 基本消息队列的消息接收流程

1. 建立 connection

2. 创建 channel

3. 利用 channel 声明队列

4. 定义 consumer 的消费行为 handleDelivery()

5. 利用 channel 将消费者与队列绑定

# SpringAMQP

SpringAMQP 是基于 RabbitMQ 封装的一套模板，并且还利用 SpringBoot 对其实现了自动装配，使用起来非常方便。

SpringAMQP 的官方地址：https://spring.io/projects/spring-amqp

SpringAMQP 提供了三个功能：

- <b>自动声明队列、交换机及其绑定关系</b>
- 基于注解的监听器模式，异步接收消息
- 封装了 RabbitTemplate 工具，用于发送消息

问题：AMQP 何时创建的消息队列？目前测试的是，消息提供者如果只是用 @Bean 声明消息队列是不会创建的，只有发送消息到队列的时候才会创建。消息监听者需要在监听服务中也配置 @Bean 声明，这样在监听的时候就会创建消息队列了。

## Basic Queue

Basic Queue，简单队列模型。

在父工程 mq-demo 中引入依赖

```xml
<!--AMQP依赖，包含RabbitMQ-->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-amqp</artifactId>
</dependency>
```

### 消息发送

首先配置 MQ 地址，在 publisher 服务的 application.yml 中添加配置

```yaml
spring:
  rabbitmq:
    host: 192.168.150.101 # 主机名
    port: 5672 # 端口
    virtual-host: / # 虚拟主机
    username: payphone # 用户名
    password: 123321 # 密码
```

配置 MQ 的队列

```java
import org.springframework.amqp.core.Queue; // 注意，是这个类
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AmqpQueueConfig {

    @Bean
    public Queue simpleQueue() {
        return new Queue("simple.queue");
    }

}
```

然后在 publisher 服务中编写测试类 SpringAmqpTest，并利用 RabbitTemplate 实现消息发送

```java
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
@SpringBootTest
public class SpringAmqpTest {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    @Test
    public void testSimpleQueue() {
        // 队列名称
        String queueName = "simple.queue";
        // 消息
        String message = "hello, spring amqp!";
        // 发送消息，图方便的话也可以自行通过 mq 的网络控制台创建一个 simple.queue
        rabbitTemplate.convertAndSend(queueName, message);
    }
}
```

### 消息接收

首先配置 MQ 地址，在 consumer 服务的 application.yml 中添加配置

```yaml
spring:
  rabbitmq:
    host: 192.168.150.101 # 主机名
    port: 5672 # 端口
    virtual-host: / # 虚拟主机
    username: payphone # 用户名
    password: 123321 # 密码
```

然后在 consumer 服务的 `cn.mq.listener` 包中新建一个类 SpringRabbitListener，代码如下

```java
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

@Component
public class SpringRabbitListener {

    @RabbitListener(queues = "simple.queue")
    public void listenSimpleQueueMessage(String msg) throws InterruptedException {
        System.out.println("spring 消费者接收到消息：【" + msg + "】");
    }
}
```

### 测试

启动 consumer 服务，然后在 publisher 服务中运行测试代码，发送 MQ 消息

## WorkQueue

Work queues，也被称为（Task queues），任务模型。简单来说就是<b>让多个消费者绑定到一个队列，共同消费队列中的消息</b>。

```mermaid
graph LR
publisher===>queue
queue===>consumer1
queue===>consumer2
```

<span style="color:orange">当消息处理比较耗时的时候，可能生产消息的速度会远远大于消息的消费速度。长此以往，消息就会堆积越来越多，无法及时处理。此时就可以使用 work 模型，多个消费者共同处理消息处理，速度就能大大提高了。</span>

### 消息发送

模拟大量消息堆积现象。在 publisher 服务中的 SpringAmqpTest 类中添加一个测试方法：

```java
/**
* workQueue
* 向队列中不停发送消息，模拟消息堆积。
*/
@Test
public void testWorkQueue() throws InterruptedException {
    // 队列名称
    String queueName = "simple.queue";
    // 消息
    String message = "hello, message_";
    for (int i = 0; i < 50; i++) {
        // 发送消息
        rabbitTemplate.convertAndSend(queueName, message + i);
        Thread.sleep(20);
    }
}
```

### 消息接收

要模拟多个消费者绑定同一个队列，我们在 consumer 服务的 SpringRabbitListener 中添加 2 个新的方法：

```java
@RabbitListener(queues = "simple.queue") // 绑定到同一个队列中
public void listenWorkQueue1(String msg) throws InterruptedException {
    System.out.println("消费者1接收到消息：【" + msg + "】" + LocalTime.now());
    Thread.sleep(20);
}

@RabbitListener(queues = "simple.queue") // 绑定到同一个队列中
public void listenWorkQueue2(String msg) throws InterruptedException {
    System.err.println("消费者2........接收到消息：【" + msg + "】" + LocalTime.now());
    Thread.sleep(200);
}
```

注意到这个消费者 sleep 了 1000 秒，模拟任务耗时。

### 测试

启动 ConsumerApplication 后，在执行 publisher 服务中刚刚编写的发送测试方法 testWorkQueue。可以看到消费者 1 很快完成了自己的 25 条消息。消费者 2 却在缓慢的处理自己的 25 条消息。也就是说消息是平均分配给每个消费者，并没有考虑到消费者的处理能力。这样显然是有问题的。

### 能者多劳

在 Spring 中有一个简单的配置，可以解决这个问题。我们修改 consumer 服务的 application.yml 文件，添加配置

```yaml
spring:
  rabbitmq:
    listener:
      simple:
        prefetch: 1 # 每次只能获取一条消息，处理完成才能获取下一个消息
```

### 总结

Work 模型的使用

- 多个消费者绑定到一个队列，同一条消息只会被一个消费者处理
- 通过设置 prefetch 来控制消费者预取的消息数量

## 发布/订阅

发布订阅的模型如图

```mermaid
graph LR
publisher===>exchange
exchange===>queue1
exchange===>queue2
queue1===>consumer1
queue1===>consumer2
queue2===>consumer3
```

可以看到，在订阅模型中，多了一个 exchange 角色，而且过程略有变化：

- Publisher：生产者，也就是要发送消息的程序，但是不再发送到队列中，而是发给 X（交换机）
- Exchange：交换机，图中的 X。一方面，接收生产者发送的消息。另一方面，知道如何处理消息，例如递交给某个特别队列、递交给所有队列、或是将消息丢弃。到底如何操作，取决于 Exchange 的类型。
- <b style="color:red">Exchange 有以下 3 种类型：</b>
  - <span style="color:orange">Fanout：广播，将消息交给所有绑定到交换机的队列</span>
  - <span style="color:orange">Direct：定向，把消息交给符合指定 routing key 的队列</span>
  - <span style="color:orange">Topic：通配符，把消息交给符合 routing pattern（路由模式） 的队列</span>
- Consumer：消费者，与以前一样，订阅队列，没有变化
- Queue：消息队列也与以前一样，接收消息、缓存消息。

<b style="color:red">Exchange（交换机）只负责转发消息，不具备存储消息的能力</b>，因此如果没有任何队列与 Exchange 绑定，或者没有符合路由规则的队列，那么消息会丢失！

## Fanout

Fanout，英文翻译是扇出，我觉得在 MQ 中叫广播更合适。

```mermaid
graph LR
publisher===>exchange[exchange-Fanout&nbsp模式]
exchange===>queue1===>consumer1
exchange===>queue2===>consumer2
```

<b>在广播模式下，消息发送流程是这样的</b>

- 1）可以有多个队列
- 2）每个队列都要绑定到 Exchange（交换机）
- 3）生产者发送的消息，只能发送到交换机，交换机来决定要发给哪个队列，生产者无法决定
- 4）交换机把消息发送给绑定过的所有队列
- 5）订阅队列的消费者都能拿到消息

<b>我们的计划是这样的</b>

- 创建一个交换机 itcast.fanout，类型是 Fanout
- 创建两个队列 fanout.queue1 和 fanout.queue2，绑定到交换机 itcast.fanout

```mermaid
graph LR
publisher===>exchange[exchange/itcast.fanout]
exchange===>fantout.queue1===>consumer1
exchange===>fantout.queue2===>consumer2
```

### 声明队列和交换机

Spring 提供了一个接口 Exchange，来表示所有不同类型的交换机

<div align="center"><img src="assets/image-20210717165552676.png"></div>

在 consumer 中创建一个类，声明队列和交换机

```java
package cn.itcast.mq.config;

import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.FanoutExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class FanoutConfig {
    /**
     * 声明交换机
     * @return Fanout类型交换机
     */
    @Bean
    public FanoutExchange fanoutExchange(){
        return new FanoutExchange("itcast.fanout");
    }

    // 第1个队列
    @Bean
    public Queue fanoutQueue1(){
        return new Queue("fanout.queue1");
    }

    // 绑定队列和交换机
    @Bean
    public Binding bindingQueue1(Queue fanoutQueue1, FanoutExchange fanoutExchange){
        return BindingBuilder.bind(fanoutQueue1).to(fanoutExchange);
    }

    // 第2个队列
    @Bean
    public Queue fanoutQueue2(){
        return new Queue("fanout.queue2");
    }

    // 绑定队列和交换机
    @Bean
    public Binding bindingQueue2(Queue fanoutQueue2, FanoutExchange fanoutExchange){
        return BindingBuilder.bind(fanoutQueue2).to(fanoutExchange);
    }
}
```

### 消息发送

在 publisher 服务的 SpringAmqpTest 类中添加测试方法：

```java
@Test
public void testFanoutExchange() {
    // 队列名称
    String exchangeName = "itcast.fanout";
    // 消息
    String message = "hello, everyone!";
    rabbitTemplate.convertAndSend(exchangeName, "", message);
}
```

### 消息接收

在 consumer 服务的 SpringRabbitListener 中添加两个方法，作为消费者：

```java
@RabbitListener(queues = "fanout.queue1")
public void listenFanoutQueue1(String msg) {
    System.out.println("消费者1接收到Fanout消息：【" + msg + "】");
}

@RabbitListener(queues = "fanout.queue2")
public void listenFanoutQueue2(String msg) {
    System.out.println("消费者2接收到Fanout消息：【" + msg + "】");
}
```

### 总结

交换机的作用是什么？

- 接收 publisher 发送的消息
- 将消息按照规则路由到与之绑定的队列
- 不能缓存消息，路由失败，消息丢失
- FanoutExchange 的会将消息路由到每个绑定的队列

声明队列、交换机、绑定关系的 Bean 是什么？

- Queue
- FanoutExchange
- Binding

## Direct

在 Fanout 模式中，一条消息，会被所有订阅的队列都消费。但是，在某些场景下，我们希望不同的消息被不同的队列消费。这时就要用到 Direct 类型的 Exchange。

<div align="center"><img src="assets/image-20210717170041447.png"></div>

 <b>在 Direct 模型下</b>

- 队列与交换机的绑定，不能是任意绑定了，而是要指定一个`RoutingKey`（路由 key）
- 消息的发送方在，向 Exchange 发送消息时，也必须指定消息的 `RoutingKey`。
- Exchange 不再把消息交给每一个绑定的队列，而是根据消息的 `Routing Key` 进行判断，只有队列的 `Routingkey` 与消息的 `Routing key` 完全一致，才会接收到消息

<b>案例需求如下</b>

1. 利用 @RabbitListener 声明 Exchange、Queue、RoutingKey

2. 在 consumer 服务中，编写两个消费者方法，分别监听 direct.queue1 和 direct.queue2

3. 在 publisher 中编写测试方法，向 itcast.direct 发送消息

<div align="center"><img src="assets/image-20210717170223317.png"></div>

### 基于注解声明队列和交换机

基于 @Bean 的方式声明队列和交换机比较麻烦，Spring 还提供了基于注解方式来声明。在 consumer 的 SpringRabbitListener 中添加两个消费者，同时基于注解来声明队列和交换机：

```java
@RabbitListener(bindings = @QueueBinding(
    value = @Queue(name = "direct.queue1"),
    exchange = @Exchange(name = "itcast.direct", type = ExchangeTypes.DIRECT),
    key = {"red", "blue"}
))
public void listenDirectQueue1(String msg){
    System.out.println("消费者接收到direct.queue1的消息：【" + msg + "】");
}

@RabbitListener(bindings = @QueueBinding(
    value = @Queue(name = "direct.queue2"),
    exchange = @Exchange(name = "itcast.direct", type = ExchangeTypes.DIRECT),
    key = {"red", "yellow"}
))
public void listenDirectQueue2(String msg){
    System.out.println("消费者接收到direct.queue2的消息：【" + msg + "】");
}
```

### 消息发送

在 publisher 服务的 SpringAmqpTest 类中添加测试方法：

```java
@Test
public void testSendDirectExchange() {
    // 交换机名称
    String exchangeName = "itcast.direct";
    // 消息
    String message = "红色警报！日本乱排核废水，导致海洋生物变异，惊现哥斯拉！";
    // 发送消息
    rabbitTemplate.convertAndSend(exchangeName, "red", message);
}
```

### 总结

描述下 Direct 交换机与 Fanout 交换机的差异？

- Fanout 交换机将消息路由给每一个与之绑定的队列
- Direct 交换机根据 RoutingKey 判断路由给哪个队列
- 如果多个队列具有相同的 RoutingKey，则与 Fanout 功能类似

基于 @RabbitListener 注解声明队列和交换机有哪些常见注解？

- @Queue
- @Exchange

## Topic

### 说明

`Topic` 类型的 `Exchange` 与 `Direct` 相比，都是可以根据 `RoutingKey` 把消息路由到不同的队列。只不过 `Topic` 类型 `Exchange` 可以让队列在绑定 `Routing key` 的时候使用通配符！

`Routingkey` 一般都是有一个或多个单词组成，多个单词之间以 `.` 分割，例如： `item.insert`

 通配符规则：

`#`：匹配一个或多个词

`*`：匹配不多不少恰好 1 个词

举例：

`item.#`，能够匹配 `item.spu.insert` 或者 `item.spu`

`item.*`，只能匹配 `item.spu`

图示：

<div align="center"><img src="assets/image-20210717170705380.png"></div>

- Queue1：绑定的是 `china.#` ，因此凡是以 `china.` 开头的 `routing key` 都会被匹配到。包括 china.news 和 china.weather
- Queue2：绑定的是 `#.news` ，因此凡是以 `.news` 结尾的 `routing key` 都会被匹配。包括 china.news 和 japan.news

案例需求：

xxxxxx

实现思路如下：

1. 并利用 @RabbitListener 声明 Exchange、Queue、RoutingKey

2. 在 consumer 服务中，编写两个消费者方法，分别监听 topic.queue1 和 topic.queue2

3. 在 publisher 中编写测试方法，向 itcast. topic 发送消息

<div align="center"><img src="assets/image-20210717170829229.png"></div>

### 消息发送

在 publisher 服务的 SpringAmqpTest 类中添加测试方法：

```java
/**
 * topicExchange
 */
@Test
public void testSendTopicExchange() {
    // 交换机名称
    String exchangeName = "itcast.topic";
    // 消息
    String message = "喜报！孙悟空大战哥斯拉，胜!";
    // 发送消息
    rabbitTemplate.convertAndSend(exchangeName, "china.news", message);
}
```

### 消息接收

在 consumer 服务的 SpringRabbitListener 中添加方法：

```java
@RabbitListener(bindings = @QueueBinding(
    value = @Queue(name = "topic.queue1"),
    exchange = @Exchange(name = "itcast.topic", type = ExchangeTypes.TOPIC),
    key = "china.#"
))
public void listenTopicQueue1(String msg){
    System.out.println("消费者接收到topic.queue1的消息：【" + msg + "】");
}

@RabbitListener(bindings = @QueueBinding(
    value = @Queue(name = "topic.queue2"),
    exchange = @Exchange(name = "itcast.topic", type = ExchangeTypes.TOPIC),
    key = "#.news"
))
public void listenTopicQueue2(String msg){
    System.out.println("消费者接收到topic.queue2的消息：【" + msg + "】");
}
```

### 总结

描述下 Direct 交换机与 Topic 交换机的差异？

- Topic 交换机接收的消息 RoutingKey 必须是多个单词，以 `*.*` 分割
- Topic 交换机与队列绑定时的 bindingKey 可以指定通配符
- `#`：代表 0 个或多个词
- `*`：代表 1 个词

## 消息转换器

之前说过，Spring 会把你发送的消息序列化为字节发送给 MQ，接收消息的时候，还会把字节反序列化为 Java 对象。

<div align="center"><img src="assets/image-20200525170410401.png"></div>

只不过，默认情况下 Spring 采用的序列化方式是 JDK 序列化。众所周知，JDK 序列化存在下列问题：

- 数据体积过大
- 有安全漏洞
- 可读性差

我们来测试一下。

### 测试默认转换器

我们修改消息发送的代码，发送一个 Map 对象：

```java
@Test
public void testSendMap() throws InterruptedException {
    // 准备消息
    Map<String,Object> msg = new HashMap<>();
    msg.put("name", "Jack");
    msg.put("age", 21);
    // 发送消息
    rabbitTemplate.convertAndSend("simple.queue","", msg);
}
```

停止 consumer 服务，发送消息后查看控制台：

<div align="center"><img src="assets/image-20210422232835363.png"></div>

### 配置JSON转换器

显然，JDK 序列化方式并不合适。我们希望消息体的体积更小、可读性更高，因此可以使用 JSON 方式来做序列化和反序列化。不过我目前都是先手动 JSON 序列化数据后再发送。

在 publisher 和 consumer 两个服务中都引入依赖：

```xml
<dependency>
    <groupId>com.fasterxml.jackson.dataformat</groupId>
    <artifactId>jackson-dataformat-xml</artifactId>
    <version>2.9.10</version>
</dependency>
```

配置消息转换器，在启动类中添加一个 Bean 即可：

```java
@Bean
public MessageConverter jsonMessageConverter(){
    return new Jackson2JsonMessageConverter();
}
```

# RabbitMQ高级特性

消息队列在使用过程中，面临着很多实际问题需要思考

- 消息可靠性问题：如何确保消息至少被消费一次
- 延迟消息问题：如何实现消息的延迟投递
- 消息堆积问题：如何解决数百万消息堆积，无法及时消费的问题
- 高可用问题：如何避免单点的 MQ 故障而导致的不可用问题

## 消息可靠性

消息从发送，到消费者接收，会经理多个过程

```mermaid
graph LR
publisher===>exchange
exchange===>queue1===>consumer1
exchange===>queue2===>consumer2
```

<b style="color:red">其中的每一步都可能导致消息丢失，常见的丢失原因包括</b>

- 发送时丢失
    - 生产者发送的消息未送达 exchange
    - 消息到达 exchange 后未到达 queue
- MQ 宕机，queue 将消息丢失
- consumer 接收到消息后未消费就宕机

<b style="color:red">针对这些问题，RabbitMQ 分别给出了解决方案</b>

- 生产者确认机制
- mq 持久化
- 消费者确认机制
- 失败重试机制

### 生产者消息确认

RabbitMQ 提供了 publisher confirm 机制来避免消息发送到 MQ 过程中丢失，只有当消息成功到了 queue 中，消息才算发送成功了。消息发送到 MQ 以后，会返回一个结果给发送者，表示消息是否处理成功。

<b>MQ 发送返回的结果有两种</b>

- publisher-confirm 发送者确认：producer 发送消息后对发送方设置一个 confirmCallback 监听，消息达到/不到达 exchange 都会被执行，告诉 producer 消息投递的结果。
    - 消息成功投递到交换机，返回 ack（只能说明到了 exchange，投递成功了一半）
    - 消息未投递到交换机，返回 nack

- publisher-return 发送者回执：exchange-->queue 的回退模式，投递失败则会返回一个 returnCallback，告诉 exchange 自己是否接收到了消息。
    - 消息投递到交换机了，但是没有路由到队列。返回 ACK，及路由失败原因。一般，消息能到交换机，那么就可以到队列，如果到不了队列，说明是代码写错了，没匹配正确的队列！


<div align="center"><img src="assets/image-20210718160907166.png"></div>

- 消息从 publisher 到 exchange 则会返回一个 confirmCallback，不管消息是否成功到达 exchange，都会调用 confirmCallback，成功则返回 true，失败则返回 false。
- 消息从 exchange-->queue 投递失败则会返回一个 returnCallback，不管消息是否成功到达 exchange，都会调用 returnCallback，成功则返回 true，失败则返回 false。

我们可以利用这两个 callback 控制消息的可靠性投递。

<b span style="color:red">注意：确认机制发送消息时，需要给每个消息设置一个全局唯一 id，以区分不同消息，避免 ack 冲突。</b>

#### 配置文件

配置文件中开启确认模式

```yml
spring:
  rabbitmq:
   # 新版的开启 confirm 确认模式
    publisher-confirm-type: correlated
    # 开启 return 退回模式
    publisher-returns: true
    template:
      mandatory: true
      
# 旧版本 开启 confirm 确认模式
# publisher-confirms: true
```

- `publish-confirm-type`：开启 publisher-confirm，这里支持两种类型：
    - `simple`：同步等待 confirm 结果，直到超时
    - `correlated`：异步回调，定义 ConfirmCallback，MQ 返回结果时会回调这个 ConfirmCallback
- `publish-returns`：开启 publish-return 功能，同样是基于 callback 机制，不过是定义 ReturnCallback
- `template.mandatory`：定义消息路由失败时的策略。true 则调用 ReturnCallback，将消息退回给生产者；false 则直接丢弃消息

#### 定义Return回调

每个 RabbitTemplate 只能配置一个 ReturnCallback，因此需要在项目加载时配置。修改 publisher 服务，添加配置类

```java
package cn.itcast.mq.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration // ApplicationContextAware bean 工厂通知，从工厂中取出 bean，设置 ReturnCallback
public class CommonConfig implements ApplicationContextAware {
    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        // 获取RabbitTemplate
        RabbitTemplate rabbitTemplate = applicationContext.getBean(RabbitTemplate.class);
        // 设置ReturnCallback
        rabbitTemplate.setReturnCallback((message, replyCode, replyText, exchange, routingKey) -> {
            // 投递失败，记录日志
            log.info("消息发送失败，应答码{}，原因{}，交换机{}，路由键{},消息{}",
                     replyCode, replyText, exchange, routingKey, message.toString());
            // 如果有业务需要，可以重发消息，拿到了失败的一些信息，也可以失败重试，重发数据。
            // 如果数据可靠性要求高，应该重发！重发可以使用Spring-retry，也可以自己写业务代码
        });
    }
}
```

使用 rabbitTemplate.setReturnCallback 设置退回函数，当消息从 exchange 路由到 queue 失败后，如果设置了 rabbitTemplate.setMandatory(true) 参数（或 yml 中设置为 true），则会将消息退回给 producer。并执行回调函数 returnedMessage（即 lambda 表达式中的代码）。

#### 定义ConfirmCallback

ConfirmCallback 可以在发送消息时指定，因为每个业务处理 confirm 成功或失败的逻辑不一定相同。

在 publisher 服务的 cn.payphone.mq.spring.SpringAmqpTest 类中，定义一个单元测试方法：

```java
public void testSendMessage2SimpleQueue() throws InterruptedException {
    // 1.消息体
    String message = "hello, spring amqp!";
    // 2.全局唯一的消息ID，需要封装到CorrelationData中
    CorrelationData correlationData = new CorrelationData(UUID.randomUUID().toString());
    // 3.添加callback
    correlationData.getFuture().addCallback(
        result -> {
            if(result.isAck()){
                // 3.1.ack，消息成功
                log.debug("消息发送成功, ID:{}", correlationData.getId());
            }else{
                // 3.2.nack，消息失败
                log.error("消息发送失败, ID:{}, 原因{}",correlationData.getId(), result.getReason());
            }
        },
        ex -> log.error("消息发送异常, ID:{}, 原因{}",correlationData.getId(),ex.getMessage())
    );
    // 4.发送消息: 交换机名称。routingKey 名称，消息体，correlationData 封装了消息的唯一 ID 和 Callback
    rabbitTemplate.convertAndSend("task.direct", "task", message, correlationData);

    // 休眠一会儿，等待ack回执
    Thread.sleep(2000);
}
```

设置回调函数。当消息发送到 exchange 后回调 confirm 方法。在方法中判断 ack，如果为 true，则发送成功，如果为 false，则发送失败，需要处理。

只要数据正常到了 exchange，那么消息就一定可以到 queue 中（除非发生极端情况，比如 mq 突然宕机没了），除非代码写错了 queue 的名字不对。因此我们需要考虑的往往是消息无法正常到达 exchange 的情况；处理的方式有两种，一种是 Spring 的重试机制，另一种就是自己写业务代码进行处理。

```properties
# spring 的重试方式
spring.rabbitmq.template.retry.enabled=true 
# 每间隔隔 2s 重发一次。第一次间隔2s重试，第二次间隔 4s，第三次间隔 6s
spring.rabbitmq.template.retry.initial-interval=2000ms
# 重试的最大次数
spring.rabbitmq.template.retry.max-attempts=5
spring.rabbitmq.template.retry.multiplier=2
```

如果是自己写代码进行业务重试的话，可以为消息设置一张表，发送消息时将消息写入表中，同时记录写入的时间，如果消息正常到 mq，就将表中的消息状态修改为 true；然后开启一个定时器去不断轮询表中超过了 1 分钟且状态仍是未消费的数据，不断去重试。重试达到 3 次后就停止，人工介入。但是这种会极大的拉低 mq 的 QPS。

### 消息持久化

生产者确认可以确保消息投递到 RabbitMQ 的队列中，但是消息发送到 RabbitMQ 以后，如果突然宕机，也可能导致消息丢失。要想确保消息在 RabbitMQ 中安全保存，必须开启消息持久化机制。

- 交换机持久化
- 队列持久化
- 消息持久化

<span style="color:orange">但是 SpringAMQP 的交换机、队列、消息默认都是持久化的，不用刻意配置。但是这里还是学习下如何配置持久化。</span>

#### 交换机持久化

RabbitMQ 中交换机<b>默认是非持久化的</b>，mq 重启后就丢失。SpringAMQP 中可以通过代码指定交换机持久化：

```java
@Bean
public DirectExchange simpleExchange(){
    // 三个参数：交换机名称、是否持久化、当没有queue与其绑定时是否自动删除
    return new DirectExchange("simple.direct", true, false);
}
```

事实上，默认情况下，由 SpringAMQP 声明的交换机都是持久化的。可以在 RabbitMQ 控制台看到持久化的交换机都会带上 `D` 的标示

<div align="center"><img src="assets/image-20210718164412450.png"></div>

#### 队列持久化

RabbitMQ 中队列默认是非持久化的，mq 重启后就丢失。SpringAMQP 中可以通过代码指定交换机持久化

```java
@Bean
public Queue simpleQueue(){
    // 使用QueueBuilder构建队列，durable就是持久化的
    return QueueBuilder.durable("simple.queue").build();
}
```

事实上，默认情况下，由 SpringAMQP 声明的队列都是持久化的。可以在 RabbitMQ 控制台看到持久化的队列都会带上 `D` 的标示

<div align="center"><img src="assets/image-20210718164729543.png"></div>

#### 消息持久化

利用 SpringAMQP 发送消息时，可以设置消息的属性（MessageProperties），指定 delivery-mode：

- 1：非持久化
- 2：持久化

用 Java 代码指定

```java
package com.platform.fight;

import com.rabbitmq.client.Channel;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.core.MessageBuilder;
import org.springframework.amqp.core.MessageDeliveryMode;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.amqp.rabbit.connection.CorrelationData;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.boot.test.context.SpringBootTest;

import javax.annotation.Resource;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

@SpringBootTest
@Slf4j
public class TestMQ {
    @Resource
    RabbitTemplate rabbitTemplate;

    @Test
    public void testDurableMessage() {
        // 创建消息
        Message build = MessageBuilder.withBody("simple queue".getBytes(StandardCharsets.UTF_8))
                .setDeliveryMode(MessageDeliveryMode.PERSISTENT)
                .build();
        // 消息ID，需要封装到 CorrelationData 中，接收方如何获取消息的 ID呢？
        CorrelationData correlationData = new CorrelationData(UUID.randomUUID().toString());
        rabbitTemplate.convertAndSend("simple.queue", build, correlationData);
        log.info("消息发送成功！");
    }
    
    
    @RabbitListener(queues = "simple.queue")
    public void testGetDurableMessage(Message message, Channel channel) {
        // debug 发现这样可以获取消息的 correlationId，
        String correlationId = message.getMessageProperties().getHeader("spring_listener_return_correlation").toString();
        System.out.println(correlationId);
    }

    @Test
    public void testRabbitListener() {
        testDurableMessage();
    }
}
```

默认情况下，SpringAMQP 发出的任何消息都是持久化的，也不用特意指定。

### 消费者消息确认

RabbitMQ 是通过消费者回执来确认消费者是否成功处理消息的：消费者获取消息后，应该向 RabbitMQ 发送 ACK 回执，表明自己已经处理消息。

ack 指 Acknowledge，确认。 表示消费端收到消息后的确认方式。而 SpringAMQP 则允许配置三种确认模式

- manual：手动 ack，需要在业务代码结束后，调用 api 发送 ack。
- auto：自动 ack，由 spring 监测 listener 代码是否出现异常，没有异常则返回 ack；抛出异常则返回 nack
- none：关闭 ack，MQ 假定消费者获取消息后会成功处理，因此消息投递后立即被删除

> 由此可知

- none 模式下，消息投递是不可靠的，可能丢失
- auto 模式类似事务机制，出现异常时返回 nack，消息回滚到 mq；没有异常，返回 ack
- manual：自己根据业务情况，判断什么时候该 ack

一般，我们都是使用默认的 auto 即可。如果设置了手动确认方式，则需要在业务处理成功后，调用 channel.basicAck()，手动签收，如果出现异常，则调用 channel.basicNack() 方法，让其自动重新发送消息，也可以调用 channel.basicReject() 方法拒接消息并且设置是否重新入队。

#### 演示none模式

修改 consumer 服务的 application.yml 文件，添加下面内容

```yaml
spring:
  rabbitmq:
    listener:
      simple:
        acknowledge-mode: none # 关闭ack
```

修改 consumer 服务的 SpringRabbitListener 类中的方法，模拟一个消息处理异常：

```java
@RabbitListener(queues = "simple.queue")
public void listenSimpleQueue(String msg) {
    log.info("消费者接收到simple.queue的消息：【{}】", msg);
    // 模拟异常
    System.out.println(1 / 0);
    log.debug("消息处理完成！");
}
```

测试可以发现，当消息处理抛异常时，消息依然被 RabbitMQ 删除了。

#### 演示auto模式

把确认机制修改为 auto

```yaml
spring:
  rabbitmq:
    listener:
      simple:
        acknowledge-mode: auto # 关闭ack
```

在异常位置打断点，再次发送消息，程序卡在断点时，可以发现此时消息状态为 unack（未确定状态）：

<div align="center"><img src="assets/image-20210718171705383.png"></div>

抛出异常后，因为 Spring 会自动返回 nack，所以消息恢复至 Ready 状态，并且没有被 RabbitMQ 删除

<div align="center"><img src="assets/image-20210718171759179.png"></div>

#### 演示manual模式

把确认机制修改为 manual

```yml
spring:
  rabbitmq:
    listener:
     # RabbitMQ模式使用simple  simple支持事务的
      simple:
        acknowledge-mode: manual # 设置为手动签收
    	prefetch: 1 # 限流,配置1 表示消费端每次向MQ拉取最大一条消息
```

在消费的创建监听类，并在方法上使用注解 `@RabbitListener(queues="队列名称")` 监听队列。

```java
import com.rabbitmq.client.Channel;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

import java.io.IOException;

/**
 * Consumer ACK机制：默认自动签收
 *  1. 设置手动签收。acknowledge="manual"
 *  2. 让监听器类实现ChannelAwareMessageListener接口
 *  3. 如果消息成功处理，则调用channel的 basicAck()签收
 *  4. 如果消息处理失败，则调用channel的basicNack()拒绝签收，broker重新发送给consumer
 */
@Component
public class AckListener {

    @RabbitListener(queues = "test_Queue_Confirm")
    public void testAck(Message message, Channel channel) throws IOException {
        //得到消息的唯一deliveryTag
        long deliveryTag = message.getMessageProperties().getDeliveryTag();

        //模拟接收到消息消费的逻辑
        try{
            //接收到消息进行消费
            System.out.println(new String(message.getBody()));
            System.out.println("消息到了ACK机制中~~~");

            // 模拟执行逻辑错误
			// int i = 1/0;

            // 手动签收消息
            /*
             * deliveryTag：表示收到的消息的参数标签(消息的唯一id)
             * 第二个参数：是否签收多条消息(批量签收消息)
             */
            channel.basicAck(deliveryTag,true);
        }catch (Exception e){
            //当我们上面的逻辑出现错误,就不会签收消息,我们在catch中就执行拒绝签收
            System.out.println("消费逻辑出现异常~~~消息被Ack机制重回队列");
            //拒绝签收
            /*
            第三个参数：requeue：重回队列。如果设置为true，则消息重新回到queue的尾部，broker会重新发送该消息给消费端,false为丢弃改消息,若设置了死信队列,就会交给死信队列
             */
            channel.basicNack(deliveryTag,true,false);
        }
    }
}
```

- 在 `rabbit:listener-container` 标签中设置 `acknowledge` 属性，设置 ack 方式 none：自动确认，manual：手动确认
- 如果在消费端没有出现异常，则调用 channel.basicAck(deliveryTag,false); 方法确认签收消息
- 如果出现异常，则在 catch 中调用 basicNack 或 basicReject，拒绝消息，让 MQ 重新发送消息。

<b>消息可靠性总结</b>

- 持久化
    - exchange 要持久化
    - queue 要持久化
    - message 要持久化
- 生产方确认 Confirm
- 消费方确认 Ack
- Broker 高可用

### 消费失败重试机制

当消费者出现异常后，消息会不断 requeue（重入队）到队列，再重新发送给消费者，然后再次异常，再次 requeue，无限循环，导致 mq 的消息处理飙升，带来不必要的压力：

<div align="center"><img src="assets/image-20210718172746378.png"></div>

怎么办呢？

#### 本地重试

可以利用 Spring 的 retry 机制，在消费者出现异常时利用本地重试，而不是无限制的 requeue 到 mq 队列。

修改 consumer 服务的 application.yml 文件，添加内容：

```yaml
spring:
  rabbitmq:
    listener:
      simple:
        retry:
          enabled: true # 开启消费者失败重试
          initial-interval: 1000 # 初识的失败等待时长为1秒
          multiplier: 1 # 失败的等待时长倍数，下次等待时长 = multiplier * last-interval
          max-attempts: 3 # 最大重试次数
          stateless: true # true无状态；false有状态。如果业务中包含事务，这里改为false
```

重启 consumer 服务，重复之前的测试可以发现：

- 在重试 3 次后，SpringAMQP 会抛出异常 AmqpRejectAndDontRequeueException，说明本地重试触发了
- 查看 RabbitMQ 控制台，发现消息被删除了，说明最后 SpringAMQP 返回的是 ack，mq 删除消息了

结论：

- 开启本地重试时，消息处理过程中抛出异常，不会 requeue 到队列，而是在消费者本地重试
- 重试达到最大次数后，Spring 会返回 ack，消息会被丢弃

#### 失败策略

在之前的测试中，达到最大重试次数后，消息会被丢弃，这是由 Spring 内部机制决定的。

在开启重试模式后，重试次数耗尽，如果消息依然失败，则需要有 MessageRecovery 接口来处理，它包含三种不同的实现：

- RejectAndDontRequeueRecoverer：重试耗尽后，直接 reject，丢弃消息。默认就是这种方式

- ImmediateRequeueMessageRecoverer：重试耗尽后，返回 nack，消息重新入队

- RepublishMessageRecoverer：重试耗尽后，将失败消息投递到指定的交换机

比较优雅的一种处理方案是 RepublishMessageRecoverer，失败后将消息投递到一个指定的，专门存放异常消息的队列，后续由人工集中处理。

1）在 consumer 服务中定义处理失败消息的交换机和队列

```java
@Bean
public DirectExchange errorMessageExchange(){
    return new DirectExchange("error.direct");
}
@Bean
public Queue errorQueue(){
    return new Queue("error.queue", true);
}
@Bean
public Binding errorBinding(Queue errorQueue, DirectExchange errorMessageExchange){
    return BindingBuilder.bind(errorQueue).to(errorMessageExchange).with("error");
}
```

2）定义一个RepublishMessageRecoverer，关联队列和交换机

```java
@Bean
public MessageRecoverer republishMessageRecoverer(RabbitTemplate rabbitTemplate){
    return new RepublishMessageRecoverer(rabbitTemplate, "error.direct", "error");
}
```

完整代码如下

```java
import org.springframework.amqp.core.Binding;
import org.springframework.amqp.core.BindingBuilder;
import org.springframework.amqp.core.DirectExchange;
import org.springframework.amqp.core.Queue;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.rabbit.retry.MessageRecoverer;
import org.springframework.amqp.rabbit.retry.RepublishMessageRecoverer;
import org.springframework.context.annotation.Bean;

@Configuration
public class ErrorMessageConfig {
    @Bean
    public DirectExchange errorMessageExchange(){
        return new DirectExchange("error.direct");
    }
    @Bean
    public Queue errorQueue(){
        return new Queue("error.queue", true);
    }
    @Bean
    public Binding errorBinding(Queue errorQueue, DirectExchange errorMessageExchange){
        return BindingBuilder.bind(errorQueue).to(errorMessageExchange).with("error");
    }

    @Bean
    public MessageRecoverer republishMessageRecoverer(RabbitTemplate rabbitTemplate){
        return new RepublishMessageRecoverer(rabbitTemplate, "error.direct", "error");
    }
}
```

### 消费端限流

<div align="center"><img src="assets/image-20221010154111483.png"></div>

进行限量的时候，确认 ack 的设置要设置成手动确认，配置限流的方式和<span style="color:red">“消息确认 ack” </span>一样。

```yml
listener:
  # RabbitMQ模式使用simple  simple支持事务的
  simple:
    # Consumer ACK机制：设置为手动签收
    acknowledge-mode: manual
    prefetch: 1 # 限流,配置1 表示消费端每次向MQ拉取最大一条消息
```

### 消息的幂等性

消息消费的幂等性可以结合 Redis 进行处理，Redis 中 ID 作为 key，消费前判断消息的消费次数是否存在于 Redis 中，存在则说明已经消费过了，不存在则消费并将消息写入 Redis 中。

消息投递的幂等性是否可以不考虑。一个消息如果多次投递，由于消息的 ID 是唯一的，因此重复投递的消息其 ID 也是唯一的。我们在消息消费的时候处理幂等即可。

以下代码为个人项目中处理消息幂等性的方式

消息发送方

```java
package com.platform.game.utils;

import lombok.extern.slf4j.Slf4j;
import org.jetbrains.annotations.NotNull;
import org.springframework.amqp.rabbit.connection.CorrelationData;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.stereotype.Component;

import javax.annotation.Resource;
import java.util.UUID;

@Slf4j
@Component
public class RabbitMQUtils {
    @Resource
    private RabbitTemplate rabbitTemplate;

    private static final String RECORD_QUEUE = "records.queue";
    private static final String EXCHANGE_NAME = "simples.direct";

    public void sendMsg2RecordQueue(String message) {
        CorrelationData correlationData = getCorrelationData();
        // 失败暂时只记录日志，后面再做调整。
        correlationData.getFuture().addCallback(
                result -> {
                    if (result.isAck()) {
                        log.info("record 消息发送成功, ID:{}", correlationData.getId());
                    } else {
                        log.error("record 消息发送失败, ID:{}, 原因{}", correlationData.getId(), result.getReason());
                    }
                },
                ex -> log.error("record 消息发送异常, ID:{}, 原因{}", correlationData.getId(), ex.getMessage())
        );
        rabbitTemplate.convertAndSend(EXCHANGE_NAME, "records", message, correlationData);
    }

    @NotNull
    private CorrelationData getCorrelationData() {
        // 设置消息的唯一 id
        return new CorrelationData(UUID.randomUUID().toString());
    }
}
```

消息接收方

```java
package com.platform.fight.mq;

import cn.hutool.json.JSONUtil;
import com.baomidou.mybatisplus.extension.toolkit.SqlRunner;
import com.platform.fight.mapper.RecordMapper;
import com.platform.fight.pojo.Record;
import com.platform.fight.pojo.User;
import com.platform.fight.pojo.UserDTO;
import com.platform.fight.utils.RedisKeyUtils;
import com.rabbitmq.client.Channel;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.core.ExchangeTypes;
import org.springframework.amqp.core.Message;
import org.springframework.amqp.rabbit.annotation.Exchange;
import org.springframework.amqp.rabbit.annotation.Queue;
import org.springframework.amqp.rabbit.annotation.QueueBinding;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.stereotype.Component;

import javax.annotation.Resource;
import java.util.concurrent.TimeUnit;

@Slf4j
@Component
public
class RabbitListenerMessage {

    @Autowired
    private RecordMapper recordMapper;
    @Resource
    StringRedisTemplate stringRedisTemplate;

    // 直接使用注解创建交换机和队列
    @RabbitListener(bindings = @QueueBinding(value = @Queue("records.queue"),
            exchange = @Exchange(value = "simples.direct", type = ExchangeTypes.DIRECT),
            key = {"records"}
    ))
    public void listenRecordQueueMessage(Message message, Channel channel) {
        long deliveryTag = message.getMessageProperties().getDeliveryTag();
        // 获取消息的唯一 id，为什么是这个方法呢？因为我尝试网上的 xxx.getCorrectionId 发现获取到的是 null，于是我看了下 mq 中的消息
        // 发现 spring_returned_message_correlation 就是 correlationId, debug 看了下 message 中的值
        // 发现 header map 中存储了 spring_returned_message_correlation
        String messageId = message.getMessageProperties().getHeader("spring_returned_message_correlation").toString();
        // 重复消费会被丢弃
        Boolean aBoolean = stringRedisTemplate.opsForValue().setIfAbsent(messageId, messageId, 60, TimeUnit.SECONDS);
        try {
            if (aBoolean == null) {
                channel.basicReject(deliveryTag, true);// requeue 重回队列尾部
            } else if (Boolean.TRUE.equals(aBoolean)) {
                // true 説明，不存在設置成功，可以正常消費
                String msg = new String(message.getBody());
                Record record = JSONUtil.toBean(msg, Record.class);
                recordMapper.insert(record);
                log.info("消费者接收到消息：【{}】", record);
                channel.basicAck(deliveryTag, true);
            } else {
                log.info("重复消费，丢弃消息 {}", new String(message.getBody()));
                channel.basicNack(deliveryTag, true, false);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

### 总结

如何确保 RabbitMQ 消息的可靠性？

- 开启生产者确认机制，确保生产者的消息能到达队列
- 开启持久化功能，确保消息未消费前在队列中不会丢失
- 开启消费者确认机制为 auto，由 spring 确认消息处理成功后完成 ack
- 开启消费者失败重试机制，并设置 MessageRecoverer，多次重试失败后将消息投递到异常交换机，交由人工处理

如何确保 RabbitMQ 消息的幂等性？

- 结合 Redis 判断是否重复消费
- 消息 ID 采用分布式自增 ID 或 UUID，消费后写入数据库，并且消息 ID 作为主键，采用数据库主键唯一的特点保证消息消费的幂等性

## 死信交换机

### 死信

什么是死信？当一个队列中的消息满足下列情况之一时，可以成为死信（dead letter）：

- 消费者使用 basic.reject 或 basic.nack 声明消费失败，并且消息的 requeue 参数设置为 false
- 消息是一个过期消息，超时无人消费
- 要投递的队列消息满了，无法投递

如果这个包含死信的队列配置了 `dead-letter-exchange` 属性，指定了一个交换机，那么队列中的死信就会投递到这个交换机中，而这个交换机称为<span style="color:red">死信交换机</span>（Dead Letter Exchange，检查 DLX）。

如图，一个消息被消费者拒绝了，变成了死信：

<div align="center"><img src="assets/image-20210718174328383.png"></div>

因为 simple.queue 绑定了死信交换机 dl.direct，因此死信会投递给这个交换机：

<div align="center"><img src="assets/image-20210718174416160.png"></div>

如果这个死信交换机也绑定了一个队列，则消息最终会进入这个存放死信的队列：

<div align="center"><img src="assets/image-20210718174506856.png"></div>


另外，队列将死信投递给死信交换机时，必须知道两个信息：

- 死信交换机名称
- 死信交换机与死信队列绑定的 RoutingKey

这样才能确保投递的消息能到达死信交换机，并且正确的路由到死信队列。

<div align="center"><img src="assets/image-20210821073801398.png"></div>

### 死信队列

其他 MQ 产品中没有交换机的概念，有的是死信队列的概念，因此这里用死信队列。

死信队列，英文缩写：DLX 。Dead Letter Exchange（死信交换机），当消息成为 Dead message 后，可以被重新发送到另一个交换机，这个交换机就是 DLX。DLX 也会绑定一个 queue，然后让其他消费者进行消费消息。

<div align="center"><img src="assets/image-20221011142144528.png"></div>

<b>消息成为死信的三种情况</b>

- 队列消息长度到达限制，即要投递的队列消息满了，无法投递；
- 消费者拒接消费消息，basicNack/basicReject, 并且不把消息重新放入原目标队列, requeue=false；
- 原队列存在消息过期设置，消息到达超时时间未被消费；

<b>队列绑定死信交换机</b>

- 给队列设置参数：x-dead-letter-exchange 和 x-dead-letter-routing-key

<b>代码逻辑</b>

死信队列：

- 声明正常的队列 (test_queue_dlx) 和正常交换机 (test_exchange_dlx)

- 声明死信队列 (queue_dlx) 和死信交换机 (exchange_dlx)

- 正常队列绑定死信交换机

- 设置两个参数：

    * x-dead-letter-exchange：死信交换机名称

    * x-dead-letter-routing-key：发送给死信交换机的 routingkey

```java
import org.springframework.amqp.core.*;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.HashMap;
import java.util.Map;

/*
死信队列：
	1. 声明正常的队列(test_queue_dlx)和正常交换机(test_exchange_dlx)
	2. 声明死信队列(queue_dlx)和死信交换机(exchange_dlx)
    3. 正常队列绑定死信交换机
    	设置两个参数：
        	* x-dead-letter-exchange：死信交换机名称
        	* x-dead-letter-routing-key：发送给死信交换机的routingkey
 */
@Configuration
public class RabbitMQDeadMessageConfig {

    //创建自定义 死信交换机  逻辑认为是用来做死信服务的
    @Bean
    public Exchange exchangeDlx(){
        return ExchangeBuilder.topicExchange("exchange_del").build();
    }

    //创建自定义 死信队列  逻辑认为是用来做死信服务的
    @Bean
    public Queue queueDlx(){
        return QueueBuilder.durable("queue_dlx").build();
    }

    //将自定义的死信队列绑定在一块
    @Bean
    public Binding bindingDlx(@Qualifier("exchangeDlx") Exchange exchange,@Qualifier("queueDlx") Queue queue){
        return BindingBuilder.bind(queue).to(exchange).with("dlx.*").noargs();
    }

    //创建正常接收消息的交换机
    @Bean
    public Exchange exchangeNormalDlx(){
        return ExchangeBuilder.topicExchange("exchange_Normal_DLX").build();
    }

    //创建正常接收消息的队列,绑定我们的死信交换机
    @Bean
    public Queue queueNormalDlx(){
        return QueueBuilder.durable("queue_Normal_DLX")//正常队列的名称
                .withArgument("x-dead-letter-exchange","exchange_del")//设置改队列的死信交换机
                .withArgument("x-dead-letter-routing-key","dlx.xf")//设置该队列的发送消息时指定的routingkey
                .withArgument("x-message-ttl",10000)//设置队列中消息的过期时间
                .withArgument("x-max-length",10).build();//设置队列的最大容量
    }

    //将正常的交换机与队列绑定
    @Bean
    public Binding bindingNormalDlx(@Qualifier("exchangeNormalDlx") Exchange exchange,@Qualifier("queueNormalDlx") Queue queue){
        return BindingBuilder.bind(queue).to(exchange).with("test.dlx.#").noargs();
    }
}
```

<b>小结</b>

什么样的消息会成为死信？

- 消息被消费者 reject 或者返回 nack
- 消息超时未消费
- 队列满了

死信交换机的使用场景是什么？

- 如果队列绑定了死信交换机，死信会投递到死信交换机；
- 可以利用死信交换机收集所有消费者处理失败的消息（死信），交由人工处理，进一步提高消息队列的可靠性。

### TTL

TTL 全称 Time To Live（存活时间/过期时间）。当消息到达存活时间后，还没有被消费，会被自动清除。RabbitMQ 可以对消息设置过期时间，也可以对整个队列（Queue）设置过期时间。

<div align="center"><img src="assets/image-20221011135850877.png"></div>

- 设置队列过期时间使用参数：x-message-ttl，单位：ms(毫秒)，这个队列过期时间怎么算的？是队列中每个消息单独计时，还是从第一个消息开始计时？写代码测测。
- 设置消息过期时间使用参数：expiration。单位：ms(毫秒)，<b>当该消息在队列头部时（消费时），会单独判断这一消息是否过期。</b>
- 如果两者都进行了设置，以时间短的为准。

<div align="center"><img src="assets/image-20210718182643311.png"></div>

<b>接收超时死信的死信交换机</b>

在 consumer 服务的 SpringRabbitListener 中，定义一个新的消费者，并且声明死信交换机、死信队列

```java
@RabbitListener(bindings = @QueueBinding(
    value = @Queue(name = "dl.ttl.queue", durable = "true"),
    exchange = @Exchange(name = "dl.ttl.direct"),
    key = "ttl"
))
public void listenDlQueue(String msg){
    log.info("接收到 dl.ttl.queue的延迟消息：{}", msg);
}
```

<b>声明一个队列，并指定 TTL</b>

要给队列设置超时时间，需要在声明队列时配置 x-message-ttl 属性

```java
@Bean
public Queue ttlQueue(){
    return QueueBuilder.durable("ttl.queue") // 指定队列名称，并持久化
        .ttl(10000) // 设置队列的超时时间，10秒
        .deadLetterExchange("dl.ttl.direct") // 指定死信交换机
        .build();
}
```

注意，这个队列设定了死信交换机为 `dl.ttl.direct`

声明交换机，将 ttl 与交换机绑定

```java
@Bean
public DirectExchange ttlExchange(){
    return new DirectExchange("ttl.direct");
}
@Bean
public Binding ttlBinding(){
    return BindingBuilder.bind(ttlQueue()).to(ttlExchange()).with("ttl");
}
```

发送消息，但是不要指定 TTL

```java
@Test
public void testTTLQueue() {
    // 创建消息
    String message = "hello, ttl queue";
    // 消息ID，需要封装到CorrelationData中
    CorrelationData correlationData = new CorrelationData(UUID.randomUUID().toString());
    // 发送消息
    rabbitTemplate.convertAndSend("ttl.direct", "ttl", message, correlationData);
    // 记录日志
    log.debug("发送消息成功");
}
```

发送消息的日志

<div align="center"><img src="assets//image-20210718191657478.png"></div>

查看下接收消息的日志

<div align="center"><img src="assets/image-20210718191738706.png"></div>

因为队列的 TTL 值是 10000ms，也就是 10 秒。可以看到消息发送与接收之间的时差刚好是 10 秒。

<b>发送消息时，设定 TTL</b>

在发送消息时，也可以指定 TTL

```java
@Test
public void testTTLMsg() {
    // 创建消息
    Message message = MessageBuilder
        .withBody("hello, ttl message".getBytes(StandardCharsets.UTF_8))
        .setExpiration("5000")
        .build();
    // 消息ID，需要封装到CorrelationData中
    CorrelationData correlationData = new CorrelationData(UUID.randomUUID().toString());
    // 发送消息
    rabbitTemplate.convertAndSend("ttl.direct", "ttl", message, correlationData);
    log.debug("发送消息成功");
}
```

查看发送消息日志

<div align="center"><img src="assets/image-20210718191939140.png"></div>

接收消息日志

<div align="center"><img src="assets/image-20210718192004662.png"></div>

这次，发送与接收的延迟只有 5 秒。说明当队列、消息都设置了 TTL 时，任意一个到期就会成为死信。

<b>总结</b>

消息超时的两种方式是？

- 给队列设置 ttl 属性，进入队列后超过ttl时间的消息变为死信
- 给消息设置 ttl 属性，队列接收到消息超过 ttl 时间后变为死信

如何实现发送一个消息 20 秒后消费者才收到消息？

- 给消息的目标队列指定死信交换机
- 将消费者监听的队列绑定到死信交换机
- 发送消息时给消息设置超时时间为 20 秒

### 延迟队列

延迟队列，即消息进入队列后不会立即被消费，只有到达指定时间后，才会被消费。

查看下面的需求：

- 下单后，30 分钟未支付，取消订单，回滚库存（把该用户抢的库存加回去）。
- 新用户注册成功 7 天后，发送短信问候。

实现方式：

- 定时器，实现不优雅，需要设置时间定期执行任务，存在一定的误差，时间短了会耗费系统资源，时间长了误差又大。
- 延迟队列，实现优雅，每条消息只需要执行一次任务，开销也很小。

如果采用定时器完成判断用户是否支付的操作，需要定期执行查询操作，查询用户是否支付了，开销大；如果采用延迟队列的方式，每条消息只需要查询一次即可。

<div align="center"><img src="assets/image-20221011144450653.png"></div>

较早版本的 RabbitMQ 没有提供延迟队列的功能。但是可以用：<b style="color:red">TTL+死信队列</b>组合实现延迟队列的效果。但是后面因为延迟队列的需求非常多，所以 RabbitMQ 的官方也推出了一个插件，原生支持延迟队列效果。

这个插件就是 DelayExchange 插件。参考 RabbitMQ 的插件列表页面：https://www.rabbitmq.com/community-plugins.html

<div align="center"><img src="assets/image-20210718192529342.png"></div>

使用方式可以参考官网地址：https://blog.rabbitmq.com/posts/2015/04/scheduling-messages-with-rabbitmq

#### DelayExchange原理

DelayExchange 需要将一个交换机声明为 delayed 类型。当我们发送消息到 delayExchange 时，流程如下：

- 接收消息
- 判断消息是否具备 x-delay 属性
- 如果有 x-delay 属性，说明是延迟消息，持久化到硬盘，读取 x-delay 值，作为延迟时间
- 返回 routing not found 结果给消息发送者
- x-delay 时间到期后，重新投递消息到指定队列

#### 使用DelayExchange

插件的使用也非常简单：声明一个交换机，交换机的类型可以是任意类型，只需要设定 delayed 属性为 true 即可，然后声明队列与其绑定即可。

#### 声明DelayExchange交换机

基于注解方式（推荐）

<div align="center"><img src="assets/image-20210718193747649.png"></div>

也可以基于 @Bean 的方式

<div align="center"><img src="assets/image-20210718193831076.png"></div>

#### 发送消息

发送消息时，一定要携带 x-delay 属性，指定延迟的时间：

<div align="center"><img src="assets/image-20210718193917009.png"></div>

#### 总结

延迟队列插件的使用步骤包括哪些？

- 声明一个交换机，添加 delayed 属性为 true
- 发送消息时，添加 x-delay 头，值为超时时间

## 惰性队列

### 消息堆积问题

当生产者发送消息的速度超过了消费者处理消息的速度，就会导致队列中的消息堆积，直到队列存储消息达到上限。之后发送的消息就会成为死信，可能会被丢弃，这就是消息堆积问题。

<div align="center"><img src="assets/image-20210718194040498.png"></div>

解决消息堆积有两种思路：

- 增加更多消费者，提高消费速度。也就是我们之前说的 work queue 模式
- 扩大队列容积，提高堆积上限

要提升队列容积，把消息保存在内存中显然是不行的。

### 惰性队列

从 RabbitMQ 的 3.6.0 版本开始，就增加了 Lazy Queues 的概念，也就是惰性队列。惰性队列的特征如下：

- 接收到消息后直接存入磁盘而非内存
- 消费者要消费消息时才会从磁盘中读取并加载到内存
- 支持数百万条的消息存储

#### 基于命令行设置lazy-queue

而要设置一个队列为惰性队列，只需要在声明队列时，指定 x-queue-mode 属性为 lazy 即可。可以通过命令行将一个运行中的队列修改为惰性队列

```sh
rabbitmqctl set_policy Lazy "^lazy-queue$" '{"queue-mode":"lazy"}' --apply-to queues  
```

- `rabbitmqctl`：RabbitMQ 的命令行工具
- `set_policy`：添加一个策略
- `Lazy`：策略名称，可以自定义
- `"^lazy-queue$"`：用正则表达式匹配队列的名字
- `'{"queue-mode":"lazy"}'`：设置队列模式为 lazy 模式
- `--apply-to queues  `：策略的作用对象，是所有的队列

#### 基于@Bean声明lazy-queue

```java
@Bean
public Queue lazyQueue(){
	return QueueBuilder
        	.durable("lazy.queue")
        	.lazy() // 开启 x-queue-mode 为 lazy
        	.build();
}
```

#### 基于@RabbitListener声明LazyQueue

```java
@RabbitListener(queuesToDeclare= @Queue(
		name = "lazy.queue",
    	durable = "true",
    	arguments = @Argument(name = "x-queue-mode", value = "lazy")
))
public void listenLazyQueue(String msg){
    log.info("接收到 lazy.queue 的消息：{}", msg);
}
```

#### 总结

消息堆积问题的解决方案？

- 队列上绑定多个消费者，提高消费速度
- 使用惰性队列，可以再 mq 中保存更多消息

惰性队列的优点有哪些？

- 基于磁盘存储，消息上限高
- 没有间歇性的 page-out，性能比较稳定

惰性队列的缺点有哪些？

- 基于磁盘存储，消息时效性会降低
- 性能受限于磁盘的 IO

## 日志与监控

RabbitMQ 默认日志存放路径： /var/log/rabbitmq/rabbit@xxx.log。日志包含了 RabbitMQ 的版本号、Erlang 的版本号、RabbitMQ 服务节点名称、cookie 的 hash 值、
RabbitMQ 配置文件地址、内存限制、磁盘限制、默认账户 guest 的创建以及权限配置等等。

<b>常用命令</b>

- 查看队列 # rabbitmqctl list_queues
- 查看 exchanges # rabbitmqctl list_exchanges
- 查看用户 # rabbitmqctl list_users
- 查看连接 # rabbitmqctl list_connections
- 查看消费者信息 # rabbitmqctl list_consumers
- 查看环境变量 # rabbitmqctl environment
- 查看未被确认的队列 # rabbitmqctl list_queues name messages_unacknowledged
- 查看单个队列的内存使用 # rabbitmqctl list_queues name memory
- 查看准备就绪的队列 # rabbitmqctl list_queues name messages_ready

## 消息追踪

消息追踪：用于跟踪记录消息的投递过程，协助开发和运维人员进行问题定位。

在使用任何消息中间件的过程中，难免会出现某条消息异常丢失的情况。对于 RabbitMQ 而言，可能是因为生产者或消费者与 RabbitMQ 断开了连接，而它们与 RabbitMQ 又采用了不同的确认机制；也有可能是因为交换器与队列之间不同的转发策略；甚至是交换器并没有与任何队列进行绑定，生产者又不感知或者没有采取相应的措施；另外 RabbitMQ 本身的集群策略也可能导致消息的丢失。这个时候就需要有一个较好的机制跟踪记录消息的投递过程，以此协助开发和运维人员进行问题的定位。

<b>在 RabbitMQ 中可以使用 Firehose 和 rabbitmq_tracing 插件功能来实现消息追踪。</b>

firehose 的机制是将生产者投递给 rabbitmq 的消息，rabbitmq 投递给消费者的消息按照指定的格式发送到默认的 exchange 上。这个默认的 exchange 的名称为 amq.rabbitmq.trace，它是一个 topic 类型的 exchange。发送到这个 exchange 上的消息的 routing key 为 publish.exchangename 和 deliver.queuename。其中 exchangename 和 queuename 为实际 exchange 和 queue 的名称，分别对应生产者投递到 exchange 的消息，和消费者从 queue 上获取的消息。

## 应用问题

- 消息可靠性保障：消息补偿机制
- 消息幂等性保障：乐观锁解决方案

### 消息补偿

<div align="center"><img src="assets/image-20221011163625846.png"></div>

### 消息幂等性

幂等性指一次和多次请求某一个资源，对于资源本身应该具有同样的结果。也就是说，其任意多次执行对资源本身所产生的影响均与一次执行的影响相同。<b>在 MQ 中指，消费多条相同的消息，得到与消费该消息一次相同的结果。</b>

<div align="center"><img src="assets/image-20221011165027828.png"></div>

用数据库的乐观锁来保证消息的幂等性。用 Redis 保证消息的幂等性。

### WebSocket集群

```mermaid
graph LR
用户1-->|与2通信|服务器1-->MQ
用户2-->|与1通信|服务器2-->MQ
```

用户 1 的 WebSocket 连接在服务器 1 上；用户 2 的 WebSocket 连接在服务器 2 上。用户 1 和 2 要进行通信的话，由于不是在一台机器上因此无法直接通信，需要借助三方中间件进行消息的传递。可以采用 mq 的发布订阅模式，所有持有 WebSocket 连接的服务器都订阅 MQ 的 Message 消息队列，用户先检测通信对象是否在本地，不在本地则将消息发送到 MQ 中，其他订阅了 MQ Message 消息队列的拿到消息，判断是否是发送给自己的消息。

如果消息的丢失是可以容忍的，无需持久化，那么也可以采用 Redis 的发布订阅模式。

创建的需要通信但是通信双方不在同一服务器上都可以采用 MQ、Redis 这种作为中间商进行通信。但是需要分析下不同的策略，带来的各方面压力如何？会增加网络通信的压力还是其他压力？

# MQ集群

## 集群分类

RabbitMQ 的是基于 Erlang 语言编写，而 Erlang 又是一个面向并发的语言，天然支持集群模式。RabbitMQ 的集群有两种模式

- 普通集群：是一种分布式集群，将队列分散到集群的各个节点，从而提高整个集群的并发能力。
- 镜像集群：是一种主从集群，普通集群的基础上，添加了主从备份功能，提高集群的数据可用性。

镜像集群虽然支持主从，但主从同步并不是强一致的，某些情况下可能有数据丢失的风险。因此在 RabbitMQ 的 3.8 版本以后，推出了新的功能：仲裁队列来代替镜像集群，底层采用 Raft 协议确保主从的数据一致性。

## 普通集群

### 集群结构和特征

普通集群，或者叫标准集群（classic cluster），具备下列特征

- 会在集群的各个节点间共享部分数据，包括：交换机、队列元信息。不包含队列中的消息。
- 当访问集群某节点时，如果队列不在该节点，会从数据所在节点传递到当前节点并返回
- 队列所在节点宕机，队列中的消息就会丢失

结构如图：

<div align="center"><img src="assets/image-20210718220843323.png"></div>

### 部署

参考课前资料：《RabbitMQ 部署指南.md》

## 镜像集群

### 集群结构和特征

镜像集群：本质是主从模式，具备下面的特征：

- 交换机、队列、队列中的消息会在各个 mq 的镜像节点之间同步备份。
- 创建队列的节点被称为该队列的主节点，备份到的其它节点叫做该队列的镜像节点。
- 一个队列的主节点可能是另一个队列的镜像节点
- 所有操作都是主节点完成，然后同步给镜像节点
- 主宕机后，镜像节点会替代成新的主

结构如图

<div align="center"><img src="assets/image-20210718221039542.png"></div>

### 部署

参考课前资料：《RabbitMQ 部署指南.md》

## 仲裁队列

### 集群特征

仲裁队列：仲裁队列是 3.8 版本以后才有的新功能，用来替代镜像队列，具备下列特征：

- 与镜像队列一样，都是主从模式，支持主从数据同步
- 使用非常简单，没有复杂的配置
- 主从同步基于 Raft 协议，强一致

### 部署

参考课前资料：《RabbitMQ 部署指南.md》

### Java代码创建仲裁队列

```java
@Bean
public Queue quorumQueue() {
    return QueueBuilder
        .durable("quorum.queue") // 持久化
        .quorum() // 仲裁队列
        .build();
}
```

### SpringAMQP连接MQ集群

注意，这里用 address 来代替 host、port 方式

```java
spring:
  rabbitmq:
    addresses: 192.168.150.105:8071, 192.168.150.105:8072, 192.168.150.105:8073
    username: itcast
    password: 123321
    virtual-host: /
```

