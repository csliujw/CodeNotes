# MyBatis源码 Qiuck Start

## MyBatis 执行体系

> JDBC执行过程

获得连接 \==> 预编译SQL \==> 设置参数 \==> 执行SQL

> MyBatis执行流程

- 动态代理 MapperProxy
- SQL 会话 SqlSession
- 执行器 Executor
- JDBC 处理器 StatementHandler

<img src="..\pics\src\MyBatis_quick_star.png">

## 会话部分源码

- 采用的门面模式，来提供 API 操作，不提供具体操作。
- 基本 API： 增、删、改、查
- 辅助 API：提交、关闭会话
- 具体的处理交给 Executor（执行器），执行 CRUD 方法时会交给 Executor 处理。

> PS 门面模式和模板方法

模板方法是父类定义好方法和执行流程。子类负责实现每个具体的方法。

门面模式是要求一个子系统的外部与其内部的通信必须通过一个统一的门面对象进行。门面模式提供一个高层次的接口，使得子系统更易于使用。【统一父子通信方式？】

### 执行器的实现

> 简单执行器 SimpleExecutor

