# 概述

Spring 注解版本

核心在IOC和AOP

应用为主，原理为辅。先学会用再说。

主要学IOC AOP这两块内容。

用的Java10，图方便，用了自动类型推断（var 关键字）。

<a href="https://docs.spring.io/spring-framework/docs/current/spring-framework-reference/core.html#beans-factory-extension">如何扩展Spring的功能</a>

POJO 可重用的Java组件？

# Spring(一) 概述

## Spring是什么

Spring 是分层的 Java SE/EE 应用 **full-stack** 轻量级开源框架，以 IoC（Inverse Of Control： 反转控制）和 AOP（Aspect Oriented Programming：面向切面编程）为内核，提供了展现层 Spring MVC 和持久层 Spring JDBC 以及业务层事务管理等众多的企业级应用技术，还能整合开源世界众多 著名的第三方框架和类库，逐渐成为使用最多的 Java EE 企业应用开源框架。

## Spring IOC概述

基本是把官网复述一遍。。

IOC，控制反转。把new对象的操作交给IOC容器来做。

### 需要的jar

- org.springframework.context

- org.springframework.beans

- 测试了一下，的确是只需要这两个jar包就可以了

- 官网原文

  ```text
  The org.springframework.beans and org.springframework.context packages are the basis for Spring Framework’s IoC container.
  ```

### 类与类的关系

BeanFactory提供了配置框架和基本功能。

The root interface for accessing a Spring bean container.

他是访问Spring bean容器的顶级父接口，可以管理任意类型的对象。所有 [应该是] 获取Bean的实例对象都实现了这个接口。

ApplicationCotext是BeanFactory的子接口，对比BeanFactory他增加了如下的功能

- 更简单的基于AOP的注入
- 消息资源处理，用于国际化
- 事件发布
- Application层有一些特殊的context 如WebApplicationContext可用于web服务

Spring关于选择xml还是JavaConfig的一段话

 XML-based metadata is not the only allowed form of configuration metadata. The Spring IoC container itself is totally decoupled from the format in which this configuration metadata is actually written. These days, many developers choose [Java-based configuration](https://docs.spring.io/spring-framework/docs/current/spring-framework-reference/core.html#beans-java) for their Spring applications.

基于xml的元数据并不是唯一允许的配置元数据形式。Spring IoC容器本身与实际编写配置元数据的格式完全解耦。现在，许多开发人员为他们的Spring应用程序选择基于java的配置。

Spring容器与配置信息完全解耦，很多人都在使用JavaConfig，暗示推荐你也用JavaConfig

<span style="color:red">基本上都是名字长的类是名字短的类的子类或子接口。</span>

<img style="margin:left" src="..\pics\Spring\BeanFactory的实现类.png"></img>

常用的几个类如下：

- ClassPathXmlApplicationContext：通过类路径下的xml获得对象

- FileSystemXmlApplicationContext ：通过磁盘下面的xml获得bean对象

- AnnotationConfigApplicationContext：通过注解获得bean对象


