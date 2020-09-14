# Spring学习前言

应用为主，原理为辅。

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

### 类与类的关系

BeanFactory提供了配置框架和基本功能。

The root interface for accessing a Spring bean container.

他是访问Spring bean容器的顶级父接口。所有 [应该是] 获取Bean的实例对象都实现了这个接口。

<img style="margin:left" src="..\pics\Spring\BeanFactory的实现类.png"></img>

常用的几个类如下：

- ClassPathXmlApplicationContext：通过类路径下的xml获得对象

- FileSystemXmlApplicationContext ：通过磁盘下面的xml获得bean对象

- AnnotationConfigApplicationContext：通过注解获得bean对象

三种获取bean的方式

注：采用注解方式把对象注入容器中的话，需要在类上加上注解，在方法上加上注解。之后会详细说明，此处不做说明！

```java
public class IOC {
    @Test
    public void testIOC() {
        var context1 = new ClassPathXmlApplicationContext("bean.xml");
        var context2 = new FileSystemXmlApplicationContext("D:\\Code\\Spring-Study\\Spring01\\src\\main\\resources\\bean.xml");
        User user1 = context1.getBean("user", User.class);
        User user2 = context2.getBean("user", User.class);
        System.out.println(user1);
        System.out.println(user2);
        context1.close();
        context2.close();
    }

    @Test
    public void testIOCByAnnotationConfigApplicationContext() {
        var context = new AnnotationConfigApplicationContext(AnnotationConfig.class);
        var bean = context.getBean(User.class);
        System.out.println(bean);
        context.close();
    }
}

@Configuration // 标识这是一个配置类
public class AnnotationConfig {
    @Bean // 表示返回的对象会被加入到容器中去
    public User getUser() {
        return new User();
    }
}
```

## xml方式注入

### xml分离

应用场景：dao层的bean配置在一个xml，service层的bean配置在一个xml，对bena进行分离

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <import resource="son_one.xml"/>
    <import resource="son_two.xml"/>
</beans>
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean id="role" class="com.bbxx.pojo.Role"/>
</beans>
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean id="user" class="com.bbxx.pojo.User"/>
</beans>
```

```java
public class MultiXML {
    @Test
    public void testMulti() {
        // 可同时加载多个xml文件
        var context = new ClassPathXmlApplicationContext("multi_xml\\son_one.xml",
                                                         "multi_xml\\son_two.xml");
        User user = context.getBean("user", User.class);
        Role role = context.getBean("role", Role.class);
        System.out.println(user + ":" + role);
        context.close();
    }

    @Test
    public void testMultiXML() {
        // 也可在paent中引入其他xml，这样只需书写一个xml文件
        var context = new ClassPathXmlApplicationContext("multi_xml\\parent.xml");
        User user = context.getBean("user", User.class);
        Role role = context.getBean("role", Role.class);
        System.out.println(user + ":" + role);
        context.close();
    }
}
```

### bean配置中包含的属性

| Property                 | Explained in…                                                |
| :----------------------- | :----------------------------------------------------------- |
| Class                    | [Instantiating Beans](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-factory-class) |
| Name                     | [Naming Beans](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-beanname) |
| Scope                    | [Bean Scopes](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-factory-scopes) |
| Constructor arguments    | [Dependency Injection](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-factory-collaborators) |
| Properties               | [Dependency Injection](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-factory-collaborators) |
| Autowiring mode          | [Autowiring Collaborators](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-factory-autowire) |
| Lazy initialization mode | [Lazy-initialized Beans](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-factory-lazy-init) |
| Initialization method    | [Initialization Callbacks](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-factory-lifecycle-initializingbean) |
| Destruction method       | [Destruction Callbacks](https://docs.spring.io/spring/docs/current/spring-framework-reference/core.html#beans-factory-lifecycle-disposablebean) |

### 实例化bean

注：如果要把内部类注入容器，应该这样写class

OtherTing是SomeThing的静态内部类【官方文档翻译】

```xml
<bean id="xxx" class="com.example.SomeThing$OtherThing."></bean>
```

- 通过构造方法实例化
- 通过静态工厂方法实例化
- 通过工厂方法实例化

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <!--构造方法实例化-->
    <bean id="user" class="com.bbxx.pojo.User"/>
    <!--静态工厂方法实例化-->
    <bean id="user_static_factory" class="com.bbxx.bean_instantiation.StaticFactory" factory-method="getUser"/>
    <!--工厂方法实例化-->
    <bean id="factory" class="com.bbxx.bean_instantiation.Factory"/>
    <bean id="user_factory" factory-bean="factory" factory-method="getUser"/>

</beans>
```

```java
public class BeanInstantiation {

    @Test
    public void constructor() {
        var context = new ClassPathXmlApplicationContext("bean_instantiation\\bean_instantiation.xml");
        User user = context.getBean("user", User.class);
        System.out.println(user+": constructor");
    }

    @Test
    public void staticFactory() {
        var context = new ClassPathXmlApplicationContext("bean_instantiation\\bean_instantiation.xml");
        User user_static_factory = context.getBean("user_static_factory", User.class);
        System.out.println(user_static_factory+": static factory");
    }

    @Test
    public void factory() {
        var context = new ClassPathXmlApplicationContext("bean_instantiation\\bean_instantiation.xml");
        User user_factory = context.getBean("user_factory", User.class);
        System.out.println(user_factory+": factory");
    }
}
```

### 依赖注入

依赖注入有两种主要的方式：基于构造方法注入和基于set/get方法注入



## 注解方式注入

### 常用注解

> **Look up Method Injection**

查找方法注入是容器重写容器管理的Bean上的方法并返回容器中另一个命名Bean的查找结果的能力。

Spring容器通过字节码技术实现了方法注入。用的CGLIB。

要使这个动态子类工作，Spring bean容器子类的类不能是final，要覆盖的方法也不能是final。

类中含有一个抽象方法，这个抽象方法需要返回一个对象。我们通过lookup指定这个抽象方法去那个类找对象即可。【不做要求】



## 注解和xml混用注入

## Bean的作用域范围

| Scope       | Description                     |
| ----------- | ------------------------------- |
| singleton   | 单例模式                        |
| prototype   | 多例                            |
| request     | 单例且request作用域有效         |
| session     | 单例且session作用域有效         |
| application | 单例 且生命周期为ServletContext |
| websocket   | 单例 且生命周期为websocket      |

用scope。

xml是scope属性

注解也是`@scope属性`

`request session application websocket需要在web环境下才可使用。对于Servlet3.0 我们可以使用WebApplicationInitializer初始化`

## 自定义Bean的性质

- 生命周期的回调
- ApplicationContextAware and BeanNameAware
- 其他装配接口

### 生命周期的回调

- 实现InitializingBean接口 【耦合度高】
- 使用JSR-250的注解 `@PostConstruct and @PreDestroy`
- 不想使用JSR-250仍想解除耦合的话 可以用`@init-method and @destory-method`

Spring内部是通过使用BeanPostProcessor实现来处理它可以找到的任何回调接口，并调用适当的方法。如果要自定义的话，可以实现一个BeanPostProcessor

常用的几个注解

```java
@PostConstruct  对应xml中的 init-method
@PreDestory 对应xml中的 destoru-method
```

## 扫描classpath和管理组件

Classpath Scanning and Managed

这部分讲的是如何通过JavaConfig用Spring来配置对象。

常用注解：

- @Configuration  标明为配置类。
- @Bean  用在方法上，方法的返回值会注入到容器中。
- @Import  导入其他JavaConfig。
- @DependsOn 设置Bean的加载顺序。
- @Repository 用于标记数据库访问层。【已经被支持作为持久化层中自动异常转换的标记】
- @Component是通用化的注解，可用在众多类上??
- @Service 给Service类加注解的。也可以用@Component，但是Service更为清晰。
- @RestController 是@Controller和@ResponseBody的组合。





## 说明

Spring不建议你的普通类继承或实现Spring中的接口，因为这样会增加代码的耦合性。

注解还是xml？看个人喜好和项目需求，各有各的优点。但是一般推荐用JavaConfig这种配置，即偏向于注解。

Spring可组合常用注解得到一个全新功能的注解。

## 常用注解总结

> **@Require**

可用于方法上的注解，表示必须为参数注入对象/值，避免空指针异常。【Spring 5.1出现的】

> **@Autowired**

自动注入。 JSR330的@Inject注解与@Autowired具备相同的功能，基本一致【个别场景可能不一样】

可用在成员变量上，可用在set方法上

> **@Qualifier**

一般与@Autowired一起使用。在有多个候选对象的使用，可以用@Qualifier指定选择那个bean。

该注解也可用在方法的形式参数上。

```java
@Autowired
public void prepare(@Qualifier("main") MovieCatalog movieCatalog,
                    CustomerPreferenceDao customerPreferenceDao) {
    this.movieCatalog = movieCatalog;
    this.customerPreferenceDao = customerPreferenceDao;
}
```

> **@Primary**

@Autowired可能会有多个候选者，所以可以用@Primary指定首选对象，避免错误。

在JavaConfig注入是，我们可以为其加上@Primary表示它为首选对象。【@Bean默认是单例模式】

> **@Value**

为字段属性赋值。也可以用在方法的形参上。
