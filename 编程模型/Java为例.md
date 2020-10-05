# 引言

Java自发布以来一直都在不断的发展，有语法层面的扩展，类库方面的优化，编译器方面的性能优化，这些优化使得我们对某些功能可以有更多的实现 甚至是创造新的编程模型。

断断续续学习Java这么久了，自己也有一些感受。

Java的一些常用类的命名很好，基本上见名知意，源码上的注释也写的很不错，基本可以通过阅读注释知道该方法的使用方法，应用场景。

Java是纯面向对象的语言，这也注定了Java源码的阅读不会轻松，各种设计模式，类与类的各种组合方式，“眼花缭乱”，用或许很简单，但源码的阅读却是艰难的（会用只是起点，了解为什么设计，如何实现才是重点）。

看过一些`JVM`的书，对`JVM`也有所了解，了解的越多，愈发觉得`JVM`强大。`JVM`的编译优化做的很优秀，一些性能差的地方也随着版本的更替越来越快。Java从运行速度慢，到其效率可与`CPP`媲美（得益于`JIT`和程序的局部性原理）尤其是主流的商用虚拟机引入了`JIT`，热点代码直接翻译成机器码，无需中转。【`JIT`默认是启用的，`JVM` 读入.class文件解释后发给`JIT`编译器，然后它将字节码编译成本机机器代码】

Java生态圈优秀，有很多优秀的工具类，开发人员多，学习资料多，相应的成本也较低。

Java发展的越来越好，也越来越“臃肿”，相同的功能，众多的实现方式，虽然灵活，但是也变相加大了学习的难度和应用难度，有些甚至需要足够理解`JVM`才能知道如何抉择。最初的Java抛弃了`CPP`的很多繁琐的特征，但是随着不断的发展Java也变得越来越繁琐。

Java的`GC`是它的亮点也是痛点。优秀的`CPP`工程师需要懂得硬件知识，优秀的Java工程师需要懂得`JVM`。`JVM`->内存自动管理，调优->懂得`JVM`，懂得`JVM`->看`JVM`设计，`JVM`设计->看`HotSpot`源码。【优秀的Java工程师也需要熟悉`CPP`】

----

# 语法层面

## assert关键字 

个别`JDK`源码中有用到assert关键字，好像是Integer源码里。

## 可边长参数

如果我们需要一个数组，我们可以通过以下方式获得。

```java
String []str = new String[]{"hello","world","java"};
```

Java的可变长参数可以优化我们获得数组的方式

```java
public static <T> T[] of(T... value){
	return value;    
}

public static <T> t[] of(T one, T...other){}

String []str = of("1","2","#"); // 省去了new的过程。
```

新语法的引入，可以让我们拥有/开发出更多的实现方式。

## Diamond语法

`JDK 1.7`提供了一个`AutoCloseable`接口，自动关闭资源。配合`JDK1.7 try...with...resource`的新语法使用。

```java
public static String XX(String path) throws IOException {
		
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {
			return br.readLine();
		}
}
```

# 类库提升

- `Java5`：`JUC`，`Formatter`，Java管理扩展（`JMX`），`XML`处理（`DOM`，`SAX`，`XPath`，`XSTL`）
- `Java6`：`JDBC 4.0`（`JSR221`）、 `JAXB 2.0`（`JSR222`）、可插拔注解处理`API`（`JSR269`）、Common Annotations（`JSR` 250）、Java Compiler `API`（`JSR199`）、Scripting in `JVM（``JSR223`）
- `Java7`：NIO 2（`JSR203`）Fork/Join框架（`JSR166`）`invokedynamic`字节码（`JSR` 292）
- `Java8`：Stream `API` （`JSR335`）、`CompletableFuture`（`J.U.C.`）、Annotation on Java Types（`JSR` 308）、Date and Time `API`（`JSR` 301）、可重复Annotations（`JSR` 337）、 JavaScript运行时（`JSR` 223）

类库提升，有的是新增`API`，有的是对原有`API`的扩充或优化，以适应当下的潮流。

# Java编程模型

- 面向对象 `OOP`
- 面向切面 `AOP`
- 面向元信息编程 `MDOP`
- 面向函数编程 `FOP`
- 面向模块编程 `MOP`

# Java契约编程

# 设计模式

## 面向对象设计模式

构造模式

结构模式

行为模式

开发模式

## 面向切面设计模式

判断模式

拦截模式

## 面向元数据设计模式

泛型接口设计

注解驱动设计

## 面向函数设计模式

函数式接口设计

Fluent API设计

Reactive / Stream API 设计

# Java模式驱动

## 接口驱动

JavaSE（GoF 23模式）

JavaEE API（Servlet、JSF、EJB）

Spring Core API（interface 21）

## 配置驱动

Java System Properties

OS 环境变量

文件配置（xml properties yaml）

JavaEE 配置（JDNI Servlet EJB）

## 注解驱动

JavaSE（Java Beans、 JMX）

JavaEE（Servlet3.0 、 JAX-RS、  Bean Validation、 EJB 3.0+ ...）

Spring（@Component、@Service、@Respository...）

Spring Boot（@SpringBootApplication）

Spring Cloud（@SpringCloudApplication）

## 函数驱动

Java 8 Stream API

Java 9 Flow API

RxJava

Vert.x

Spring Boot WebFlux

Spring Cloud Gateway/Function

## 模块驱动

Java OSGI

Java 9 Module

Spring @Enable

Spring Boot AutoConfiguration

Spring Boot Actuator

