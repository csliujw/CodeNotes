# 引言

Java自发布以来一直都在不断的发展，有语法层面的扩展，类库方面的优化，编译器方面的性能优化，这些优化使得我们对某些功能可以有更多的实现 甚至是创造新的编程模型。

Java是纯面向对象的语言，这也注定了Java源码的阅读不会轻松，各种设计模式，类与类的各种组合方式，“眼花缭乱”，用或许很简单，但源码的阅读却是艰难的（会用只是起点，了解为什么设计，如何实现才是重点）。

`JVM`的编译优化做的很优秀，一些性能差的地方也随着版本的更替越来越快。Java从运行速度慢，到其效率可与`CPP`媲美（得益于`JIT`和程序的局部性原理）尤其是主流的商用虚拟机引入了`JIT`，热点代码直接翻译成机器码，无需中转。【`JIT`默认是启用的，`JVM` 读入.class文件解释后发给`JIT`编译器，然后它将字节码编译成本机机器代码】

Java生态圈优秀，有很多优秀的工具类，开发人员多，学习资料多，相应的成本也较低。

Java发展的越来越好，也越来越“臃肿”，相同的功能，众多的实现方式，虽然灵活，但是也变相加大了学习的难度和应用难度，有些甚至需要足够理解`JVM`才能知道如何抉择。最初的Java抛弃了`CPP`的很多繁琐的特征，但是随着不断的发展Java也变得越来越繁琐。

Java的`GC`是它的亮点也是痛点。

----

# 语法层面

## assert关键字 

个别`JDK`源码中有用到assert关键字，好像是Integer源码里。

要想开启关键字assert生效，需要启用虚拟机参数 "-ea"

```java
public static void main(String[] args) {
    assert 1 == 1;
    System.out.println("A Go!");
    System.out.println("\n----------------------------\n");
    assert 1 != 1 : "Wrong";
    System.out.println("B Go!");
}
```

## 可变长参数

如果我们需要一个数组，我们可以通过以下方式获得。

```java
String []str = new String[]{ "hello", "world", "java"};
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

## 稍微补一下泛型

<span style="color:green">**带尖括号的才是泛型！！**</span>

- `class Demo<T>{}`泛型类
- `public static<T> T getXXX(T...a)`泛型方法

**<span style="color:green">泛型命名规则</span>**

- `E：Element` (used extensively by the Java Collections Framework)
- `K：Key`
- `N：Number`
- `T：Type`
- `V：Value`
- `S，U，V：etc. -2nd, 3rd, 4th types`
- `R：Result`
- `A：Accumulator`

```java
package org.example.video.generic;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * 泛型的高级用法
 * > 限制泛型可用类型和使用类型通配符
 * class ClassName<T extends anyClass>
 * > 使用类型通配符
 * genericName< ? extends List > a = null
 */
public class Senior {
    /**
     * 伪泛型 会报错
     * public void say(List<String> list1) {
     * }
     * public void say(List<Integer> list1) {
     * }
     */

    public static void main(String[] args) {
        Senior senior = new Senior();
        senior.test4();
    }


    // 限制泛型的使用类型
    public void test1() {
        LimitClass<ArrayList> c1 = new LimitClass<>();
        LimitClass<LinkedList> c2 = new LimitClass<>();
        // 报错，因为HashMap并不 extends List
        // LimitClass<HashMap> c3 = new LimitClass<>();
    }

    // 基本的泛型使用后 用反射越过泛型检查
    public void test2() {
        ArrayList<String> strings = new ArrayList<>();
        List o = (List) Proxy.newProxyInstance(
            ArrayList.class.getClassLoader(), 
            strings.getClass().getInterfaces(), 
        	new InvocationHandlerDemo<ArrayList<String>>(strings));
        
        o.add(123);
        // 越过了泛型检查
        System.out.println(strings.size());
    }

    /**
     * =================================
     * ==============内部类==============
     * =================================
     */
    static class LimitClass<T extends List> {

    }

    static class InvocationHandlerDemo<T> implements InvocationHandler {
        private T target = null;

        public InvocationHandlerDemo() {
        }

        public InvocationHandlerDemo(T target) {
            this.target = target;
        }

        @Override
        public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
            System.out.println("add elements");
            Object o = method.invoke(target, args);
            return o;
        }
    }
}
```

## Class类与反射

<span style="color:green">**稍微补一补反射的知识**</span>

反射这块的知识，我确实不是很熟悉~~

### 反射可访问的主要描述信息

| 组成部分       | 访问方法            | 返回值类型  | 说明                                              |
| -------------- | ------------------- | ----------- | ------------------------------------------------- |
| 包路径         | getPackage()        | Package对象 | 获得该类的存放路径                                |
| 类名称         | getName()           | String对象  | 获得该类的名称                                    |
| 继承类         | getSuperclass()     | Class对象   | 获得该类继承的类                                  |
| 实现接口       | getInterfaces()     | Class型数组 | 获得该类实现的所有接口                            |
| 构造方法       |                     |             |                                                   |
| 方法           |                     |             |                                                   |
| 成员变量       |                     |             |                                                   |
| 内部类         | getClasses()        | Class型数组 | 获得所有权限为public的内部类                      |
| 内部类的声明类 | getDeclaringClass() | Class对象   | 如果该类为内部类，则返回它的成员类，否则返回null. |

<span style="color:green">**单纯的看反射API是没什么意思的，要用反射的API写些工具类才有意思。**</span>

## 注解

### 注解的定义&注意点

```java
public @interface AnnotationDemo{
    
}
```

也就几个注意点：

- ElementType的值，每个值的含义看注释即可
- RetentionPolicy中的值（指定作用范围）
  - `SOURCE` 不编译Annotation到类文件中，有效范围最小，仅一个查看作用？？
  - `CLASS` 表示编译Annotation到类文件中，但是在运行时不加载Annotation到JVM中
  - `RUNTIME` <span style="color:green">**表示在运行时加载Annotation到JVM中，有效范围最大，一般就是用这个**</span>

### 注解与反射

> <span style="color:green">**类Constructor、Field、Method均继承了AccessibleObject类，有些是间接继承的~~**</span>

```java
class Field extends AccessibleObject implements Member{
    // .....
}
```

**`AccessibleObject`中包含这三个方法：**

普通开发人员，用这几个方法用的多一些~~

```java
// 是否添加了指定类型的注解
public boolean isAnnotationPresent(Class<? extends Annotation> annotationClass) {
    return AnnotatedElement.super.isAnnotationPresent(annotationClass);
}
// 获得指定类型的注解
public <T extends Annotation> T getAnnotation(Class<T> annotationClass) {
    throw new AssertionError("All subclasses should override this method");
}
// 获得指定类型的所有注解
public Annotation[] getAnnotations() {
    return getDeclaredAnnotations();
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

