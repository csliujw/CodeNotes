# 引言

Java自发布以来一直都在不断的发展，有语法层面的扩展，类库方面的优化，编译器方面的性能优化，这些优化让我们可以合理地运用这些优化创造新的编程模型。

# 语法层面

## assert关键字

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

- `Java5`：JUC，Formatter，Java管理扩展（JMX），XML处理（DOM，SAX，XPath，XSTL）
- `Java6`：JDBC 4.0（JSR221）、 JAXB 2.0（JSR222）、可插拔注解处理API（JSR269）、Common Annotations（JSR 250）、Java Compiler API（JSR199）、Scripting in JVM（JSR223）
- `Java7`：NIO 2（JSR203）Fork/Join框架（JSR166）invokedynamic字节码（JSR 292）
- `Java8`：Stream API （JSR335）、CompletableFuture（J.U.C.）、Annotation on Java Types（JSR 308）、Date and Time API（JSR 301）、可重复Annotations（JSR 337）、 JavaScript运行时（JSR 223） 

# Java编程模型

- 面向对象 OOP
- 面向切面 AOP
- 面向元信息编程 MDOP
- 面向函数编程 FOP
- 面向模块编程 MOP