# 概述

[(1条消息) Spring源码_从头再来_f的博客-CSDN博客](https://blog.csdn.net/weixin_42128429/category_11339692.html)

## 学习内容

> 容器

- AnnotationConfigApplicationContext
- 组件添加
- 组件赋值
- 组件注入
- AOP
- 声明式事务

> 扩展原理

- BeanFactoryPostProcessor
- BeanDefinitionRegistryPostProcessor
- ApplicationListener
- Spring 容器创建过程

> web

- Servlet 3.0 请求
- 异步请求

<a href="https://docs.spring.io/spring-framework/docs/current/spring-framework-reference/core.html#beans-factory-extension">如何扩展 Spring 的功能</a>

> 配置文件注意点

- 配置文件需要放在源码文件夹，这样合并的时候才会出现在 bin 目录下

- 目录层级关系

   src|

  ​      |com 类所在的包名

  conf 配置文件所在的文件夹，与 src 目录同级别

## maven报错

这个报错了，怎么办？

```xml
<plugin>
    <artifactId>maven-project-info-reports-plugin</artifactId>
    <version>3.0.0</version>
</plugin>
```

引入这个依赖就行！

```xml
<dependency>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-project-info-reports-plugin</artifactId>
    <version>3.0.0</version>
    <type>maven-plugin</type>
</dependency>
```

## Spring的优点

1️⃣非入侵式框架。可以使应用程序代码对框架的依赖最小化（其实也小不到哪里去）

2️⃣方便解耦，简化开发。Spring就是一个大工厂，可以将所有对象的创建和依赖关系的维护工作都交给Spring容器管理，大大地降低了组件之间的耦合性。

3️⃣支持 AOP。Spring 提供了对 AOP 的支持，它允许将一些通用任务，如安全、事务、日志等进行集中式处理，从而提高了程序的复用性。

4️⃣支持声明式事务处理。只需要通过配置就可以完成对事务的管理，而无须手动编程。

5️⃣方便集成各种优秀框架。Spring 内部提供了对各种优秀框架（如Struts、Hibernate、MyBatis、Quartz 等）的直接支持。

6️⃣降低 Java EE API 的使用难度。Spring 对 Java EE 开发中非常难用的一些 API（如 JDBC、JavaMail 等），都提供了封装，使这些 API 应用难度大大降低。

7️⃣提供了对 JUnit 的支持，方便程序测试。

# IOC

## 核心容器

Spring 框架的主要功能是通过其核心容器来实现的，而 Spring 框架提供了两种核心容器，分别为 BeanFactory 和 ApplicationContext。

### BeanFactory

BeanFactory 是基础类型的 IoC 容器。简单说，BeanFactory 就是一个管理 Bean 的工厂，它主要负责初始化各种 Bean，并调用它们的生命周期方法。

### ApplicationContext

ApplicationContext 是 BeanFactory 的子接口，也被称为应用上下文，是另一种常用的 Spring 核心容器。ApplicationContext 里不仅包含了 BeanFactory 的所有功能，还对 BeanFactory 做了功能增强，添加了对国际化、资源访问、事件传播等方面的支持。

### 依赖注入示例

依赖注入（Dependency Injection，简称 DI）与控制反转（IoC）的含义相同，只不过这两个称呼是从两个角度描述的同一个概念。

<b>控制反转：</b>不再是自己实例化对象，而是交给 Spring 容器来创建对象，控制权发生了反转。

<b>依赖注入：</b>A 类和 B 类，如果 A 要用到 B，就是 A 依赖了 B。Spring 的 IOC 容器会为 A 初始化这个 B 对象，即注入这个依赖。

> Spring 常用的依赖注入的方式有如下三种

1️⃣setter 方法注入，需要在 setter 方法上加上 @Atuowired 注解。

2️⃣构造方法注入：<a href="https://blog.csdn.net/weixin_42128429/article/details/121395148">一篇优质博客</a>

- 如果只有一个构造方法，不管是有参还是无参，都可以正常初始化，正常完成 bean 的注入，且不用加 @Autowired 注解。
- 如果有多个构造方法，没有给任何构造方法加 @Autowired 注解的话，默认会使用无参构造方法进行初始化。
- 如果有多个构造方法，但是没有无参数的，那么会报错。报错了，怎么办呢？为某个构造方法加上 @Autowired，就会使用那个构造方法进行初始化。
- 如果需要根据不同的情况来实例化对象怎么办？请看下面的多构造实例化代码

3️⃣属性注入，就是在属性上加上注解 @Autowired，大多数时候依赖注入都要结合注解注入来使用的。

> 依赖注入示例

所以依赖的 jar 包，导入 spring-context 包后，其他一些包也会自动导入哦~ 即核心容器所依赖的所有环境也会被导入。

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.review.spring</groupId>
    <artifactId>review-spring</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-context</artifactId>
            <version>5.3.3</version>
        </dependency>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-test</artifactId>
            <version>5.3.3</version>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13.2</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <version>1.18.24</version>
        </dependency>
    </dependencies>

</project>
```

相关的 POJO 类 -- 都位于 com.review.spring.pojo 包下

```java
@Component
@Data
public class User {
    private UserOne one;
    private UserTwo two;

//    public User() {
//        System.out.println("User ~");
//    }

    //    @Autowired
    public User(UserTwo two) {
        this.two = two;
        System.out.println("User have param two");
    }

//    public User(UserOne one) {
//        this.one = one;
//        System.out.println("User have param one");
//    }

    @Autowired
    public void setOne(UserOne one) {
        this.one = one;
    }
}

@Component
public class UserOne {}

@Component
public class UserTwo {}
```

Spring 配置扫描包的主体类

```java
@ComponentScan(basePackages = "com.review.spring")
public class Main {}
```

测试代码

```java
import com.review.spring.pojo.User;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = Main.class)
public class TestSetterAutowired {
    @Autowired
    User user;

    @Test
    public void testSetter() {
        System.out.println(user);
    }
}
```

可以用上述代码依次测试 setter 注入，构造方法注入；可以发现 setter 注入需要配合注解使用不然无法注入；而构造方法注入可以不加注解，如果有多个构造方法，且没有无参的构造，需要通过 @Autowired 标识使用那个构造方法创建对象。

不想使用单元测试的写法，可以采用下面的方式

```java
import com.review.spring.pojo.User;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.ComponentScan;

@ComponentScan(basePackages = "com.review.spring")
public class Main {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(Main.class);
        User bean = context.getBean(User.class);
        System.out.println(bean);
    }
}
```

如果 Bean 不是单实例的，而是多实例的，那么可以通过在 getBean 中指定构造函数来实例化，不过多实例的 Bean 不由 Spring 进行管理。

```java
@Component
@Data
@Scope(scopeName = "prototype")
// 修改 User 为多实例
public class User {
    private UserOne one;
    private UserTwo two;

    public User() {
        System.out.println("User ~");
    }

    //    @Autowired
    public User(UserTwo two) {
        this.two = two;
        System.out.println("User have param two");
    }

    public User(UserOne one) {
        this.one = one;
        System.out.println("User have param one");
    }

    //    @Autowired
    public void setOne(UserOne one) {
        this.one = one;
    }
}

@ComponentScan(basePackages = "com.review.spring")
public class Main {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(Main.class);
        User bean = context.getBean(User.class, new UserOne());
        System.out.println(bean);
    }
}
```

### Bean

[(1条消息) Spring源码学习（十）--推断构造方法_从头再来_f的博客-CSDN博客_spring推断构造方法](https://blog.csdn.net/weixin_42128429/article/details/121395148)

#### 实例化方式

在 Spring 中，要想使用容器中的 Bean，需要实例化 Bean。实例化 Bean 有三种方式，分别为<span style="color:orange">构造器实例化、静态工厂方式实例化和实例工厂方式实例化</span>

##### 构造器实例化

默认是使用无参构造方法。如果既有无参又有有参，默认使用无参。可以通过 @Autowire 指定使用某个构造方法。

```java
@Component
@Data
public class User {
    private UserOne one;
    private UserTwo two;

    public User() {
        System.out.println("User ~");
    }

	@Autowired
    public User(UserOne one) {
        this.one = one;
        System.out.println("User have param one");
    }
}


@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = Main.class)
public class TestSetterAutowired {
    @Autowired
    User user;

    @Test
    public void f1() {
        Assert.assertNotNull(user);
        System.out.println(user.getOne());
    }
}
```

##### 静态工厂方法实例化

静态工厂是实例化 Bean 的另一种方式。该方式要求开发者创建一个静态工厂的方法来创建 Bean 的实例。

```java
@Configuration
public class StaticFactoryMethod {

    @Bean("staticFactoryUser")
    public static User createUser() {
        return new User();
    }
}
```

##### 实例工厂实例化

```java
@Configuration
public class InstanceFactory {

    @Bean("instanceFactoryUser")
    public User createBean() {
        return new User();
    }
}
```

测试代码

```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = SpringConfig.class)
public class Hello {

    @Autowired
    @Qualifier("staticFactoryUser")
    User staticUser;

    @Autowired
    @Qualifier("instanceFactoryUser")
    User instanceUser;

    @Test
    public void f1() {
        Assert.assertNotNull(staticUser);
        Assert.assertNotNull(instanceUser);
    }
}
```

#### 作用域

通过 Spring 容器创建一个 Bean 的实例时，不仅可以完成 Bean 的实例化，还可以为 Bean 指定特定的作用域。Spring 中为 Bean 的实例定义了 7 种作用域。

| 作用域名称          | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| singleton（单实例） | 使用 singleton 定义的 Bean 在 Spring 容器中将只有一个实例，即单例模型。 |
| prototype（多实例） | 每次通过 Spring 容器获取的 prototype 定义的 Bean 时，容器都将创建一个新的 Bean 实例。 |
| request             | 在一次 HTTP 请求中，容器会返回一个 Bean 实例，不同的 HTTP 请求会产生不同的 Bean，且仅在当前 HTTP request 内有效。 |
| session             | 在一次 HTTP Session 中，容器会返回同一个 Bean 实例，且仅在当前 HTTP Session 内有效。 |
| globalSession       | 在一次 HTTP Session 中，容器会返回一个 Bean 实例，仅在使用 portlet 上下文时有效。 |
| application         | 为每个 ServletContext 对象创建一个实例。仅在 Web 相关的 ApplicationContext 中生效。 |
| websocket           | 为每个 websocket 对象创建一个实例。仅在 Web 相关的 ApplicationContext 中生效。 |

#### 生命周期

<span style="color:red">Spring 会管理 singleton 作用域的生命周期，不会管理 prototype 作用域的 Bean。在 singleton 作用域下，Spring 能够精确地知道该 Bean 何时被创建，何时初始化完成以及何时被销毁。</span>

<div align="center"><b>Spring 流程图</b></div>
<div align="center">
    <img src="img/spring/quick_start.png">
</div>
#### 装配方式

Spring 提供了基于 XML 的配置、基于注解的配置和自动装配等。主要讲解基于注解的配置。

Spring 中定义了一系列的注解，常用的注解如下：

- @Component：可以使用此注解描述 Spring 中的 Bean，但它是一个泛化的概念，仅仅表示一个组件（Bean），并且可以作用在任何层次。使用时只需将该注解标注在相应类上即可。
- @Repository：用于将数据访问层（DAO 层）的类标识为 Spring 中的 Bean，其功能与 @Component 相同。
- @Service：通常作用在业务层（Service 层），用于将业务层的类标识为 Spring 中的 Bean，其功能与 @Component 相同。
- @Controller：通常作用在控制层（如 Spring MVC 的 Controller），用于将控制层的类标识为 Spring 中的 Bean，其功能与 @Component 相同。
- @Autowired：用于对 Bean 的属性变量、属性的 setter 方法及构造方法进行标注，配合对应的注解处理器完成 Bean 的自动配置工作。<span style="color:orange">默认按照 Bean 的类型进行装配。如果按类型匹配发现有多个，就以字段名为 name 进行匹配，如果还没有匹配的 Bean，会报错。</span>
- @Resource：其作用与 Autowired 一样。其区别在于 @Autowired 默认按照 Bean 类型装配，而 @Resource 默认按照 Bean 实例名称进行装配。@Resource 中有两个重要属性：name 和 type。
    - Spring 将 name 属性解析为 Bean 实例名称，type 属性解析为 Bean 实例类型。
    - 如果指定 name 属性，则按实例名称进行装配；如果指定 type 属性，则按 Bean 类型进行装配；
    - 如果都不指定，则先按 Bean 实例名称装配，如果不能匹配，再按照 Bean 类型进行装配；如果都无法匹配，则抛出 NoSuchBeanDefinitionException 异常。
- @Qualifier：与 @Autowired 注解配合使用，会将默认的按 Bean 类型装配修改为按 Bean 的实例名称装配，Bean 的实例名称由 @Qualifier 注解的参数指定。

## 组件注解

### 组件注入

#### xml写法

POJO 对象

```java
package org.example.pojo;

public class Person {
    private String name;
    private Integer age;

    public Person() {}

    public Person(String name) {
        this.name = name;
    }
	// 省略 setter getter
}
```

获取 bean

```java
package org.example;

import org.example.pojo.Person;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class BeanXMLTest {
    public static void main(String[] args) {
        ClassPathXmlApplicationContext context = new ClassPathXmlApplicationContext("bean.xml");
        Person person = (Person) context.getBean("person");
        Person person2 = (Person)context.getBean("person2");
        System.out.println(person2);
        context.close();
    }
}
```

- `xml` 配置文件。在 maven 的 `resrouce` 目录下。resource 目录下的资源最后会变成项目根目录下的文件。所以是直接 `Classxxx("bean.xml")`
- `JavaSE` 的项目和 `JavaEE` 项目最后的输出路径好像都是 classes，但是 `JavaEE` 里写的路径是 `classpath`！

#### 注解配置类写法

@Configuration 可以替代 XML，进行类的配置。典型的应用有三方 jar 包，我们需要把它交给 Spring 容器进行管理，于是用 @Configuration 的方式把这个类注入到 Spring 中。

JavaConfig 配置类

```java
package org.example.configuration;

import org.example.pojo.Person;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MainConfiguration {
    // 给容器中注册一个Bean  默认是以方法名为 bean 的名称
    // 如果不想要方法名可以这样 @Bean("person") 或 @Bean({"person1","person2"})
    // 具体看看源码注释 一目了然。
    // value 与 name 之间 是别名关系
    @Bean("person3")
    public Person person() {
        return new Person();
    }
}
```

测试代码

```java
public class BeanXMLTest {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context2 = new AnnotationConfigApplicationContext(MainConfiguration.class);
        Person person21 = (Person) context2.getBean( "person3");
        System.out.println(person21);
    }
}

// =============================================================================
public class BeanXMLTest {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context2 = new AnnotationConfigApplicationContext(MainConfiguration.class);
        String[] beanNamesForType = context2.getBeanNamesForType(Person.class);
        String[] beanDefinitionNames = context2.getBeanDefinitionNames();
        for (int i = 0; i < beanDefinitionNames.length; i++) {
            System.out.println(beanDefinitionNames[i]);
        }
        System.out.println("===========");
        for (int i = 0; i < beanNamesForType.length; i++) {
            // 同一个类的多个别名只会输出其中一个。
            System.out.println(beanNamesForType[i]);
        }
    }
}
```

### 包扫描

用到的注解有 @Configuration、@ComponentScan，如果是 JDK8，它被设置成了重复注解，可以重复用。

#### xml 的方式

```xml
<!-- 配置包扫描 , 只要标注了@Controller、@Service、@Repository、@Component的都会被自动的扫描加入容器中-->
<context:component-scan base-package="org.example" />
```

#### 注解方式

注解方式，根据定类型进行排除

```java
// excludeFilters指定排除那些  用@Filter指定排除那些
// includeFilters指定包含那些  用@Filter指定包含那些
// 要让includeFilters生效需要设置@ComponentScan的useDefaultFilters=false，默认过滤器会导入所有的。
// MainConfiguration的配置对象不会被排除的
@Configuration
@ComponentScan(basePackages = "org.example", excludeFilters = {
        @ComponentScan.Filter(type = FilterType.ANNOTATION, 
                              classes = {Controller.class, Service.class})
})
public class MainConfiguration {
    // 给容器中注册一个Bean
    @Bean(name = {"person1", "person2", "person3"})
    public Person person() {
        return new Person();
    }

    @Bean
    public Person person007() {
        return new Person();
    }
}
```

注解方式，按指定规则包含

```java
// IncludeConfiguration的配置对象是也会包含的。
@Configuration
@ComponentScan(basePackages = "org.example", includeFilters = {
        @ComponentScan.Filter(type = FilterType.ASSIGNABLE_TYPE, 
                              classes = DemoService.class)
}, useDefaultFilters = false)
public class IncludeConfiguration {
    // 给容器中注册一个Bean
    @Bean(name = {"person1", "person2", "person3"})
    public Person person() {
        return new Person();
    }

    @Bean
    public Person person007() {
        return new Person();
    }
}
```

```java
public class ScanTest {

    @Test
    public void test1() {
        AnnotationConfigApplicationContext anno = new AnnotationConfigApplicationContext(MainConfiguration.class);
        String[] beanDefinitionNames = anno.getBeanDefinitionNames();
        for (int i = 0; i < beanDefinitionNames.length; i++) {
            System.out.println(beanDefinitionNames[i]);
        }
    }

    @Test
    public void test2() {
        AnnotationConfigApplicationContext anno = new AnnotationConfigApplicationContext(IncludeConfiguration.class);
        String[] beanDefinitionNames = anno.getBeanDefinitionNames();
        for (int i = 0; i < beanDefinitionNames.length; i++) {
            System.out.println(beanDefinitionNames[i]);
        }
    }
}
```

#### @Filter自定义过滤规则

自定义过滤规则的代码

```java
package org.example.configuration;

import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.FilterType;
import org.springframework.core.io.Resource;
import org.springframework.core.type.AnnotationMetadata;
import org.springframework.core.type.ClassMetadata;
import org.springframework.core.type.classreading.MetadataReader;
import org.springframework.core.type.classreading.MetadataReaderFactory;
import org.springframework.core.type.filter.TypeFilter;

import java.io.IOException;

@Configuration
@ComponentScan(basePackages = "org.example", includeFilters = {
        @ComponentScan.Filter(type = FilterType.CUSTOM, classes = {DefineFilter.class})
}, useDefaultFilters = false)
public class DefineFilterConfiguration {}

class DefineFilter implements TypeFilter {
    // 自定义匹配规则
    @Override
    public boolean match(MetadataReader metadataReader, MetadataReaderFactory metadataReaderFactory) throws IOException {
        AnnotationMetadata annotationMetadata = metadataReader.getAnnotationMetadata();
        // 获得当前正在扫描的类信息
        ClassMetadata classMetadata = metadataReader.getClassMetadata();
        // 获得当前类资源（类路径）
        Resource resource = metadataReader.getResource();
        // 类名
        String className = classMetadata.getClassName();
        System.out.println("---->" + className);
        if (className.contains("Dao")) {
            return true;
        }
        return false;
    }
}
```

测试代码

```java
package org.example;

import org.example.configuration.DefineFilterConfiguration;
import org.example.configuration.IncludeConfiguration;
import org.example.configuration.MainConfiguration;
import org.junit.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

public class ScanTest {

    @Test
    public void test3() {
        AnnotationConfigApplicationContext anno = new AnnotationConfigApplicationContext(DefineFilterConfiguration.class);
        String[] beanDefinitionNames = anno.getBeanDefinitionNames();
        for (int i = 0; i < beanDefinitionNames.length; i++) {
            System.out.println(beanDefinitionNames[i]);
        }
    }
}
```

### 单元测试

引入依赖，需要的 Junit 的版本有点高，如果单元测试出错，可能是 Junit 包的版本出错了。

[Spring测试官方文档](https://docs.spring.io/spring-framework/docs/current/reference/html/testing.html)

```xml
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-context</artifactId>
    <version>5.3.3</version>
</dependency>
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-test</artifactId>
    <version>5.3.3</version>
</dependency>
<dependency>
    <groupId>junit</groupId>
    <artifactId>junit</artifactId>
    <version>4.13.2</version>
    <scope>test</scope>
</dependency>
```

编写测试代码 (注入的测试类就不贴代码了)

```java
@ContextConfiguration(classes = ScopeConfiguration.class)
@RunWith(SpringJUnit4ClassRunner.class)
public class ScopeConfigurationTest {

    @Autowired
    Person person1;

    @Autowired
    Person person2;

    @Autowired
    Person person3;

    @Test
    public void test1() {
        System.out.println(person1 == person2);
        System.out.println(person1 == person3);
    }
}
```

### Bean作用域范围

通过 Spring 容器创建一个 Bean 的实例时，不仅可以完成 Bean 的实例化，还可以为 Bean 指定特定的作用域。而常见的作用域范围有下面五种。

- singleton 单例
- prototype 多例
- request 
- session
- global-session

完整的 Spring 中 Bean 的 7 种作用域。

| 作用域名称          | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| singleton（单实例） | 使用 singleton 定义的 Bean 在 Spring 容器中将只有一个实例，即单例模型。 |
| prototype（多实例） | 每次通过 Spring 容器获取的 prototype 定义的 Bean 时，容器都将创建一个新的 Bean 实例。 |
| request             | 在一次 HTTP 请求中，容器会返回一个 Bean 实例，不同的 HTTP 请求会产生不同的 Bean，且仅在当前 HTTP request 内有效。 |
| session             | 在一次 HTTP Session 中，容器会返回同一个 Bean 实例，且仅在当前 HTTP Session 内有效。 |
| globalSession       | 在一次 HTTP Session 中，容器会返回一个 Bean 实例，仅在使用 portlet 上下文时有效。<br>global session 类似于 HTTP Session 作用域，它只有对 portlet 才有意义。对于 Servlet 的 web 应用就相当于 session。 |
| application         | 为每个 ServletContext 对象创建一个实例。仅在 Web 相关的 ApplicationContext 中生效。 |
| websocket           | 为每个 websocket 对象创建一个实例。仅在 Web 相关的 ApplicationContext 中生效。 |

### 懒加载

- @Lazy，针对单实例容器启动时不创建对象，第一次获取 bean 时再进行初始化。
- 验证代码如下

```java
@Configuration
public class LazyConfiguration {
    @Scope("prototype")
    @Bean
    @Lazy public Person person() {
        System.out.println("Create Person");
        return new Person();
    }

    @Bean
    public Person getP() {
        System.out.println("Create getP");
        return new Person();
    }
}
```

### @Conditional条件注入

符合条件的 Bean 才会被注册到 IoC 容器中。

> @Conditional

```java
@Target({ElementType.TYPE, ElementType.METHOD}) // 方法
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface Conditional {

	/**
	 * All {@link Condition} classes that must {@linkplain Condition#matches match}
	 * in order for the component to be registered.
	 */
	Class<? extends Condition>[] value();
}

// 再看Class<? exntends Condition>[] 中的Condition
@FunctionalInterface
public interface Condition {
	/**
	 * Determine if the condition matches.
	 * @param context the condition context
	 * @param metadata the metadata of the {@link org.springframework.core.type.AnnotationMetadata class}
	 * or {@link org.springframework.core.type.MethodMetadata method} being checked
	 * @return {@code true} if the condition matches and the component can be registered,
	 * or {@code false} to veto the annotated component's registration
	 */
	boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata);

}
// 由此可看出，Conditional传入的是Condition数组
```

> 按条件进行注入

```java
package cn.study.ioc;

import cn.study.ioc.pojo.Person;
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.context.annotation.*;
import org.springframework.core.type.AnnotatedTypeMetadata;

import java.util.Arrays;

@Configuration
public class BeanConditionInject {
    @Bean("linux")
    @Conditional(value = {LinuxCondition.class})
    public Person getLinux() {
        return new Person("linux");
    }

    @Bean("windows")
    @Conditional(value = {WindowsCondition.class})
    public Person getWindows() {
        return new Person("windows");
    }

    // 包含指定的Bean才注入此obj对象
    @Bean("obj")
    @Conditional(value = {OtherCondition.class})
    public Object getObj() {
        return new Object();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanConditionInject.class);
        Arrays.stream(context.getBeanNamesForType(Person.class)).forEach(System.out::println);
        System.out.println("=========================");
        Arrays.stream(context.getBeanDefinitionNames()).forEach(System.out::println);

    }
}

class LinuxCondition implements Condition {

    /**
     * @param context  判断能使用的上下文环境
     * @param metadata 当前标注了Condtion注解的标注信息
     */
    @Override
    public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
        String property = context.getEnvironment().getProperty("os.name");
        if (property != null && property.contains("linux"))
            return true;
        return false;
    }
}

class WindowsCondition implements Condition {

    @Override
    public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
        String property = context.getEnvironment().getProperty("os.name");
        if (property != null && property.contains("Window"))
            return true;
        return false;
    }
}

// 包含 名为 windows 的 bean才 注入
class OtherCondition implements Condition {

    @Override
    public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
        context.getBeanFactory();
        context.getClassLoader();
        context.getEnvironment();
        BeanDefinitionRegistry registry = context.getRegistry();
        boolean windows = registry.containsBeanDefinition("windows");
        if (windows)
            return true;
        return false;
    }
}
```

### @Import导入其他组件

> 容器注入组件

- 包扫描 + 组件标注注解（@Controller / @Service / @Repository / @Component），局限于我们自己写的类
- @Bean [导入的第三方包里面的组件]，xml 的 bean 配置方式也可以做到。 
- @Import [快速给容器中导入一个组件]，xml 也有对应的引入方式。
  - @ImportSelector [导入的选择器，返回需要导入的组件的全类名数组]
  - @ImportBeanDefinitionRegistrar [也是一个接口]
- 使用 Spring 提供的 FactoryBean
  - 默认获取到的是工厂 bean 调用 getObject 创建的对象
  - 要获取工厂 Bean 本身，我们需要给 id 前面加一个& 如：&ColorFactoryBean
  - 这个的特点或者是优势到底是什么？为什么会提供这种方法？

import 注解的具体定义及注释

```java
public @interface Import {

	/**
	 * {@link Configuration @Configuration}, {@link ImportSelector},
	 * {@link ImportBeanDefinitionRegistrar}, or regular component classes to import.
	 */
	Class<?>[] value();
}
```

#### Import的基本用法

将没有使用 @Component 注解的普通 class 加入到 Spring 容器, 由 Spring 管理。

```java
@Configuration
@Import(Color.class) // 导入 Color
public class BeanImport {
    public static void main(String[] args) {
        ApplicationContext context = new AnnotationConfigApplicationContext(BeanImport.class);
        Arrays.stream(context.getBeanDefinitionNames()).forEach(System.out::println);
    }
}

class Color {
    @Bean
    public Person getColor() {
        return new Person("color");
    }
}
/*
...
beanImport
cn.study.ioc.Color
getColor
*/
```

#### Import的高级用法

> <b>高级用法一</b>

ImportSelector，最重要的是 selectImports 方法。

selectImports 方法的返回值是一个字符串数组，如果在配置类中，通过 @Import 注解，导入了该类，那么 selectImports 返回的字符串数组中的类名就会被 Spring 容器 new 出来，然后再把这些对象放到工厂当中去。

```java
public interface ImportSelector {

	// 选择并返回需要导入的类的名称
	String[] selectImports(AnnotationMetadata importingClassMetadata);

	// 返回排除的类，是一个类过滤器
	@Nullable
	default Predicate<String> getExclusionFilter() {
		return null;
	}
}
```

```java
package cn.study.ioc;

import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.context.annotation.ImportSelector;
import org.springframework.core.type.AnnotationMetadata;

import java.util.Arrays;
import java.util.Set;

/**
 * 测试 ImportSelect 接口的功能
 */
@Configuration
@Import(MyImportSelector.class)
public class BeanImportSelector {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanImportSelector.class);
        Arrays.stream(context.getBeanDefinitionNames()).forEach(System.out::println);
    }
}

class MyImportSelector implements ImportSelector {

    /***
     * @param importingClassMetadata 当前标注@Import注解的类的所有注解信息，可以用它获取到Import注解和其他注解的信息
     * @return 要导入到组件的全类名
     */
    @Override
    public String[] selectImports(AnnotationMetadata importingClassMetadata) {
        Set<String> annotationTypes = importingClassMetadata.getAnnotationTypes();
        annotationTypes.stream().forEach(System.out::println);
        System.out.println("==============");
        return new String[]{Person.class.getName(), B.class.getName(), C.class.getName()};
    }
}
```

> <b>高级用法二</b>

ImportBeanDefinitionRegistrar 接口，这个接口的功能比 ImportSelector 接口要更为强大，可以拿到所有 bean 的定义信息（BeanDefinitionRegistry）。

```java
public interface ImportBeanDefinitionRegistrar {

	default void registerBeanDefinitions(AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry,
			BeanNameGenerator importBeanNameGenerator) {
		registerBeanDefinitions(importingClassMetadata, registry);
	}

    /**
    通过调这个方法，给容器自己添加一些组件
    AnnotationMetadata 是当前类的注解信息
    BeanDefinitionRegistry Bean定义的注册类，通过它给容器注册Bean
    */
	default void registerBeanDefinitions(AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry) {
	}
}
```

我们可以通过 ImportBeanDefinitionRegistart 接口实现这个功能：如果存在 xxbean，就把 oobean 注册进去。

```java
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.beans.factory.support.RootBeanDefinition;
import org.springframework.context.annotation.*;
import org.springframework.core.type.AnnotationMetadata;

@Configuration
@Import({ImportBeanDefinitionDemo.class})
public class BeanImportBeanDefinitionRegistrar {
    @Bean("red")
    public Red red() {
        return new Red();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanImportBeanDefinitionRegistrar.class);
        System.out.println(context.containsBean("red")); // true
        System.out.println(context.containsBean("rain")); // true
    }
}

class ImportBeanDefinitionDemo implements ImportBeanDefinitionRegistrar {
    /***
     * @param importingClassMetadata
     * @param registry 拿到所有 bean 的注册信息。
     * 如果存在名为 red 的 bean 定义信息，就把 rain 也注册进去。和 Condition 的功能类似
     */
    @Override
    public void registerBeanDefinitions(AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry) {
        boolean red = registry.containsBeanDefinition("red");
        if (red) {
            RootBeanDefinition rootBeanDefinition = new RootBeanDefinition(Rain.class);
            registry.registerBeanDefinition("rain", rootBeanDefinition);
        }
    }
}

class Red {}
class Rain {}
```

### FactoryBean创建

使用 Spring 提供的 FactoryBean

- 默认获取到的是工厂 bean 调用 getObject 创建的对象
- 要获取工厂 Bean 本身，我们需要给 id 前面加一个& 如：&ColorFactoryBean
- 这个的特点或者是优势到底是什么？为什么会提供这种方法？

代码

```java
import org.springframework.beans.factory.FactoryBean;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

public class BeanFactoryBean implements FactoryBean<Person> {
    @Override
    public Person getObject() throws Exception {
        return new Person();
    }

    @Override
    public Class<?> getObjectType() {
        return Person.class;
    }

    /**
     * @return true 单实例 容器中只保存一份
     * false 多实例 每次调用创建新对象
     */
    @Override
    public boolean isSingleton() {
        return false;
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanFactoryBean.class);
        Person p1 = (Person) context.getBean("beanFactoryBean");
        Person p2 = (Person) context.getBean("beanFactoryBean");

        System.out.println(p1.getClass()); // xx.Person
        System.out.println(p1 == p2); // false

        // 加上&符号 获取的是工厂对象 而非getObject返回的Bean
        BeanFactoryBean bean2 = (BeanFactoryBean) context.getBean("&beanFactoryBean");
        // 也可以直接按类型查找 Bean
        BeanFactoryBean bean1 = context.getBean(BeanFactoryBean.class);

        System.out.println(bean1 == bean2); // true
    }
}
```

解释，请看源码

```java
public interface BeanFactory {

	/**
	 * Used to dereference a {@link FactoryBean} instance and distinguish it from
	 * beans <i>created</i> by the FactoryBean. For example, if the bean named
	 * {@code myJndiObject} is a FactoryBean, getting {@code &myJndiObject}
	 * will return the factory, not the instance returned by the factory.
	 */
	String FACTORY_BEAN_PREFIX = "&";
}
```

## 生命周期

Bean 的生命周期

```mermaid
graph LR
bean创建===>初始化===>销毁过程
```

容器管理 bean 的生命周期，我们可以自定义初始化和销毁方法；容器在 bean 进行到当前生命周期的时候来调用我们自己定义的初始化和销毁方法。

### Bean的初始化/销毁

#### 指定初始化和销毁方法

单实例的 bean 在容器启动的时候创建对象，多实例的 bean 在每次获取的时候创建对象。对象创建完成，并赋值好（包括依赖注入吗？），调用初始化方法。单实例和多实例都是一样的。

单实例的 bean 在容器关闭的时候调用 destoryMethod，<b>而多实例的 bean 不会调用 destoryMethod。</b>

- @Bean(initMethod = "init", destroyMethod = "destroy")

- 原本在 xml 中的配置方式

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd http://www.springframework.org/schema/context https://www.springframework.org/schema/context/spring-context.xsd">
    <bean id="demo" class="org.example.pojo.Person" init-method="toString" destroy-method="toString"></bean>
    <!-- 配置包扫描 -->
    <context:component-scan base-package="org.example"></context:component-scan>
</beans>
```

使用 JavaConfig

```java
@Configuration
public class BeanLifeCycle {
    // 可以在自定义数据源，用init和destroy进行数据源的初始化和关闭
    // @Scope("prototype")
    @Bean(initMethod = "init", destroyMethod = "destroy")
    public Car car() {
        return new Car();
    }

    public static void main(String[] args) throws IOException {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanLifeCycle.class);
        Object car = context.getBean("car");
        context.close();
        // 多实例的bean在获取时才创建对象
    }
}

class Car {
    public Car() {
        System.out.println("Car constructor...");
    }

    public void init() {
        System.out.println("car ... init");
    }

    public void destroy() {
        System.out.println("car ... destroy");
    }
}
/*
Car constructor...
car ... init
car ... destroy
*/
```

#### 实现接口自定义初始化和销毁

通过实现接口，自定义初始化和销毁的逻辑

```java
@Configuration
public class BeanLifeCycle {
    // 可以在自定义数据源，用init和destroy进行数据源的初始化和关闭
    // @Scope("prototype")
    @Bean
    public Car2 car2() {
        return new Car2();
    }

    public static void main(String[] args) throws IOException {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanLifeCycle.class);
        Object car = context.getBean("car2");
        context.close();
        // 多实例的bean在获取时才创建对象
    }
}

class Car2 implements InitializingBean, DisposableBean {
    public Car2() {
        System.out.println("Car2 constructor...");
    }

    public void destroy() {
        System.out.println("car2 ... destroy");
    }

    // 属性设置好后，调用 init-method
    public void afterPropertiesSet() throws Exception {
        System.out.println("car2 ... init");
    }
}
```

#### JS250注解

- @PostConstruct，在 bean 创建完成并属性赋值完成，来执行初始化方法
- @PreDestroy，在容器销毁 bean 之前通知我们进行清理操作
- 这几个注解是 Java 提供的，需要提供 J2EE 的依赖。

```xml
<dependency>
    <groupId>jakarta.annotation</groupId>
    <artifactId>jakarta.annotation-api</artifactId>
    <version>1.3.5</version>
    <scope>compile</scope>
</dependency>
```

```java
@Configuration
public class BeanLifeCycle {
    @Bean
    public Car3 car3() {
        return new Car3();
    }

    public static void main(String[] args) throws IOException {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanLifeCycle.class);
        Object car = context.getBean("car3");
        context.close();
    }
}

class Car3 {
    @PostConstruct
    public void init() {
        System.out.println("car3 ... init");
    }

    @PreDestroy
    public void destroy() {
        System.out.println("car3 ... destroy");
    }
}
```

### Bean后置处理器

参照官方文档

- interface BeanPostProcessor：bean 的后置处理器；在 bean 初始化（执行 init 方法）前后进行一些处理工作
  - postProcessBeforeInitialization：初始化之前工作
  - postProcessAfterInitialization：初始化之后工作

可以通过自定义 Bean 后置处理器，对特定的 Bean 进行后处理。

```mermaid
graph LR
bean创建===>|执行 bean processor|初始化===>|执行 bean processor|销毁过程
```

bean 的创建应该包括了完成依赖注入？

#### 基本用法

定义后置处理器，为 name 是 getUser1 的 name 字段赋值。

```java
@Data
public class User {
    private String name;
    private UserOne one;
    private UserTwo two;

    public User() {
        System.out.println("User ~");
    }

    public User(UserTwo two) {
        this.two = two;
        System.out.println("User have param two");
    }

    public User(UserOne one) {
        this.one = one;
        System.out.println("User have param one");
    }

    public void setOne(UserOne one) {
        this.one = one;
    }
}

@Configuration
public class BeanConfig {
    @Bean
    public User getUser1() {
        System.out.println("create getUser1");
        return new User();
    }

    @Bean
    public User getUser2() {
        System.out.println("create getUser2");
        return new User();
    }
}
```

自定义的后置处理器

```java
@Component
public class TestBeanPostProcessor implements BeanPostProcessor {

    public TestBeanPostProcessor() {
        System.out.println("construct TestBeanPostProcessor");
    }

    @Nullable
    public Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException {
        if (beanName.equals("getUser1") && bean.getClass() == User.class) {
            User user = (User) bean;
            user.setName("hello");
            System.out.println("postProcessBeforeInitialization == " + bean + " == " + beanName);
            return user;
        }
        return bean;
    }

    @Nullable
    public Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException {
        if (beanName.equals("getUser1")) {
            System.out.println("postProcessBeforeInitialization == " + bean + " == " + beanName);
        }
        return bean;
    }
}
```

测试代码

```java
@ComponentScan(basePackages = "com.review.spring")
public class Main {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(Main.class);
    }
}
/*
construct TestBeanPostProcessor
create getUser1
User ~
postProcessBeforeInitialization == User(name=hello, one=null, two=null) == getUser1
postProcessBeforeInitialization == User(name=hello, one=null, two=null) == getUser1
create getUser2
User ~
car3 ... init
*/
```

用 @Bean 注册 BeanPostProcessor 也可以。

```java
@Configuration
public class BeanConfig {
    @Bean
    public User getUser1() {
        System.out.println("create getUser1");
        return new User();
    }

    @Bean
    public User getUser2() {
        System.out.println("create getUser2");
        return new User();
    }

    @Bean
    public TestBeanPostProcessor testBeanPostProcessor() {
        System.out.println("bean 中创建 TestBeanPostProcessor");
        return new TestBeanPostProcessor();
    }
}
```

#### 基本原理

在 postProcessBeforeInitialization 方法上打一个断点，然后看方法调用栈，看看都调用了那些方法。然后看看那个方法和 init-method，BeanPostProcessor 有关。

<div align="center"><img src="img/image-20230108005420505.png"></div>

点击调用栈 initializeBean 可以发现，该方法是用来执行 bean 的 init methods 的。查阅 initializeBean 的代码

```java
protected Object initializeBean(String beanName, Object bean, @Nullable RootBeanDefinition mbd) {
    // some code ...
    if (mbd == null || !mbd.isSynthetic()) {
        // 该方法是用来执行 BeanPostProcessorBeforeInitialization 方法的
        wrappedBean = applyBeanPostProcessorsBeforeInitialization(wrappedBean, beanName);
    }

    try {
        invokeInitMethods(beanName, wrappedBean, mbd);
    }
    // some code ...
}
```

继续查阅 applyBeanPostProcessorsBeforeInitialization 的代码发现，该方法循环遍历所有 BeanPostProcessor 挨个执行 postProcessBeforeInitialization，一旦返回 null，跳出 for 循环，不会执行后面的 postProcessBeforeInitialization。

```java
	public Object applyBeanPostProcessorsBeforeInitialization(Object existingBean, String beanName)
			throws BeansException {

		Object result = existingBean;
		for (BeanPostProcessor processor : getBeanPostProcessors()) {
			Object current = processor.postProcessBeforeInitialization(result, beanName);
			if (current == null) {
				return result;
			}
			result = current;
		}
		return result;
	}
```

 BeanPostProcessor 的大致执行流程

```java
populateBean(beanName, mbd, instanceWrapper); 给 bean 进行属性赋值
initializeBean{
    applyBeanPostProcessorsBeforeInitialization //for 循环得到全部 beanPost
        invokeInitMethods(beanName, wrappedBean, mbd); //初始化方法
    applyBeanPostProcessorsAfterInitialization //for 循环得到全部 beanPost
}
```

#### Spring中的使用

Spring 中大量使用到了 BeanPostProcessor。如 Bean 的赋值，注入其他组件，@Autowired，生命周期注解功能，@Async，xxxBeanPostProcessor。

<div align="center"><img src="img/image-20230108010331463.png"></div>

> Spring 通过 BeanPostProcessor 注入其他组件

ApplicationContextAwareProcessor 可以帮我们在组件中注入 IoC 容器，如果想要在组件中使用 IoC 容器，实现 ApplicationContextAware 接口即可。

```java
// ApplicationContextAwareProcessor 中的方法，为 ApplicationContextAware 的子类注入 IoC 容器。
public Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException {
    if (!(bean instanceof EnvironmentAware || bean instanceof EmbeddedValueResolverAware ||
          bean instanceof ResourceLoaderAware || bean instanceof ApplicationEventPublisherAware ||
          bean instanceof MessageSourceAware || bean instanceof ApplicationContextAware ||
          bean instanceof ApplicationStartupAware)) {
        return bean;
    }

    AccessControlContext acc = null;

    if (System.getSecurityManager() != null) {
        acc = this.applicationContext.getBeanFactory().getAccessControlContext();
    }

    if (acc != null) {
        AccessController.doPrivileged((PrivilegedAction<Object>) () -> {
            invokeAwareInterfaces(bean);
            return null;
        }, acc);
    }
    else {
        invokeAwareInterfaces(bean);
    }

    return bean;
}

private void invokeAwareInterfaces(Object bean) {
	// some code ...
    // 注入 IoC 容器
    if (bean instanceof ApplicationContextAware) {
        ((ApplicationContextAware) bean).setApplicationContext(this.applicationContext);
    }
}
```

如何调试看调用流程呢？和前面一样，在 setApplicationContext 方法打上断点，看方法调用栈

```java
@Component
public class UserOne implements ApplicationContextAware {
    private ApplicationContext context;

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        this.context = applicationContext;
    }
}
```

<div align="center"><img src="img/image-20230108011402182.png"></div>

#### 实用功能

可以利用 Bean 后置处理的特点来自定义 Spring 注解。在 Bean 后置处理器中做相应的注解处理。此处通过自定义的注解来实现用户访问次数的统计。

> 自定义注解

```java
@Target({ElementType.ANNOTATION_TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface Count {
    int value() default 0;
}
```

> 声明一个类，在该类的方法上加上注解

方法增强用的 JDK 动态代理，因此需要一个接口。

```java
public interface TestCount {
    void say(int a);
}

@Service
public class TestCountImpl implements TestCount {
    @Count
    public void say(int a) {}
}
```

> 处理自定义注解逻辑的 BeanPostProcessor

需要注意的是，有时候 BeanPostProcessor 拿到的是经过动态代理增强后的 Bean，此时无法获得原始 Bean 上的注解，如何获取到原始的 Bean 呢？学完 AOP 就知道了。

```java
@Configuration
// 获取所有的方法，对方法上有 @Count 注解的进行方法增强，执行前后统计执行次数。
public class CountBeanPostProcessor implements BeanPostProcessor {
    private Map<String, Integer> countMap = new HashMap<>();

    @Override
    public Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException {
        Class<?> beanClass = bean.getClass();
        Method[] declaredMethods = beanClass.getDeclaredMethods();
        boolean returnBean = true;
        for (Method declaredMethod : declaredMethods) {
            if (declaredMethod.isAnnotationPresent(Count.class)) {
                returnBean = false;
                break;
            }
        }
        if (returnBean) {
            return bean;
        }

        // 需要增强才进行增强。
        return generateProxyBean(bean, beanClass);
    }

    private Object generateProxyBean(Object bean, Class<?> beanClass) {        
        return Proxy.newProxyInstance(beanClass.getClassLoader(), beanClass.getInterfaces(), (proxy, method, args) -> {
            // 方法需要增强则执行增强
            try {
                Method declaredMethod = beanClass.getDeclaredMethod(method.getName(), method.getParameterTypes());
                if (declaredMethod.isAnnotationPresent(Count.class)) {
                    synchronized (method.getName().intern()) {
                        countMap.put(method.getName(), countMap.getOrDefault(method.getName(), 0) + 1);
                        System.out.println(method.getName() + "====>" + countMap.get(method.getName()));
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println("为找到指定方法");
            }
            return method.invoke(bean, args);
        });
    }

}
```



## 属性赋值

Spring 属性赋值注解。

### @Value

Value 的用法，请看源码注释。这个注解还可作用于字段上。

```java
@Target({ElementType.FIELD, ElementType.METHOD, ElementType.PARAMETER, ElementType.ANNOTATION_TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface Value {

	/**
	 * The actual value expression such as <code>#{systemProperties.myProp}</code>
	 * or property placeholder such as <code>${my.app.myProp}</code>.
	 */
	String value();
}
```

使用 @Value 赋值

- 基本数值
- 可以写 SpEL；#{}
- 可以写 ${}; 取出配置文件中的值（在运行环境变量里面的值）；<b>properties 配置文件，放在 resource 目录下！！</b>
- pojo 对象

```java
public class Person {
    // 使用@Value赋值
    // 1 基本数值
    // 2 可以写SpEL， #{}，取出配置文件中的值
    @Value("张三")
    private String name;
    @Value("#{20-5}")
    private Integer age;

    public Person() {}

    public Person(String name) {
        this.name = name;
    }
	// 省略 setter getter
}
```

JavaConfig

```java
package org.example.configuration.assign;

import org.example.pojo.Person;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ValueConfig {
    @Bean
    public Person person() {
        return new Person();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(ValueConfig.class);
        Person person = context.getBean(Person.class);
        System.out.println(person);
    }
}
// output  Person{name='张三', age=15} 赋值成功
```

### @PropertySource

@PropertySource，读取配置文件中的值，将其保存到运行的环境中。@PropertySources 可以指定多个 @PropertySource。

properties 配置文件，在 resource 根目录下

```properties
person.name=zhangsan
```

为什么是在根目录下？请看该注解的注释！！

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Repeatable(PropertySources.class)
/**
   Given a file app.properties containing the key/value pair testbean.name=myTestBean, 
   the following @Configuration class uses @PropertySource to contribute app.properties to the Environment's set of PropertySources.
   @Configuration
   @PropertySource("classpath:/com/myco/app.properties")
   public class AppConfig {

       @Autowired
       Environment env;

       @Bean
       public TestBean testBean() {
           TestBean testBean = new TestBean();
           testBean.setName(env.getProperty("testbean.name"));
           return testBean;
       }
   }
**/
public @interface PropertySource {
	String name() default "";
	String[] value();
	boolean ignoreResourceNotFound() default false;
	String encoding() default "";
	Class<? extends PropertySourceFactory> factory() default PropertySourceFactory.class;
}
```

pojo

```java
package org.example.pojo;

import org.springframework.beans.factory.annotation.Value;

public class Person {
    // 使用@Value赋值
    // 1 基本数值
    // 2 可以写SpEL， #{}，取出配置文件中的值
    @Value("${person.name}")
    private String name;

    @Value("#{20-5}")
    private Integer age;

    @Override
    public String toString() {
        return "Person{ name='" + name + "'\' , age=" + age '}";
    }
}
```

JavaConfig

```java
import org.example.pojo.Person;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

@Configuration
// 使用@PropertySource读取外部配置文件中的k/v保存到运行的环境中
@PropertySource(value = {"classpath:/person.properties"})
public class PropertySourceConfig {

    @Bean
    public Person person() {
        return new Person();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(PropertySourceConfig.class);
        ConfigurableEnvironment environment = context.getEnvironment();
        System.out.println(environment.getProperty("person.name"));
        Person person = context.getBean(Person.class, "person");
        System.out.println(person);
    }
}
// Person{name='kkx', age=15}
```

## 自动装配

### 自动装配概述

<b>Spring 利用依赖注入（DI），完成对 IOC 容器中各个组件依赖关系的赋值；</b>

1️⃣@AutoWired：自动注入【Spring 定义的】

* 默认按照类型去容器中找对应的组件 applicationContext.getBean(BookService.class)，找到就赋值
* 如果找到相同类型的组件，再将属性的名称作为组件的 id 去容器中查找 applicationContext.getBean("bookDao")

2️⃣@Qualifier("bookDao")：使用该注解来指定需要装配的组件的 id，而不是使用属性名

3️⃣自动装配默认一定要将属性赋值好，没有就会报错，可通过在 Autowire 的注解中将 required=false 来使该配置设置为非必需

4️⃣@Primary：让 Spring 进行自动装配的时候，默认使用首选的 bean, 也可以继续使用 @Qualifier 来指定需要装配的 bean

<b>Spring 还支持使用 @Resource（JSR250）和 @Inject（JSR330）【Java 规范】</b>

 * @Resource：可以和 @Autowired 一样实现自动装配功能；默认是按照组件名称进行装配的；没有能支持 @Primary 的功能以及 @Autowired(required=false) 的功能
 * @Inject（需要导入依赖）：导入 javax.inject 的包，和 Autowired 的功能一样，没有 required=false 的功能

### @Autowired

<span style="color:red">先按类型来，找到就赋值；如果找到相同类型的组件，再将属性名作为组件的 id 去容器中查找。以前常见的一个错误，如果是按接口注入，找到了很多相同类型的组件，且属性名查找失败，则会提示 NoUniqueBeanDefinitionException</span>

- @Autowired
- @Autowired(required=false) 能装配上就装，不能就不装，默认为 true。

```java
@Configuration
@ComponentScan(basePackages = "org.example")
public class AutowiredConfig {

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AutowiredConfig.class);
        Book book = context.getBean(Book.class, "book");
        System.out.println("book's person name is " + book.person);
    }
}

@Service
class Book {
    @Autowired
    public Person person;
}
```

### @Qualifier

与 @Autowired 结合，指定装配什么名称的 Bean

### @Primary

首选的，主要的注解，默认装配时，会优先使用 @Primary 修饰的 Bean 进行自动装配。如果 @Qualifier 也指定了使用 xxx Bean 则按 @Qualifier 的规则进行装配。

```java
@Configuration
@ComponentScan(basePackages = "org.example")
public class AutowiredConfig {

    @Bean("p1")
    public Person person() {
        return new Person("1");
    }

    @Bean("p2")
    @Primary // 首选装配这个bean
    public Person person2() {
        return new Person("2");
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AutowiredConfig.class);
        Books book = context.getBean(Books.class, "book");
        System.out.println("book's person name is " + book.person);
    }
}

@Service("book")
class Books {
    @Autowired(required = false)
    public Person person;
}
```

### JSR250-@Resource

@Resource 是 Java 规范。也可以实现自动装配的功能，不过是默认是按照组件名称进行装配的。但是没能支持 @Primary 和 @Autowired(required=false) 这样的功能。

@Resource(name="p1")

需要导入 jar 包

```xml
<dependency>
    <groupId>jakarta.annotation</groupId>
    <artifactId>jakarta.annotation-api</artifactId>
    <version>1.3.5</version>
    <scope>compile</scope>
</dependency>
```

### JSR330-@Inject

@Inject 是 Java 规范，需要导入 jar 包。

```xml
<dependency>
    <groupId>javax.inject</groupId>
    <artifactId>javax.inject</artifactId>
    <version>1</version>
</dependency>
```

@Inject Autowired 的功能一样，没有 required=false 的功能，支持 @Primary，但是没有 required=false 的功能

<b>@Autowired 还是 JSR</b>

JSR 是会被其他 IOC 框架支持的，使用 JSR 的，脱离了 Spring，换其他 IOC 框架也可。

### 自动装配功能原理

AutowiredAnnotationBeanPostProcessor 解析完成自动装配功能

- AutowiredAnnotationBeanProcessor 类

```java
public class AutowiredAnnotationBeanPostProcessor implements SmartInstantiationAwareBeanPostProcessor,
		MergedBeanDefinitionPostProcessor, PriorityOrdered, BeanFactoryAware {
  	public AutowiredAnnotationBeanPostProcessor() {
		this.autowiredAnnotationTypes.add(Autowired.class);
		this.autowiredAnnotationTypes.add(Value.class);
		try {
			this.autowiredAnnotationTypes.add((Class<? extends Annotation>)
					ClassUtils.forName("javax.inject.Inject", AutowiredAnnotationBeanPostProcessor.class.getClassLoader()));
			logger.trace("JSR-330 'javax.inject.Inject' annotation found and supported for autowiring");
		}
		catch (ClassNotFoundException ex) {
			// JSR-330 API not available - simply skip.
		}
	}
}
```

### 方法、构造器位置

构造器注入容易发生循环依赖。

#### 方法

<b>@Autowired：构造器，参数，方法，属性</b>

1️⃣<b>标注在方法位置：</b>标注在方法，Spring 容器创建当前对象，就会调用方法，完成赋值。方法使用的参数，自定义类型的值从 IOC 容器中获取，@Bean 标注的方法创建对象的时侯，<span style="color:orange">方法参数的值默认从 IOC 容器中获取，写不写 @Autowired，效果都是一样的。</span>

2️⃣<b>标注在构造器位置：</b>默认加载 IOC 容器中的组件，容器启动会调用无参构造器创建对象，再进行初始化赋值等操作。标注在构造器上可以指定创建对象时使用该构造方法，方法中用的参数同样从 IOC 容器中获取。<span style="color:orange">如果容器只有一个有参构造器，这个有参构造器的 @Autowired 可以省略，参数位置的组件还是可以自动从容器中获取。</span>

3️⃣<b>标注在参数位置：</b>从 IOC 容器中获取参数组件的值。

4️⃣<b>标注在属性位置：</b>

> @Bean 的示例

```java
@Bean
// 这个 car 默认就是从 IoC 容器获取的，不用加 @Autowired 注解
public Color color(Car car){
    return new Color(car);
}
```

#### 构造器

@Component 注解。

默认在加载 IOC 容器中的组件，容器启动会调用无参构造器创建对象，再进行初始化赋值等操作。

如果当前类只有一个有参构造器，那么 Autowired 是可以省略的。@Bean 注入，若只有一个有参构造则也是可以省略的。

```java
@Component
public class Boss{
    private Car car;
    
    public Boss(@Autowired Car car){
        this.car = car;
    }
}

// 等价于
@Component
public class Boss{
    private Car car;
    
    public Boss(Car car){
        this.car = car;
    }
}
```

### Aware注入Spring底层原理

#### 概述

自定义组件想要使用 Spring 容器底层的一些组件时（如：ApplicationContext，BeanFactory，xxx），只需要让自定义组件实现 xxxAware 接口。在创建对象的时候，会调用xxxAware 接口中规定的方法注入相关组件。

```java
import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

@Configuration
public class AwareConfig {

    @Bean
    public Dog dog() {
        return new Dog();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AwareConfig.class);
        Dog bean = context.getBean(Dog.class);
        context.close();
    }
}

class Dog implements ApplicationContextAware {

    private ApplicationContext context;

    // 在 dog 对象创建后，这个方法会被调用，为 Dog 对象注入 context 组件。当然，你直接使用 @Autowired 
    @Override
    public void setApplicationContext(ApplicationContext context) throws BeansException {
        this.context = context;
        System.out.println("context hashcode is " + context);
    }

    public Dog() {
        System.out.println("class dog construct");
    }

    @PostConstruct
    public void init() {
        System.out.println("class dog post construct");
    }

    @PreDestroy
    public void destroy() {
        System.out.println("class dog destroy construct");
    }
}
```

#### 常用的Aware接口

1️⃣ApplicationContextAware 设置 ApplicationContext 对象

2️⃣BeanNameAware 设置 BeanName

3️⃣EmbeddedValueResolverAware 解析字符用

- 字符串解析，如解析 `#{} ${}`，表达式解析？【占位符解析】
- `${}`取出环境变量中的值。`#{}` Spring 的表达式语言

4️⃣使用 xxxProcessor 进行处理的，每个 xxxAware 都有对应的 xxxProcessor

- 利用后置处理器，判断这个 Bean。是这个 Aware 接口，然后把组件传过来。

```java
package cn.study.ioc.aware;

import org.springframework.beans.BeansException;
import org.springframework.beans.factory.BeanNameAware;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.EmbeddedValueResolverAware;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringValueResolver;

@Configuration
public class AwareCommonConfig {

    @Bean
    public AwareCommonDemo get() {
        return new AwareCommonDemo();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AwareCommonConfig.class);
        AwareCommonDemo bean = context.getBean(AwareCommonDemo.class);
        System.out.println(bean);
    }
    /*
    setBeanName is get
    setEmbeddedValueResolver is os.name is Windows 10 Spring EL 7
    setApplicationContext org.springframework.context.annotation.AnnotationConfigApplicationContext@60c6f5b, started on Sat May 28 21:34:02 CST 2022
    cn.study.ioc.aware.AwareCommonDemo@2657d4dd
    */
}

class AwareCommonDemo implements ApplicationContextAware, BeanNameAware, EmbeddedValueResolverAware {
    private ApplicationContext context;

    @Override
    public void setBeanName(String name) {
        // 设置bean的名字
        System.out.println("setBeanName is " + name);
    }

    @Override
    public void setApplicationContext(ApplicationContext context) throws BeansException {
        this.context = context;
        System.out.println("setApplicationContext " + context);
    }

    @Override
    public void setEmbeddedValueResolver(StringValueResolver resolver) {
        String s = resolver.resolveStringValue("os.name is ${os.name} Spring EL #{12-5}");
        System.out.println("setEmbeddedValueResolver is " + s);
    }
}
```

#### Aware注入原理

以 ApplicationContextAware 为例，对 ApplicationContext 的注入原理进行分析。

> 测试代码

```java
@Component
public class UserOne implements ApplicationContextAware {
    private ApplicationContext context;

    @Override
    public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
        this.context = applicationContext;
    }
}
```

<div align="center"><img src="img/image-20230108163506440.png"></div>

ApplicationContextAwareProcessor --> 调用 postProcessBeforeInitialization方法--> 调用 invokeAwareInterfaces



### Profile注解

#### 概述

可根据当前环境，动态激活和切换一系列组件的功能。环境被激活了，才可用。如何激活？【使用命令参数 】

- 开发环境
- 测试环境
- 生产环境

如不同环境的数据库不一样。可用数据源切换达成。

- 写在方法上，在指定环境中方法注入的 Bean 才生效
- 写在类上，在指定环境中，该类的相关信息和配置信息才生效

```java
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Conditional(ProfileCondition.class)
public @interface Profile {

	// The set of profiles for which the annotated component should be registered.
	String[] value();
}
```

#### 数据源切换

通过加上 @Profile 注解指定组件在那个环境的情况下才能被注册到容器中，不指定，任何环境下都能注册该组件。默认是 default 环境。

- 添加 C3P0 数据源

```xml
<dependency>
    <groupId>c3p0</groupId>
    <artifactId>c3p0</artifactId>
    <version>0.9.1.2</version>
</dependency>
```

- 注册数据源

```java
import com.mchange.v2.c3p0.ComboPooledDataSource;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.EmbeddedValueResolverAware;
import org.springframework.context.annotation.*;
import org.springframework.util.StringValueResolver;

import javax.sql.DataSource;
import java.beans.PropertyVetoException;
import java.util.stream.Stream;

@Configuration
public class ProfileConfig {

    public static void test1() {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(ProfileDemo.class);
        String[] beanNamesForType = context.getBeanNamesForType(DataSource.class);
        Stream.of(beanNamesForType).forEach(System.out::println);
    }

    public static void main(String[] args) {
        test1();
    }
}

@PropertySource("classpath:/dbconfig.properties")
class ProfileDemo implements EmbeddedValueResolverAware {

    private StringValueResolver resolver;

    @Value("${db.user}")
    private String user;
    @Value("${db.password}")
    private String password;
    @Value("${db.driverClass}")
    private String driverClass;
	
    // 加了环境标识的 bean，只有这个环境被激活的时候才能注册到容器中
    @Profile("test")
    @Bean("testDataSource")
    public DataSource dataSourceTest() throws PropertyVetoException {
        ComboPooledDataSource dataSource = new ComboPooledDataSource();
        dataSource.setUser(user);
        dataSource.setPassword(password);
        dataSource.setDriverClass(driverClass);
        dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC");
        return dataSource;
    }

    @Profile("prod")
    @Bean("prodDataSource")
    public DataSource dataSourceProd() throws PropertyVetoException {
        ComboPooledDataSource dataSource = new ComboPooledDataSource();
        dataSource.setUser(user);
        dataSource.setPassword(password);
        dataSource.setDriverClass(driverClass);
        dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mysql_book?serverTimezone=UTC");
        return dataSource;
    }

    @Override
    public void setEmbeddedValueResolver(StringValueResolver resolver) {
        this.resolver = resolver;
    }
}
```

<b>激活环境后 bean 才有效。如何激活？</b>

1️⃣使用命令行动态参数：在虚拟机参数位置加载 `-Dspring.profiles.active=test`

- IDEA是在 `VM options` 里面写参数 `-Dspring.profiles.active=test`
- `Eclipse `是在 `VM arguments` 里面写参数

2️⃣使用代码激活。需要注意的是，容器创建的时候会创建一些列的 bean，就无法做到按照环境进行激活 bean 了，因此先创建一个空的 context，然后再将相关的配置代码注册到 context 中。

```java
import com.mchange.v2.c3p0.ComboPooledDataSource;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.EmbeddedValueResolverAware;
import org.springframework.context.annotation.*;
import org.springframework.util.StringValueResolver;

import javax.sql.DataSource;
import java.beans.PropertyVetoException;
import java.util.stream.Stream;

public class ProfileConfig {

    public static void main(String[] args) {
        // 调用无参构造器！
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext();
       	// 设置环境，激活 test 和 prod 环境
        context.getEnvironment().setActiveProfiles("test", "prod");
        // 注册配置
        context.register(ProfileDemo.class);
        // 刷新容器
        context.refresh();
        String[] beanNamesForType = context.getBeanNamesForType(DataSource.class);
        Stream.of(beanNamesForType).forEach(System.out::println);
    }
}

@PropertySource("classpath:/dbconfig.properties")
@Configuration
class ProfileDemo implements EmbeddedValueResolverAware {

    private StringValueResolver resolver;

    @Value("${db.user}")
    private String user;
    @Value("${db.password}")
    private String password;
    @Value("${db.driverClass}")
    private String driverClass;

    @Profile("test")
    @Bean("testDataSource")
    public DataSource dataSourceTest() throws PropertyVetoException {
        ComboPooledDataSource dataSource = new ComboPooledDataSource();
        dataSource.setUser(user);
        dataSource.setPassword(password);
        dataSource.setDriverClass(driverClass);
        dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC");
        return dataSource;
    }

    @Profile("prod")
    @Bean("prodDataSource")
    public DataSource dataSourceProd() throws PropertyVetoException {
        ComboPooledDataSource dataSource = new ComboPooledDataSource();
        dataSource.setUser(user);
        dataSource.setPassword(password);
        dataSource.setDriverClass(driverClass);
        dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mysql_book?serverTimezone=UTC");
        return dataSource;
    }

    @Override
    public void setEmbeddedValueResolver(StringValueResolver resolver) {
        this.resolver = resolver;
    }
}
```

## 带泛型的DI

父类类型 com.xxx.xxx.BaseService

带泛型的父类类型 com.xxx.xxx.BaseService<com.xx.Book>

Spring 可以用带泛型的父类类型来确定这个子类的类型

obj.getClass.getGeneriSuperclass()

泛型依赖注入，注入一个组件的时候，他的泛型也是参考标准。

## IOC小结

### 容器

- AnnotationConfigApplicationContext
- 组件添加
    - @ComponentScan
    - @Bean、@Configuration
    - @Component
    - @Service
    - @Controller
    - @Repository
    - @Conditional★
    - @Primary
    - @Lazy
    - @Scope 
    - @Import★
    - ImportSelector
    - 工厂模式
- 组件赋值
    - @Value [ ${} 读 properties 文件  #{} 表达式语言 ]
    - @Autowired
        - @Qualifier
        - 其他方式 [ @Resource (JSR250)  @Inject (JSR330, 需要导入  javax.inject) ]
    - @PropertySource
    - @PropertySources
    - @Prifile
        - Environment
        - -Dspring.profiles.active=test
- 组件注入
    - 方法参数 
    - 构造器注入
    - ApplicationContextAware --> ApplicationContextAwareProcessor
    - xxxAware
- AOP
- 声明式事务

### 扩展原理

- BeanFactoryPostProcessor
- BeanDefinitionRegistryPostProcessor
- ApplicationListener
- Spring 容器创建过程

### 其他

IOC 是一个容器，棒我们管理所有的组件

1️⃣依赖注入：@Autowired 自动赋值

2️⃣某个组件要使用 Spring 提供的更多（IOC、AOP）必须加入到容器中

3️⃣容器启动。创建所有单实例 Bean

4️⃣@Autowired 自动装配的时候，是从容器中找这些符合要求的 bean。

5️⃣ioc.getBean("bookServlet")；也是从容器中找到这个 bean

6️⃣容器中包含了所有的 bean

7️⃣探索，单实例的 bean 都保存到了哪个 map 中了。【源码-扩展】

8️⃣源码调试思路：从 HelloWorld 开始；给 HelloWorld 每一个关键步骤打上断点，进去看里面都做了些什么工作。怎么知道哪些方法都是干什么的？

​	-  翻译方法名称，猜猜是做什么的

​	-  放行这个方法，看控制台

​	-  看方法注释

9️⃣创建 Java 对象做了那些事？

​	- 实例化：在堆空间中申请一块空间，对象的属性值都是默认的。

​	- 初始化：填充属性，调用初始化方法。

# AOP

AOP：面向切面编程

OOP：面向对象编程

面向切面编程：基于 OOP 基础之上新的编程思想；

指在程序运行期间，将某段代码动态的切入到指定方法的指定位置进行运行的这种编程方式，面向切面编程；

> 使用场景

==>日志记录

==>事务控制

> 基本概念

AOP 的全称是 Aspect-Oriented Programming，即面向切面编程（也称面向方面编程）。它是面向对象编程（OOP）的一种补充。

在传统的业务处理代码中，通常都会进行事务处理、日志记录等操作。虽然使用 OOP 可以通过组合或者继承的方式来达到代码的重用，但如果要实现某个功能（如日志记录），同样的代码仍然会分散到各个方法中。这样，如果想要关闭某个功能，或者对其进行修改，就必须要修改所有的相关方法。增加了开发人员的工作量，提高了代码的出错率。

AOP 采取横向抽取机制，将分散在各个方法中的重复代码提取出来，然后在程序编译或运行时，再将这些提取出来的代码应用到需要执行的地方。

<b>PS：AOP 是横向抽取机制，OOP 是父子关系的纵向的重用。</b>

<div align="center"><img src="img/spring/aop.jpg"></div>

> 环境搭建 -- 导入 aop 模块

```xml
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-aspects</artifactId>
    <version>5.3.3</version>
</dependency>
```

## AOP术语

Aspect、Joinpoint、Pointcut、Advice、TargetObject、Proxy 和 Weaving。

1️⃣<b>Aspect（切面）</b>，在实际应用中，切面通常是指封装的用于横向插入系统功能（如事务、日志等）的类。

2️⃣Joinpoint（连接点），在程序执行过程中的某个阶段点，它实际上是对象的一个操作，例如方法的调用或异常的抛出。<span style="color:orange">在 Spring AOP 中，连接点就是指方法的调用。</span>

3️⃣<b>Pointcut（切入点）</b>，是指切面与程序流程的交叉点，即那些需要处理的连接点，<span style="color:orange">通常在程序中，切入点指的是类或者方法名，如某个通知要应用到所有以 add 开头的方法中，那么所有满足这一规则的方法都是切入点。</span>

4️⃣<b>Advice（通知/增强处理）</b>，AOP 框架在特定的切入点执行的增强处理，即在定义好的切入点处所要执行的程序代码。<span style="color:orange">可以将其理解为切面类中的方法，它是切面的具体实现。</span>

5️⃣Target Object（目标对象），是指所有被通知的对象，也称为<span style="color:orange">被增强对象</span>。如果 AOP 框架采用的是动态的 AOP 实现，那么该对象就是一个被代理对象。

6️⃣Proxy（代理），将通知应用到目标对象之后，被动态创建的对象。

7️⃣Weaving（织入），将切面代码插入到目标对象上，从而<span style="color:orange">生成代理对象的过程</span>

> <b>几种通知</b>

前置通知：在目标方法之前运行，@Before		

后置通知：在目标方法结束之后运行，@After

返回通知：在目标方法正常返回之后运行，@AfterReturning

异常通知：在目标方法抛出异常之后运行，@AftreThrowing

环绕通知：就是一个动态代理，手动推进目标方法的运行（joinPoint.procced()）			

```java
try{
    @Before
    method.invoke(obj,args);
    @AfterReturning
}catch(e){
    @AftreThrowing
}finally{
    @After
}
/*
通知注解
@Before：在目标方法之前运行				前置通知
@After：在目标方法结束之后				后置通知
@AfterReturning：在目标方法正常返回之后		返回通知
@AftreThrowing：在目标方法抛出异常之后运行	异常通知
@Around：环绕							环绕通知
*/
```

<span style="color:red">通知只是告知执行的时机</span>，那到底在那些方法上进行增强呢？<span style="color:red">用切入点表达式告知对那些方法进行增强</span>。

> 重要概念图

<div align="center"><img src="img/spring/SpringAOP.png"></div>

## 动态代理

Spring 中的 AOP 是基于代理实现的，可以是 JDK 动态代理，也可以是 CGLIB 代理。

### JDK动态代理

JDK 动态代理是通过 java.lang.reflect.Proxy 类来实现的，我们可以调用 Proxy 类的 newProxyInstance() 方法来创建代理对象。对于使用业务接口的类，Spring 默认会使用 JDK 动态代理来实现 AOP。

> 以下是 JDK 动态代理的代码示例

切面代码

```java
public class MyAspect {
    public void checkPermissions() {
        System.out.println("检查权限");
    }

    public void log(){
        System.out.println("记录日志");
    }
}
```

接口及实现类

```java
public interface UserDao {
    void addUser();
    void deleteUser();
}

public class UserDaoImpl implements UserDao {
    @Override
    public void addUser() {
        System.out.println("add");
    }

    @Override
    public void deleteUser() {
        System.out.println("delete");
    }

    public static void main(String[] args) {
        JdkProxy jdkProxy = new JdkProxy();
        UserDao dao = (UserDao) jdkProxy.createProxy(new UserDaoImpl());
        dao.addUser();
    }
}
```

动态代理

```java
public class JdkProxy implements InvocationHandler {

    // 目标类接口
    private UserDao userDao;

    public Object createProxy(UserDao userDao) {
        this.userDao = userDao;
        ClassLoader classLoader = JdkProxy.class.getClassLoader();
        Class<?>[] interfaces = userDao.getClass().getInterfaces();
        // 当前类的类加载器
        return Proxy.newProxyInstance(classLoader, interfaces, this);
    }

    /**
     * 所有动态代理类的方法调用，都会交由 invoke 方法处理
     * @param proxy  被代理后的对象 class com.sun.proxy.$Proxy0 假定原来是 A 对象，然后由 $Proxy0 代理对象 A。
     * @param method 将要被执行的方法信息
     * @param args   执行方法时需要的参数
     */
    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        MyAspect aspect = new MyAspect();
        System.out.println("proxy "+proxy.getClass());
        aspect.checkPermissions();
        Object invoke = method.invoke(userDao, args);
        aspect.log();
        return invoke;
    }
}
```

### CGLIB代理

CGLIB（Code Generation Library）是一个高性能开源的代码生成包，它采用非常底层的字节码技术，对指定的目标类生成一个子类，并对子类进行增强。Spring 的核心包中已经集成了 CGLIB 所需要的包，如果是单独使用 CGLIB 则需要导入相关的依赖包。

<span style="color:orange">CGLIB 无需使用接口，比 JDK 动态代理方便一些。</span>

```java
public class UserDao {
    public void addUser() {
        System.out.println("add");
    }

    public void deleteUser() {
        System.out.println("delete");
    }
}

public class CGLIBProxy implements MethodInterceptor {

    public Object createProxy() {
        Enhancer enhancer = new Enhancer();
        enhancer.setSuperclass(UserDao.class);
        // 添加回调函数
        enhancer.setCallback(this);
        // 返回创建的代理类
        return enhancer.create();
    }

    @Override
    /**
     * proxy        CGLib 根据指定父类生成的代理对象
     * method       拦截的方法
     * args         拦截方法的参数
     * methodProxy  方法的代理对象，用于执行父类的方法
     */
    public Object intercept(Object proxy, Method method, Object[] args, MethodProxy methodProxy) throws Throwable {
        MyAspect aspect = new MyAspect();
        aspect.checkPermissions();
        Object retVal = methodProxy.invokeSuper(proxy, args);
        aspect.log();
        return retVal;
    }
}

```

## 基于代理类的AOP实现

Spring 中的 AOP 代理默认就是使用 JDK 动态代理的方式来实现的。在 Spring 中，使用 ProxyFactoryBean 是创建 AOP 代理的最基本方式。

ProxyFactoryBean 是 FactoryBean 接口的实现类，FactoryBean 负责实例化一个 Bean，而 ProxyFactoryBean 负责为其他 Bean 创建代理实例。在 Spring 中，使用 ProxyFactoryBean 是创建 AOP 代理的基本方式。

<div align="center"><b>ProxyFactoryBean 的常用属性</b></div>

| 属性名称         | 描述                                                      |
| ---------------- | --------------------------------------------------------- |
| target           | 代理的目标对象                                            |
| proxyInterfaces  | 代理类要实现的接口                                        |
| proxyTargetClass | 是否对类代理而不是接口，设置为 true 时使用 cglib 动态代理 |
| interceptorNames | 需要织入目标的 Advice                                     |
| singleton        | 返回的代理是否为单实例，默认为 true                       |
| optimize         | 设置为 true 时，强制使用 cglib                            |

ProxyFactoryBean 使用示例：

1️⃣定义接口

```java
public interface UserDao {
    void addUser();
    void deleteUser();
}
```

2️⃣接口实现类

```java
@Service
public class UserDaoImpl implements UserDao {
    @Override
    public void addUser() {
        System.out.println("add");
    }

    @Override
    public void deleteUser() {
        System.out.println("delete");
    }
}
```

3️⃣AOP 通知

```java
import org.springframework.aop.MethodBeforeAdvice;
import org.springframework.stereotype.Component;

import java.lang.reflect.Method;


@Component("beforeAop")
public class BeforeAOP implements MethodBeforeAdvice {
    @Override
    public void before(Method method, Object[] args, Object target) throws Throwable {
        System.out.println("执行了前置通知");
    }
}

```

4️⃣测试代码

```java
import org.springframework.aop.framework.ProxyFactoryBean;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

public class Test {

    public static void main(String[] args) {

        AnnotationConfigApplicationContext ac = new AnnotationConfigApplicationContext("cn.study.aop");

        // 使用Spring 的 AOP，
        // 配置好 ProxyFactoryBean，给 ProxyFactoryBean 设置一个bean id
        // 通过 ac.getBean, 可以获得 ProxyFactoryBean代 理的对象，不是 ProxyFactoryBean
        // 这个bean id 虽然代表 ProxyFactoryBean 对象，直接 getBean 获取的是 
        // ProxyFactoryBean.getObject()返回的对象，即代理对象
        //ac.getBean(&bean id),才能取得 ProxyFactoryBean 对象

        ProxyFactoryBean proxyFactoryBean = new ProxyFactoryBean();

        proxyFactoryBean.setBeanFactory(ac.getBeanFactory());
        // aop拦截处理类
        proxyFactoryBean.setInterceptorNames("beforeAop");

        // 代理的接口
        proxyFactoryBean.setInterfaces(UserDao.class);

        // 被代理对象
        proxyFactoryBean.setTarget(ac.getBean(UserDaoImpl.class));

        // 放入bean工厂，实际开发是在config下使用注解，设置多个proxyFactoryBean代理，设置不同bean id
        ac.getBeanFactory().registerSingleton("myProxy", proxyFactoryBean);

        UserDao servInterProxy = ac.getBean("myProxy", UserDao.class);
        servInterProxy.addUser();
        // 获取直接的ProxyFactoryBean对象，加&
        System.out.println(ac.getBean("&myProxy"));
    }
}
```

## AspectJ开发

### AOP注解

点进 `@EnableAspectJAutoProxy` 注解里，会发现文档注释里给了很详细的用法！！！AspectJ 相关注解如下表。

| 注解            | 描述                                                         |
| --------------- | ------------------------------------------------------------ |
| @Aspect         | 定义切面                                                     |
| @Pointcut       | 定义切入点表达式                                             |
| @Before         | 定义前置通知                                                 |
| @AfterReturning | 定义后置通知，returning 属性用于表示 Advice 方法中可定义与此同名的形参，<br>该形参可用于访问目标方法的返回值。（简单说就是方法的返回值） |
| @Around         | 定义环绕通知                                                 |
| @AfterThrowing  | 定义异常通知，returning 属性用于表示 Advice 方法中可定义与此同名的形参，<br/>该形参可用于访问目标方法的抛出的异常。 |
| @After          | 定义最终 final 通知                                          |
| @DeclareParents | 定义引介通知                                                 |

### AOP注解开发

> <b>三步走</b>

- 将业务逻辑组件和切面类都加入到容器中，告诉 Spring 哪个是切面类（<span style="color:green">@Aspect 注解标注</span>）
- 在切面类上的每一个通知方法上标注通知注解，告诉 Spring 何时何地运行（<span  style="color:green">切入点表达式</span>）
    - @After("public int com.cc.ClassName.method(int,int)")
- 开启基于注解的 `aop` 模式：`@EnableAspectJAutoProxy`，注意，这个是要加载配置类中的，如果主启动类不是配置类，那么加在主启动类上是无效的！

> <b>具体步骤</b>

 * 1️⃣导入 AOP 模块：Spring AOP（spring-aspects）
 * 2️⃣定义一个业务逻辑类（MathCalculator），在业务逻辑运行的时候将日志进行打印（方法运行之前，方法运行之后，方法出现异常，xxx）。
 * 3️⃣定义一个日志切面类（LogAspects），切面里面的方法需要动态感知 MathCalculator.div 运行到了哪里，然后执行。
    * --------通知方法：
    * -----------前置通知 (@Before)：logStart 在目标方法（div）运行之前运行
    * -----------后置通知 (@After)：logEnd 在目标方法（div）运行结束之后运行
    * -----------返回通知 (@AfterReturning)：logReturn 在目标方法（div）正常返回之后
    * -----------异常通知 (@AfterThrowing)：logException 在目标方法（div）出现异常以后运行
    * -----------环绕通知 (@Around)：动态代理，手动推进目标方法运行（joinPoint.procced()）
 * 4️⃣给切面类的目标方法标准何时何地运行 (通知注解)
 * 5️⃣将切面类和业务逻辑类 (目标方法所在类) 都加入到容器中
 * 6️⃣必须告诉Spring，那个类是切面类 (给切面类加注解)
 * 7️⃣给配置类中加 @EnableAspectJAutoProxy [开启基于注解的 AOP 模式]

在 Spring 中 EnableXxx 都是开启某项功能的。

配置环境

```xml
<!-- aop 需要再额外导入切面包 -->
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-aspects</artifactId>
    <version>5.3.3</version>
</dependency>
```

业务类代码

```java
@Service
public class MathCalculator {
    public void div(int n, int eps) {
        System.out.println(n / eps);
    }
}
```

切面类代码，切面类的注册可以用 @Component 也可以用 @Bean 注入。

```java
@Aspect
@Component
public class LogAspects {
    @Pointcut("execution(* com.review.spring.service.MathCalculator.*(..))")
    public void pointCut() {
    }

    @Before("pointCut()")
    // JoinPoint一定要出现在参数列表的第一位
    public void logStart(JoinPoint joinPoint) {
        Signature signature = joinPoint.getSignature();
        Object[] args = joinPoint.getArgs();
        System.err.println("log Before 的方法签名是：" + signature + " 参数列表是：" + Arrays.asList(args));
    }

    @After("pointCut()")
    public void logEnd() {
        System.err.println("log After");
    }

    @AfterReturning(value = "pointCut()", returning = "res")
    public void logReturn(Object res) {
        System.err.println("log AfterReturning, 运行结果是：" + res);
    }

    @AfterThrowing(value = "pointCut()", throwing = "exc")
    public void logException(JoinPoint joinPoint, Exception exc) {
        System.err.println("log logException, 方法签名是：" + joinPoint.getSignature().getName() + ",异常是：" + exc);
    }
}
```

测试代码

```java
@ComponentScan(basePackages = "com.review.spring")
@EnableAspectJAutoProxy
public class Main {

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(Main.class);
        MathCalculator bean = context.getBean(MathCalculator.class);
        bean.div(1, 2);
    }
}
/*
log Before 的方法签名是：void com.review.spring.service.MathCalculator.div(int,int) 参数列表是：[1, 2]
log AfterReturning, 运行结果是：null
log After
*/
```

AOP 创建的是代理对象不是创建原有的 Object 对象，而是创建它的代理对象 ObjectProxy。IOC 中有代理对象，但是没有原对象！

### 通知方法的执行顺序

Spring5 和 Spring4 通知方法的执行顺序不一样了。

> Spring5 是先执行前置通知 @Before，最后执行后置通知 @After。

正常执行：@Before（前置通知）---- @AfterReturning（正常返回）---- @After（后置通知）

异常执行：@Before（前置通知）---- @AfterThrowing（方法异常）---- @After（后置通知）

> Spring4 是 @After 在中间执行，即 

正常执行：@Before（前置通知）---- @After（后置通知）---- @AfterReturning（正常返回）

异常执行：@Before（前置通知）---- @After（后置通知）---- @AfterThrowing（方法异常）

### 其他细节

- JoinPoint 获取目标方法的信息
- 在注解 @AfterReturning 和 @AfterThrowing 为 throwing，return 赋值可以用于接收异常信息、返回值

> 告诉 Spring 哪个参数是用来接受异常

```java
// JoinPoint 在第一位！ Exception用最大的异常来接收！
public static void sfasf(JoinPoint join, Exception ex){
    // do somethings
}
```

> 环绕通知

其实就是动态代理的一次简单封装。环绕通知和其他通知共同作用的情况下：

环绕通知先运行，且环绕通知把异常处理了，其他方法就感受不到异常了！为了能让外界知道这个异常，这个异常一定要抛出去！`throw new RuntimeException()`，不抛出异常的典型错误有 Spring 事务失效。

要是写动态代理的话，可以用环绕通知。

多切面运行的话，可以用 @Order 注解改变切面顺序！

```java
@Aspect
@Component
@Order(1)// 使用Order改变切面顺序
```

## AOP源码解析

<b>使用步驟回顾</b>

1. @EnableAspectJAutoProxy 开启基于注解的 aop 模式
2. @Aspect：定义切面类，切面类里定义通知
3. @PointCut 切入点，可以写切入点表达式，指定在哪个方法切入
4. 通知方法
    - @Before (前置通知)
    - @After (后置通知)
    - @AfterReturning (返回通知)
    - @AfterTrowing (异常通知) 
    - @Around(环绕通知)
5. JoinPoint：连接点，是一个类，配合通知使用，用于获取切入的点的信息

### AOP原理导读

<b>原理：我们看的是它给容器中注册了什么组件，组件的功能是什么，如何注册这个组件，这个组件又是什么时候工作。</b>

- 1）`@EnableAspectJAutoProxy`

- 2）`AspectJAutoProxyRegistrar`
- 3）`AnnotationAspectJA`
- 4）`AnnotationAwareAspect`

<b>AOP 后置处理器的注册</b>

1. 在配置类上加上注解 @EnableAspectJAutoProxy，该注解会导入 AspectJAutoProxyRegistrar 类。
2. AspectJAutoProxyRegistrar 会注册后置处理器 AnnotationAwareAspectJAutoProxyCreator，该后置处理器用与在 Bean 创建之前查看该 Bean 是否需要创建代理对象进行 AOP 增强。

<b>AOP 对象的代理流程</b>

1. 先在 IoC 容器中创建 Bean 后置处理器，这些后置处理器用来对 Bean 做一些操作，而 AOP 的后置处理器就是对 Bean 做动态代理增强的。
2. 然后创建普通的 Bean，在创建普通 Bean 的时候会使用后置处理器对符合后置处理器匹配规则的 Bean 做一些处理操作。
    - 在 AOP 中的体现是，创建 Bean 的时候会先调用这些后置处理器，调用 AOP 后置处理器的时候，会看是否要创建代理对象，需要创建的话就创建出代理对象，然后返回。这样使用的时候用的就是经过增强后的代理对象了。

<b>如何阅读相关源码？</b>

1. 写一个最基本的 demo，运行起来。
2. 在所写代码逻辑/其他你所知道的关键代码逻辑上打上断点，然后借助 IDE 的断点调试栈看是如何一步一步执行到此处的
3. 分析方法调用栈中的每一步，借助方法名和类名推断出关键的类和方法，逐个击破。

<b>阅读顺序</b>

```mermaid
graph LR
EnableAspectJAutoProxy-->AnnotationAwareAspectJAutoProxyCreator-->BeanPostProcessorsAfterInitialization-->CglibAopProxy.intercept
```

<b>相关内容</b>

通过查阅一些 Spring 创建的注解可以发现，这些注解大多都是用 xxxAnnotationBeanPostPocessor 来处理的。而 @Enable 开头的注解往往是用 @Import 导入了一个类，然后通过该类来做一些功能的注册。以后阅读源码都可以套用这个流程进行阅读。

但是阅读源码是为了更好的使用框架，遇到问题时有分析的思路，不要为了读源码而读源码。要有所收获：使用上的收获、代码设计上的收获 etc...

### EnableAspectJAutoProxy

`@EnableAspectJAutoProxy` 注解是 Spring AOP 开启的标志（即注册 AOP 的后置处理器 AnnotationAwareAspectJAutoProxyCreator），在启动类/配置类上标记此注解，即可加载对应的切面类（这些普通的非后置处理器的 Bean 都是在 BeanPostProcessor 后面注入的）逻辑。

通过 `@Import(AspectJAutoProxyRegistrar.class)` 给 spring 容器中导入了一个 `AnnotationAwareAspectJAutoProxyCreator` 来注册 AOP 后置处理器。通过创建后置处理器，在创建普通 bean 的时候，判断是否需要用 AOP 后置处理器来创建 AOP 代理类，从而实现 AOP 功能增强。

- 1️⃣AspectJAutoProxyRegistrar 类实现了 ImportBeanDefinitionRegistrar 接口，可以手动加载组件。利用 AspectJAutoProxyRegistrar 在容器中注册 bean 的定义信息 AnnotationAwareAspectJAutoProxyCreator。
- 1️⃣查看 AnnotationAwareAspectJAutoProxyCreator 的继承关系，发现他是一个 bean 后置处理器。
- 3️⃣回想之前写的自定义注解解析的 BeanPostProcessor，后置处理器在 bean 初始化完成前后做事情，AnnotationAwareAspectJAutoProxyCreator 这个后置处理器被注册到容器后，普通的 Bean 在执行 initMethod 的前后就会去执行这些后置处理器的方法。

> @EnableAspectJAutoProxy 注解所作的工作

导入了 AspectJAutoProxyRegistrar 类来注册后置处理器，因此主要关注后置处理器的注册流程。

```mermaid
graph LR
AspectJAutoProxyRegistrar-->|执行方法registerBeanDefinitions|注册后置处理器
```

注册后置处理器用的是 AopConfigUtils 类的方法。

```java
public void registerBeanDefinitions(
    AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry) {
	
    // 注册后置处理器。
    AopConfigUtils.registerAspectJAnnotationAutoProxyCreatorIfNecessary(registry);

    AnnotationAttributes enableAspectJAutoProxy =
        AnnotationConfigUtils.attributesFor(importingClassMetadata, EnableAspectJAutoProxy.class);
    if (enableAspectJAutoProxy != null) {
        if (enableAspectJAutoProxy.getBoolean("proxyTargetClass")) {
            AopConfigUtils.forceAutoProxyCreatorToUseClassProxying(registry);
        }
        if (enableAspectJAutoProxy.getBoolean("exposeProxy")) {
            AopConfigUtils.forceAutoProxyCreatorToExposeProxy(registry);
        }
    }
}
```

一路跟踪 AopConfigUtils 的方法，发现注册的是 AnnotationAwareAspectJAutoProxyCreator 类，查看该类的继承关系可以发现，这个类应该就是后置处理器。

```java
public static BeanDefinition registerAspectJAnnotationAutoProxyCreatorIfNecessary(
    BeanDefinitionRegistry registry, @Nullable Object source) {
	// 注册的是
    return registerOrEscalateApcAsRequired(AnnotationAwareAspectJAutoProxyCreator.class, registry, source);
}
```

把它封装成 BeanDefinition，然后就可以注册到 IoC 容器了。对应的名称为  internalAutoProxyCreator

```java
private static BeanDefinition registerOrEscalateApcAsRequired(
    Class<?> cls, BeanDefinitionRegistry registry, @Nullable Object source) {

    Assert.notNull(registry, "BeanDefinitionRegistry must not be null");

    if (registry.containsBeanDefinition(AUTO_PROXY_CREATOR_BEAN_NAME)) {
        BeanDefinition apcDefinition = registry.getBeanDefinition(AUTO_PROXY_CREATOR_BEAN_NAME);
        if (!cls.getName().equals(apcDefinition.getBeanClassName())) {
            int currentPriority = findPriorityForClass(apcDefinition.getBeanClassName());
            int requiredPriority = findPriorityForClass(cls);
            if (currentPriority < requiredPriority) {
                apcDefinition.setBeanClassName(cls.getName());
            }
        }
        return null;
    }
	
    // 封装成 bean 定义信息，用与后期注册到 IoC 容器中。
    RootBeanDefinition beanDefinition = new RootBeanDefinition(cls);
    beanDefinition.setSource(source);
    beanDefinition.getPropertyValues().add("order", Ordered.HIGHEST_PRECEDENCE);
    beanDefinition.setRole(BeanDefinition.ROLE_INFRASTRUCTURE);
    // AUTO_PROXY_CREATOR_BEAN_NAME = internalAutoProxyCreator
    registry.registerBeanDefinition(AUTO_PROXY_CREATOR_BEAN_NAME, beanDefinition);
    return beanDefinition;
}
```

后面就是看 AnnotationAwareAspectJAutoProxyCreator 如何注册到容器，如何发挥作用了。

### AnnotationAwareAspectJAutoProxyCreator的创建

根据前面的分析可以知道 AnnotationAwareAspectJAutoProxyCreator 是个后置处理器，我们需要研究的是它是如何被注册到 IoC 容器的，它后置处理器的相关方法又是如何被调用的。

#### 分析思路

1️⃣查看它的继承关系，可以发现它确实是实现了后置处理器接口，并且对后置处理器接口扩充了一些方法，进行了增强。可以逐步 Debug 看原始的后置处理器方法的作用和增强的方法的作用。

<div align="center"><img src="img/image-20230109191827846.png"></div>

2️⃣查看 AnnotationAwareAspectJAutoProxyCreator 的类继承图可以实现，它实现了 InstantiationAwareBeanPostProcessor 接口，而 InstantiationAwareBeanPostProcessor 也是一个 BeanPostProcessor。它可以拦截 spring 的 Bean 初始化 (Initialization) 前后和实例化 (Initialization) 前后。它还实现了 BeanFactoryAware 接口，用来自动装配 BeanFactory（这个的作用是什么呢？做了那些工作？）。

3️⃣我们关注的是后置处理器相关的方法。自底向上看那些方法和后置处理器、BeanFactory 相关。

- AbstractAutoProxyCreator#setBeanFactory
- AbstractAutoProxyCreator#postProcessAfterInitialization
- AbstractAutoProxyCreator#postProcessAfterInitialization
- AbstractAutoProxyCreator#postProcessBeforeInstantiation
- AbstractAdvisorAutoProxyCreator#setBeanFactory
- AbstractAdvisorAutoProxyCreator#initBeanFactory
- AnnotationAwareAspectJAutoProxyCreator#initBeanFactory

通过猜想，这些方法可能和后置处理器有关的方法上打上断点

#### 源码阅读

我们是希望知道 AOP 的后置处理器如何注册的，而使用它之前就一定要完成注册。因此，我们可以通过调用一个 AOP 增强后的方法，然后查看方法栈的调用流程，逐步分析 Bean 是如何创建处理来的，又是如何使用的。

按照上面的猜想打上断点，逐步 debug 调试。

<b>beanFactory</b>

先是在 setBeanFactory 方法上停了下来，可以通过方法调用栈看看是如何执行到 setBeanFactory 方法的。（我们的最终目的时为了查看后置处理器的创建时机和作用，beanFactory 只是一个切入点！）

<div align="center"><img src="img/image-20230109194930310.png"></div>

流程

1. 传入配置类，创建 IoC 容器

2. 注册配置来，调用 `refresh` 方法刷新容器

    - 容器刷新 refresh 是锁定在了 `registerBeanPostProcessors` 方法上。

3. registerBeanPostProcessors(beanFactory); 方法上的注释是，注册 Bean 的后置处理器，这些后置处理器是用来拦截 Bean 的创建。具体的注册逻辑是在 `PostProcessorRegistrationDelegate#registerBeanPostProcessors`，观察该方法的源码

    - 通过 `beanFactory#getBeanNamesForType` 根据类型拿到所有已经定义了的，需要创建的 BeanPostProcessor 的名称。查看该方法的返回值可以发现，其中一个后置处理器的名称是 org.springframework.aop.config.internalAutoProxyCreator

    - 会优先注册实现了 PriorityOrdered 的后置处理器

    - 再注册实现了 Ordered 的后置处理器

    - 最后再注册普通的后置处理器

    - 如何注册的呢？通过调用 `beanFactory.getBean(ppName, BeanPostProcessor.class)` 尝试获取。但是第一次获取时没有对象，如果发现获取不到就会创建出一个对象。创建的流程如下

        ```mermaid
        graph LR
        getBean-->doGetBean-->getSingleton-->getObject-->lambda$doGetBean-->creteBean-->doCreateBean
        ```

    - 综上所述，注册后置处理器其实就是创建后置处理器对象，保存在容器中。接下来就是看如何创建名为 internalAutoProxyCreator 这个后置处理器。这个创建过程内容较多，暂且把它放到 4 来写。

4. 创建 internalAutoProxyCreator 后处理器的流程。此处，贴出部分代码，并对关键步骤加上注释，省略非关键步骤。

    ```java
    protected Object doCreateBean(String beanName, RootBeanDefinition mbd, @Nullable Object[] args)
        throws BeanCreationException {
    
        // Instantiate the bean.
        BeanWrapper instanceWrapper = null;
        if (mbd.isSingleton()) {
            instanceWrapper = this.factoryBeanInstanceCache.remove(beanName);
        }
        if (instanceWrapper == null) {
            // 创建 bean 实例
            instanceWrapper = createBeanInstance(beanName, mbd, args);
        }
        Object bean = instanceWrapper.getWrappedInstance();
        Class<?> beanType = instanceWrapper.getWrappedClass();
        if (beanType != NullBean.class) {
            mbd.resolvedTargetType = beanType;
        }
    
        // Allow post-processors to modify the merged bean definition.
        synchronized (mbd.postProcessingLock) {
            if (!mbd.postProcessed) {
                try {
                    applyMergedBeanDefinitionPostProcessors(mbd, beanType, beanName);
                }
                catch (Throwable ex) {
                    throw new BeanCreationException(mbd.getResourceDescription(), beanName,
                                                    "Post-processing of merged bean definition failed", ex);
                }
                mbd.postProcessed = true;
            }
        }
    
        // Eagerly cache singletons to be able to resolve circular references
        // even when triggered by lifecycle interfaces like BeanFactoryAware.
        boolean earlySingletonExposure = (mbd.isSingleton() && this.allowCircularReferences &&
                                          isSingletonCurrentlyInCreation(beanName));
        if (earlySingletonExposure) {
            if (logger.isTraceEnabled()) {
                logger.trace("Eagerly caching bean '" + beanName +
                             "' to allow for resolving potential circular references");
            }
            addSingletonFactory(beanName, () -> getEarlyBeanReference(beanName, mbd, bean));
        }
    
        // 创建好了后就在这里初始化
        Object exposedObject = bean;
        try {
            // 给 bean 的各种属性赋值
            populateBean(beanName, mbd, instanceWrapper);
            // 初始化 bean，这步非常关键，因为后置处理器就是在 initializaBean 前后工作的。
            // initializaBean 中调用了方法 invokeAwareMethods、applyBeanPostProcessorsBeforeInitialization、invokeInitMethods、applyBeanPostProcessorsAfterInitialization
            // invokeAwareMethods 判断类是否实现了 Aware 接口，实习了则为其注入 beanFactory
            // applyBeanPostProcessorsBeforeInitialization 调用完后返回一个被包装的 bean，拿到所有的后置处理器，调用后置处理器的 postProcessAfterInitialization 
            // invokeInitMethods 执行初始化方法
            exposedObject = initializeBean(beanName, exposedObject, mbd);
        }
        catch (Throwable ex) {
            if (ex instanceof BeanCreationException && beanName.equals(((BeanCreationException) ex).getBeanName())) {
                throw (BeanCreationException) ex;
            }
            else {
                throw new BeanCreationException(
                    mbd.getResourceDescription(), beanName, "Initialization of bean failed", ex);
            }
        }
    
        if (earlySingletonExposure) {
            Object earlySingletonReference = getSingleton(beanName, false);
            if (earlySingletonReference != null) {
                if (exposedObject == bean) {
                    exposedObject = earlySingletonReference;
                }
                else if (!this.allowRawInjectionDespiteWrapping && hasDependentBean(beanName)) {
                    String[] dependentBeans = getDependentBeans(beanName);
                    Set<String> actualDependentBeans = new LinkedHashSet<>(dependentBeans.length);
                    for (String dependentBean : dependentBeans) {
                        if (!removeSingletonIfCreatedForTypeCheckOnly(dependentBean)) {
                            actualDependentBeans.add(dependentBean);
                        }
                    }
                    if (!actualDependentBeans.isEmpty()) {
                        throw new BeanCurrentlyInCreationException(beanName,
                                                                   "Bean with name '" + beanName + "' has been injected into other beans [" +
                                                                   StringUtils.collectionToCommaDelimitedString(actualDependentBeans) +
                                                                   "] in its raw version as part of a circular reference, but has eventually been " +
                                                                   "wrapped. This means that said other beans do not use the final version of the " +
                                                                   "bean. This is often the result of over-eager type matching - consider using " +
                                                                   "'getBeanNamesForType' with the 'allowEagerInit' flag turned off, for example.");
                    }
                }
            }
        }
    
        // Register bean as disposable.
        try {
            registerDisposableBeanIfNecessary(beanName, bean, mbd);
        }
        catch (BeanDefinitionValidationException ex) {
            throw new BeanCreationException(
                mbd.getResourceDescription(), beanName, "Invalid destruction signature", ex);
        }
    
        return exposedObject;
    }
    ```

    然后误打误撞知道了 `setBeanFactory` 如何执行的了。那我们就顺势看看 setBeanFactory 如何执行的。

5. 阅读 setBeanFactory 的流程，注意是 AnnotationAwareAspectJAutoProxyCreator#initBeanFactory

    ```mermaid
    graph LR
    setBeanFactory-->initBeanFactory
    ```

    ```java
    protected void initBeanFactory(ConfigurableListableBeanFactory beanFactory) {
        super.initBeanFactory(beanFactory);
        if (this.aspectJAdvisorFactory == null) {
            // 反射的 AspectJ 通知工厂
            this.aspectJAdvisorFactory = new ReflectiveAspectJAdvisorFactory(beanFactory);
        }
        // 通知构建器的适配器
        this.aspectJAdvisorsBuilder =
            new BeanFactoryAspectJAdvisorsBuilderAdapter(beanFactory, this.aspectJAdvisorFactory);
    }
    ```

6. 至此，AnnotationAwareAspectJAutoProxyCreator 创建成功。后面的流程就是把 BeanPostProcessor 注册到 BeanFactory 中。

    - `beanFactory.addBeanPostProcessor(postProcessor)`

<b>注意</b>

BeanPostProcessor 接口定义的方法是

- postProcessBefore<b>Initialization()</b>

- postProcessAfter<b>Initialization()</b>

InstantiationAwareBeanPostProcessor 接口继承了 BeanPostProcessor，另外又增加了两个方法

- postProcessBefore<b>Instantiation()</b>

- postProcessAfter<b>Instantiation()</b>

### AnnotationAwareAspectJAutoProxyCreator的执行

AnnotationAwareAspectJAutoProxyCreator 的 postProcessBeforeInstantiation (bean 实例化前) 会通过调用isInfrastructureClass(beanClass) 来判断被拦截的类是否是基础类型的 Advice、PointCut、Advisor、AopInfrastructureBean，或者是否是切面（@Aspect），若是则放入 adviseBean 集合。

#### 分析思路

AnnotationAwareAspectJAutoProxyCreator 是一个后置处理器，所以我们需要关注的是后置处理器相关的方法。

- AbstractAutoProxyCreator#postProcessAfterInitialization
- AbstractAutoProxyCreator#postProcessAfterInitialization
- AbstractAutoProxyCreator#postProcessBeforeInstantiation

#### 源码阅读

在上述几个方法上打上断点，调试运行，代码停在了 `AbstractAutoProxyCreator#postProcessBeforeInstantiation`

注意：这个方法和 `BeanPostProcessor#postProcessBeforeInitialization` 方法名不一样。XXInstantiation 方法是接口 InstantiationAwareBeanPostProcessor 的方法。

为什么会停在 `AbstractAutoProxyCreator#postProcessBeforeInstantiation` 这里呢？我们观察下调用栈可以看到方法的调用流程。

<div align="center"><img src="img/image-20230109220525973.png"></div>

```mermaid
graph LR
创建IoC容器-->refresh-->finishBeanFactoryInitialization-->preInstantiateSingletons
```

finishBeanFactoryInitialization 是用于初始化剩下的单实例 bean，即创建剩下的单实例 bean（BeanPostProcssor 在单实例 bean 之前创建）。

- preInstantiateSingletons -- 遍历获取容器中所有的 bean，依次创建对象。

    ```mermaid
    graph LR
    getBean-->doGetBean-->getSingleton-->createBean-->resolveBeforeInstantiation-->|省略其他|postProcessBeforeInstantiation
    ```

- 创建 bean

    - 先从缓存中获取当前 bean，如果能获取到，说明 bean 是之前被创建过，直接使用，否则就创建。只要被创建好的 Bean 都会被缓存起来。

    - createBean 创建 Bean

        - 拿到要创建的 bean 的定义信息

        - resolveBeforeInstantiation，解析 BeforeInstantiation，给后置处理器一个机会，来返回代理对象来替代 bean 的实例。如果后置处理器没有创建代理对象，则走 doCreateBean。调用 doCreateBean，后面的流程就和 AnnotationAwareAspectJAutoProxyCreator 里说的一样了。

            ```java
            protected Object resolveBeforeInstantiation(String beanName, RootBeanDefinition mbd) {
                Object bean = null;
                if (!Boolean.FALSE.equals(mbd.beforeInstantiationResolved)) {
                    // Make sure bean class is actually resolved at this point.
                    if (!mbd.isSynthetic() && hasInstantiationAwareBeanPostProcessors()) {
                        Class<?> targetType = determineTargetType(beanName, mbd);
                        if (targetType != null) {
                            // 这个不就是 AOP 后置处理器里的某个方法吗。注意这个方法不是
                            // BeanPostProcessor 里的方法。InstantiationAwareBeanPostProcessor 
                            // 的方法是在创建 Bean 实例之前先尝试用后置处理器返回对象的。
                            bean = applyBeanPostProcessorsBeforeInstantiation(targetType, beanName);
                            if (bean != null) {
                                bean = applyBeanPostProcessorsAfterInitialization(bean, beanName);
                            }
                        }
                    }
                    mbd.beforeInstantiationResolved = (bean != null);
                }
                return bean;
            }
            
            protected Object applyBeanPostProcessorsBeforeInstantiation(Class<?> beanClass, String beanName) {
                for (InstantiationAwareBeanPostProcessor bp : getBeanPostProcessorCache().instantiationAware) {
                    Object result = bp.postProcessBeforeInstantiation(beanClass, beanName);
                    if (result != null) {
                        return result;
                    }
                }
                return null;
            }
            ```

        - 到这里我们就知道 AOP 的后置处理器的执行时机了。

### 创建AOP代理

#### 分析思路

这里我们关注的是如何创建出的 AOP 代理。需要关注的代码是

- AbstractAutoProxyCreator#postProcessBeforeInstantiation

在关注的代码上打上断点。因为我们关注的是如何创建出 AOP 代理，因此我们也在被代理对象上打上断点。

回顾下 AnnotationAwareAspectJAutoProxyCreator#InstantiationAwareBeanPostProcessor 的作用。在创建一个 bean 之前，调用 postProcessBeforeInstantiation 方法，尝试创建一个代理对象。

#### 源码阅读

这里我们关心的是 MathCalculator 和 LogAspects 的创建。在 postProcessBeforeInstantiation 方法上打上断点，并阅读下它的代码。

1️⃣每一个 bean 在创建之前，会调用 postProcessBeforeInstantiation 方法。此处我们先关心 MathCalculator 的创建。

```java
public Object postProcessBeforeInstantiation(Class<?> beanClass, String beanName) {
    Object cacheKey = getCacheKey(beanClass, beanName);
    // MathCalculator 第一次走到这里时不会被包含到 advisedBeans 中,因为它并没有被增强。
    // 当 MathCalculator 被增强后就会包含在 advisedBeans 里。
    if (!StringUtils.hasLength(beanName) || !this.targetSourcedBeans.contains(beanName)) {
        if (this.advisedBeans.containsKey(cacheKey)) {
            return null;
        }
        // 是否是基础类 || 是否需要跳过
        if (isInfrastructureClass(beanClass) || shouldSkip(beanClass, beanName)) {
            this.advisedBeans.put(cacheKey, Boolean.FALSE);
            return null;
        }
    }

    // Create proxy here if we have a custom TargetSource.
    // Suppresses unnecessary default instantiation of the target bean:
    // The TargetSource will handle target instances in a custom fashion.
    TargetSource targetSource = getCustomTargetSource(beanClass, beanName);
    if (targetSource != null) {
        if (StringUtils.hasLength(beanName)) {
            this.targetSourcedBeans.add(beanName);
        }
        Object[] specificInterceptors = getAdvicesAndAdvisorsForBean(beanClass, beanName, targetSource);
        Object proxy = createProxy(beanClass, beanName, specificInterceptors, targetSource);
        this.proxyTypes.put(cacheKey, proxy.getClass());
        return proxy;
    }

    return null;
}

protected boolean shouldSkip(Class<?> beanClass, String beanName) {
    // TODO: Consider optimization by caching the list of the aspect names
    // 拿到增强器，此处是 @Aspect 里的四个增强器。但是这四个增强器都不是 AspectJPointcutAdvisor
    // 类型的。
    List<Advisor> candidateAdvisors = findCandidateAdvisors();
    for (Advisor advisor : candidateAdvisors) {
        if (advisor instanceof AspectJPointcutAdvisor &&
            ((AspectJPointcutAdvisor) advisor).getAspectName().equals(beanName)) {
            return true;
        }
    }
    return super.shouldSkip(beanClass, beanName);
}
```

- 判断当前 bean 是否在 advisedBeans 中，advisedBeans 中保存了所有<b>已经增强的 bean。</b>
- 最开始是不包含的，因此还会走当前 bean 是否是基础类型的判断（Advice、Pointcut、Advisor、AopInfrastructureBean 或者是否是切面）////，如果是基础类型就加入 advisedBeans
- 是否需要跳过
    - 获取候选的增强器（切面里的通知方法）【List\<Advisor\> candidateAdvisors】每一个封装的通知方法的增强器是 InstantiationModelAwarePointcutAdvisor，判断每一个增强器是否是 AspectJPointcutAdvisor 类型的。
    - 返回值为 false

2️⃣创建完 MathCalculator 对象后会调用 postProcessAfterInitialization 方法

- 拿到 bean 的名字，判断是否需要包装以下。 wrapIfNecessary
    - 找到候选的所有增强器（通知方法）Object[] specificInterceptors
    - 获取到能在当前 bean 使用的增强器
    - 给增强器排序
- 如果当前 bean 需要增强，创建当前 bean 的代理对象。
    - 保存当前 bean 在 advisedBeans 中表示已经增强过了，然后再对对象进行增强
    - 获取所有增强器 `buildAdvisors(beanName, specificInterceptors)`
    - 把增强器（通知方法）保存到 proxyFactory
    - 由 proxyFactory 创建代理对象，采用何种方式创建由 Spring 自动决定
        - JdkDynamicAopProxy  -- 有接口就采用 JDK
        - ObjenesisCglibAopProxy -- 无接口就采用 cglib
- 如果对象无需增强，则 advisedBeans 中保存该对象，并设置值为 FALSE。（`advisedBeans.put(cacheKey, Boolean.TRUE)`）
- 给容器中返回当前组件使用 cglib 增强了的代理对象
- 以后容器中获取的就是这个组件的代理对象，执行目标方法的时候，代理对象就会执行通知方法。

### 获取拦截器链

#### 分析思路

创建 AOP 代理对象后就是思考如何执行方法了。我们在 MathCalculator 的 div 方法上打上断点，看 div 方法是如何执行的。前面打断点的一些方法可以直接跳过了。而拦截器链的功能就是将每一个通知方法包装成方法拦截器，后面就可以利用 MethodInterceptor 机制去调用这些通知方法。

#### 源码阅读

```java
public class Main {

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(DruidConfig.class);
        MathCalculator bean = context.getBean(MathCalculator.class);
        bean.div(1, 10);
        System.out.println(123);
    }
}
```

断点停在 bean#div 方法上。可以看到，这个 bean 是一个代理对象。

<div align="center"><img src="img/image-20230110151108533.png"></div>

容器中保存了组件的代理对象（cglib 增强后的对象），这个对象中保存了详细的信息，如增强器，目标对象。debug 调试中途会停在 `CglibAopProxy#intercept` 方法上。我们来看下拦截的逻辑。

1️⃣CglibAopProxy#intercept 拦截逻辑

```java
public Object intercept(Object proxy, Method method, Object[] args, MethodProxy methodProxy) throws Throwable {
    Object oldProxy = null;
    boolean setProxyContext = false;
    Object target = null;
    TargetSource targetSource = this.advised.getTargetSource();
    try {
        if (this.advised.exposeProxy) {
            // Make invocation available if necessary.
            oldProxy = AopContext.setCurrentProxy(proxy);
            setProxyContext = true;
        }
        // Get as late as possible to minimize the time we "own" the target, in case it comes from a pool...
        target = targetSource.getTarget();
        Class<?> targetClass = (target != null ? target.getClass() : null);
        // 获取拦截器链
        List<Object> chain = this.advised.getInterceptorsAndDynamicInterceptionAdvice(method, targetClass);
        Object retVal;
        // Check whether we only have one InvokerInterceptor: that is,
        // no real advice, but just reflective invocation of the target.
        if (chain.isEmpty() && Modifier.isPublic(method.getModifiers())) {
            Object[] argsToUse = AopProxyUtils.adaptArgumentsIfNecessary(method, args);
            retVal = methodProxy.invoke(target, argsToUse);
        }
        else {
            // 拦截器链不为空时We need to create a method invocation...
            retVal = new CglibMethodInvocation(proxy, target, method, args, targetClass, chain, methodProxy).proceed();
        }
        retVal = processReturnType(proxy, target, method, retVal);
        return retVal;
    }
    finally {
        if (target != null && !targetSource.isStatic()) {
            targetSource.releaseTarget(target);
        }
        if (setProxyContext) {
            // Restore old proxy.
            AopContext.setCurrentProxy(oldProxy);
        }
    }
}
```

- 根据 ProxyFactory 对象获取将要执行的目标方法的拦截器链
- 如果没有拦截器链，直接执行目标方法。
- 如果有拦截器链，把需要执行的目标对象，目标方法，拦截器链等信息传入创建一个 `CglibMethodInvocation` 对象，并调用 `Object retVal=mi.proceed()`

2️⃣根据上面的分析，我们需要看的是拦截器链这个方法 `getInterceptorsAndDynamicInterceptionAdvice`

```java
public List<Object> getInterceptorsAndDynamicInterceptionAdvice(Method method, 
                                                                @Nullable Class<?> targetClass) {
    MethodCacheKey cacheKey = new MethodCacheKey(method);
    List<Object> cached = this.methodCache.get(cacheKey);
    if (cached == null) {
        // 获取目标方法的拦截器链
        cached = this.advisorChainFactory.getInterceptorsAndDynamicInterceptionAdvice(
            this, method, targetClass);
        this.methodCache.put(cacheKey, cached);
    }
    return cached;
}
```

重点关注 `getInterceptorsAndDynamicInterceptionAdvice` 方法。

- List\<Object\> interceptorList 保存拦截器
- 遍历所有的增强器，将其转为 Interceptor --> `registry.getInterceptors(advisor)`
- getInterceptors 会将增强器转为要用的 MethodIntercptor，如果是 MethodIntercptor 则直接加入集合，不是则先做个转换再加入集合。转换完成后返回 MethodIntercptor 类型的数组。

```java
@Override
public List<Object> getInterceptorsAndDynamicInterceptionAdvice(
    Advised config, Method method, @Nullable Class<?> targetClass) {

    AdvisorAdapterRegistry registry = GlobalAdvisorAdapterRegistry.getInstance();
    Advisor[] advisors = config.getAdvisors();
    // 创建 list 保存所有拦截器
    List<Object> interceptorList = new ArrayList<>(advisors.length);
    Class<?> actualClass = (targetClass != null ? targetClass : method.getDeclaringClass());
    Boolean hasIntroductions = null;
	// 遍历所有的增强器，把这些增强器封装成 Interceptor，registry.getInterceptors(advisor);
    //		getInterceptors 内部会做一些判断和转换，如果不是 MethodInterceptor 则做个转换再加进来
    for (Advisor advisor : advisors) {
        if (advisor instanceof PointcutAdvisor) {
            // Add it conditionally.
            PointcutAdvisor pointcutAdvisor = (PointcutAdvisor) advisor;
            if (config.isPreFiltered() || pointcutAdvisor.getPointcut().getClassFilter().matches(actualClass)) {
                MethodMatcher mm = pointcutAdvisor.getPointcut().getMethodMatcher();
                boolean match;
                if (mm instanceof IntroductionAwareMethodMatcher) {
                    if (hasIntroductions == null) {
                        hasIntroductions = hasMatchingIntroductions(advisors, actualClass);
                    }
                    match = ((IntroductionAwareMethodMatcher) mm).matches(method, actualClass, hasIntroductions);
                }
                else {
                    match = mm.matches(method, actualClass);
                }
                if (match) {
                    MethodInterceptor[] interceptors = registry.getInterceptors(advisor);
                    if (mm.isRuntime()) {
                        // Creating a new object instance in the getInterceptors() method
                        // isn't a problem as we normally cache created chains.
                        for (MethodInterceptor interceptor : interceptors) {
                            interceptorList.add(new InterceptorAndDynamicMethodMatcher(interceptor, mm));
                        }
                    }
                    else {
                        interceptorList.addAll(Arrays.asList(interceptors));
                    }
                }
            }
        }
        else if (advisor instanceof IntroductionAdvisor) {
            IntroductionAdvisor ia = (IntroductionAdvisor) advisor;
            if (config.isPreFiltered() || ia.getClassFilter().matches(actualClass)) {
                Interceptor[] interceptors = registry.getInterceptors(advisor);
                interceptorList.addAll(Arrays.asList(interceptors));
            }
        }
        else {
            Interceptor[] interceptors = registry.getInterceptors(advisor);
            interceptorList.addAll(Arrays.asList(interceptors));
        }
    }

    return interceptorList;
}
```



### 链式调用通知方法

#### 分析思路

处理完拦截器链后就是根据有无拦截器选择如何执行方法。

- 如果拦截器链长度为 0，则直接调用 `methodProxy.invoke(target, argsToUse)` 方法。
- 如果拦截器链长度不为 0，则创建一个 CglibMethodInvocation 对象，用该对象来执行 proceed() 方法。

因此，我们重点关注 CglibMethodInvocation#proceed() 方法的执行。

#### 源码阅读

在 CglibMethodInvocation#proceed() 方法上打上断点。

```java
public Object proceed() throws Throwable {
    // We start with an index of -1 and increment early.
    // this.currentInterceptorIndex 为 -1
    // 如果拦截器数量-1 = -1 说明没有拦截器，则直接用反射执行方法。
    if (this.currentInterceptorIndex == this.interceptorsAndDynamicMethodMatchers.size() - 1) {
        return invokeJoinpoint();
    }
	// 获取第 0 个拦截器，ExposeInvocationInterceptor。会直接执行
    // return ((MethodInterceptor) interceptorOrInterceptionAdvice).invoke(this);
    // 注意！它传入了 this，当前对象，然后再看看 invoke 是如何执行的！
    Object interceptorOrInterceptionAdvice =
        this.interceptorsAndDynamicMethodMatchers.get(++this.currentInterceptorIndex);
    if (interceptorOrInterceptionAdvice instanceof InterceptorAndDynamicMethodMatcher) {
        // Evaluate dynamic method matcher here: static part will already have
        // been evaluated and found to match.
        InterceptorAndDynamicMethodMatcher dm =
            (InterceptorAndDynamicMethodMatcher) interceptorOrInterceptionAdvice;
        Class<?> targetClass = (this.targetClass != null ? this.targetClass : this.method.getDeclaringClass());
        if (dm.methodMatcher.matches(this.method, targetClass, this.arguments)) {
            return dm.interceptor.invoke(this);
        }
        else {
            // Dynamic matching failed.
            // Skip this interceptor and invoke the next in the chain.
            return proceed();
        }
    }
    else {
        // It's an interceptor, so we just invoke it: The pointcut will have
        // been evaluated statically before this object was constructed.
        return ((MethodInterceptor) interceptorOrInterceptionAdvice).invoke(this);
    }
}
```

invoke 的执行，又执行了 proceed()，有调用了 proceed()，此时 proceed() 方法中的 `this.currentInterceptorIndex` 是 1。

```java
public Object invoke(MethodInvocation mi) throws Throwable {
    MethodInvocation oldInvocation = invocation.get();
    invocation.set(mi);
    try {
        return mi.proceed();
    }
    finally {
        invocation.set(oldInvocation);
    }
}
```

逐步 debug 会发现，会一个一个调用通知的 invoke，（调用 MethodBeforeAdviceInterceptor#invoke 方法，AfterReturningAdviceInterceptor#invoke 方法 etc...）

注意，MethodBeforeAdviceInterceptor#invoke 方法是先执行 Advice 的方法，再调用 proceed()，而其他 AdviceInterceptor 则是先调用 proceed() 再执行 Advice 方法，这样就确保了方法本身在 beforeAdvice 后被调用，然后再调用其他 Advice 了。

总而言之：链式获取每一个拦截器，拦截器执行 invoke 方法，每一个拦截器等待下一个拦截器执行完成返回以后再来执行；拦截器链的机制，保证通知方法与目标方法的执行顺序。

### 总结

- @EnableAspectJAutoProxy 开启 AOP 功能
- @EnableAspectJAutoProxy 会给容器注册一个组件 AnnotationAwareAspectJAutoProxyCreator。
- AnnotationAwareAspectJAutoProxyCreator 是一个后置处理器
- 容器的创建流程
    - registerBeanPostProcessors -- 注册后置处理器，会创建 AnnotationAwareAspectJAutoProxyCreator 对象
    - finishBeanFactoryInitialization -- 初始化剩下的单实例 bean
        - 创建业务逻辑组件和切面组件
        - AnnotationAwareAspectJAutoProxyCreator 拦截组件的创建过程
        - 组件创建完之后，调用 wrapIfNecessary 判断组件是否需要增强；
            - 是：切面的通知方法包装成增强器（Advisor）；给业务逻辑组件创建一个代理对象（cglib）
- 执行目标方法
    - 代理对象执行目标方法
    - CglibAopProxy.intercept()
        - 得到目标方法的拦截器链（增强器包装成拦截器，MethodInterceptor）
        - 利用拦截器的链式机制，依次进入每一个拦截器进行执行
        - 效果：前置通知-->目标方法-->返回通知-->后置通知

# 事务控制

Spring 事务管理有 3 个核心接口

1️⃣PlatformTransactionManager：Spring 提供的平台事务管理器，主要用于管理事务。

2️⃣TransactionDefinition：定义了事务规则，并提供了获取事务相关信息的方法。

3️⃣TransactionStatus：描述了某一时间点上事务的状态信息。

## 事务管理核心接口

### PlatformTransactionManager

该接口中提供了 3 个事务操作的方法，getTRansaction 获取事务状态信息；commit 提交事务信息；rollback 回滚事务。

PlatformTransactionManager 的几个实现类如下

```txt
org.springframework.jdbc.datasource.DataSourceTransactionManager
    ：用于配置JDBC数据源的事务管理器
org.springframework.orm.hibernate4.HibernateTransactionManager
    ：用于配置Hibernate的事务管理器
org.springframework.transaction.jta.JtaTransactionManager
    ：用于配置全局事务管理器
```

当底层采用不同的持久层技术时，系统需使用不同的 PlatformTransactionManager实现类。

### TransactionDefinition

接口中定义了如下方法

```java
public interface TransactionDefinition {
    
    // 获取事务的传播行为
	default int getPropagationBehavior() {
		return PROPAGATION_REQUIRED;
	}

    // 获取事务的隔离级别
	default int getIsolationLevel() {
		return ISOLATION_DEFAULT;
	}

    // 获取事务的超时时间
	default int getTimeout() {
		return TIMEOUT_DEFAULT;
	}

    // 获取事务是否只读
	default boolean isReadOnly() {
		return false;
	}

	// 获取事务对象名称
	default String getName() {
		return null;
	}

	static TransactionDefinition withDefaults() {
		return StaticTransactionDefinition.INSTANCE;
	}
}
```

上述方法中，事务的传播行为是指在同一个方法中，不同操作前后所使用的事务。传播行为有很多种

```java
int PROPAGATION_REQUIRED = 0;

int PROPAGATION_SUPPORTS = 1;

int PROPAGATION_MANDATORY = 2;

int PROPAGATION_REQUIRES_NEW = 3;

int PROPAGATION_NOT_SUPPORTED = 4;

int PROPAGATION_NEVER = 5;

int PROPAGATION_NESTED = 6;

int ISOLATION_DEFAULT = -1;

int ISOLATION_READ_UNCOMMITTED = 1;  // same as java.sql.Connection.TRANSACTION_READ_UNCOMMITTED;

int ISOLATION_READ_COMMITTED = 2;  // same as java.sql.Connection.TRANSACTION_READ_COMMITTED;

int ISOLATION_REPEATABLE_READ = 4;  // same as java.sql.Connection.TRANSACTION_REPEATABLE_READ;

int ISOLATION_SERIALIZABLE = 8;  // same as java.sql.Connection.TRANSACTION_SERIALIZABLE;
```

| 属性名称                            | 值   | 描述                                                         |
| ----------------------------------- | ---- | ------------------------------------------------------------ |
| <b>PROPAGATION_REQUIRED（默认）</b> | 0    | 当前方法在运行的时候，如果没有事务，则会创建一个新的事务；如果有事务则直接用。 |
| PROPAGATION_SUPPORTS                | 1    | 当前方法在运行的时候，有事务就用，没有就不用，不会创建新的事务 |
| PROPAGATION_MANDATORY               | 2    | 如果当前方法没有事务会抛出异常，                             |
| PROPAGATION_REQUIRES_NEW            | 3    | 要求方法在一个新的事务环境中执行。如果改方法已经处于一个事务中，则会暂停当前事务，然后启动新的事务执行该方法。 |
| PROPAGATION_NOT_SUPPORTED           | 4    | 不支持事务，总是以非事务的状态执行。如果有事务，会先暂停事务，然后执行该方法。 |
| PROPAGATION_NEVER                   | 5    | 不支持当前事务。如果方法处于事务环境中，会抛出异常。         |
| PROPAGATION_NESTED                  | 6    | 即使当前执行的方法处于事务环境中，依旧会启动一个新的事务，并且方法在嵌套的事务里执行；如果不在事务环境中，也会启动一个新事务，然后执行方法。（有无事务都会创建新事务，然后在新事务中执行方法） |

### TransactionStatus

描述了某一时间点上事务的状态信息。接口中的方法如下：

```java
void flush();  // 刷新事务
boolean hasSavepoint(); // 获取是否存在保存点
boolean isCompleted(); // 获取事务是否完成
boolean isNewTransaction(); // 获取是否是新事务。
boolean isRollbackOnly(); // 获取是否回滚
void setRollbackOnly(); // 设置事务回滚。
```

## 声明式事务

告诉 Spring 哪个方法是事务即可，Spring 会自动进行事务控制。

### 环境搭建

- 导入数据库驱动，Druid 数据源、Spring-JDBC 模块

```xml
<dependency>
    <groupId>com.mysql</groupId>
    <artifactId>mysql-connector-j</artifactId>
    <version>8.0.31</version>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-jdbc</artifactId>
    <version>5.3.3</version>
</dependency>

<dependency>
    <groupId>com.alibaba</groupId>
    <artifactId>druid</artifactId>
    <version>1.2.15</version>
</dependency>
```

- 配置数据源

```java
import com.alibaba.druid.pool.DruidDataSource;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;

@Configuration
public class DruidConfig {
    @Bean
    public DataSource dataSource() {
        DruidDataSource druidDataSource = new DruidDataSource();
        druidDataSource.setUsername("root");
        druidDataSource.setPassword("root");
        druidDataSource.setDriverClassName("com.mysql.cj.jdbc.Driver");
        druidDataSource.setUrl("jdbc:mysql://localhost:3306/mybatis_plus");
        return druidDataSource;
    }

    @Bean
    public JdbcTemplate jdbcTemplate(DataSource dataSource) {
        JdbcTemplate jdbcTemplate = new JdbcTemplate(dataSource);
        return jdbcTemplate;
    }
}
```

- 测试代码，可正常执行 SQL。

```java
import com.review.spring.config.DruidConfig;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.jdbc.core.JdbcTemplate;

public class Main {

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(DruidConfig.class);
        JdbcTemplate template = context.getBean(JdbcTemplate.class);
        String update = "update tb_u set age = 1100 where id = ?";
        int rows = template.update(update, 1);
        System.out.println(rows);
    }
}
```

### 体验声明式事务

在方法上加上注解 @Transactional 即可声明事务，然后在配置类上加上注解 @EnableTransactionManagement 开启事务管理功能（SpringBoot 默认开启），最后在容器中注入事务管理器。

- 编写一个 UserDao 类，与数据库进行交互
- 编写一个 UserService 类，调用 UserDao 的方法操作数据库，其中 UserService 类中有一个方法执行了两条 SQL。

> UserDao 类的代码

```java
@Repository
public class UserDao {
    @Autowired
    JdbcTemplate jdbcTemplate;

    public boolean update(String sql, int... args) {
        return jdbcTemplate.update(sql, args[0], args[1]) > 0;
    }
}
```

> UserService 类的代码

```java
@Service
public class UserService {
    @Autowired
    UserDao userDao;

    @Transactional
    public void testTX() {
        String update = "update tb_u set age = ? where id = ?";
        userDao.update(update, 10, 1);
        int i = 1 / 0;
        userDao.update(update, 10, 2);
    }
}
```

> 修改配置类，添加事务配置

- 开启事务配置 @EnableTransactionManagement
- 注入事务管理器

```java
@Configuration
@ComponentScan(basePackages = "com.review.spring")
@EnableTransactionManagement
public class DruidConfig {
    @Bean
    public DataSource dataSource() {
        DruidDataSource druidDataSource = new DruidDataSource();
        druidDataSource.setUsername("root");
        druidDataSource.setPassword("root");
        druidDataSource.setDriverClassName("com.mysql.cj.jdbc.Driver");
        druidDataSource.setUrl("jdbc:mysql://localhost:3306/mybatis_plus");
        return druidDataSource;
    }

    @Bean
    public JdbcTemplate jdbcTemplate() {
        // 虽然写的是调用 dataSource 方法获取数据源，但是实际上是从 IoC 容器中取的对象
        return new JdbcTemplate(dataSource());
    }

    @Bean
    public PlatformTransactionManager transactionManager() {
        return new DataSourceTransactionManager(dataSource());
    }
}
```

### 事务失效

#### 事务失效的场景

1️⃣抛出检查异常导致事务不能正确回滚

- 原因：Spring 默认只会回滚非检查异常
- 解法：配置 rollbackFor 属性

2️⃣业务方法内自己 try-catche 异常导致事务不能正确回滚

- 原因：事务通知只捉到了目标抛出的异常，才能进行后续的回滚处理，如果目标自己处理掉异常，事务通知无法知悉
- 解法 1：异常原样抛出
- 解法 2：手动设置 TranscactionStatus.setRollbackOnly()

3️⃣aop 切面顺序导致事务不能正常回滚

- 原因：事务切面优先级最低（最后执行），但是如果自定义的切面优先级和他一样，则还是自定义切面在内层，这是若自定义切面自己把异常处理了，没有抛出去，事务切面就捕获不到异常，也就无法回滚事务了。
- 解法：同情况 2；

4️⃣非 public 方法导致的事务失效

- 原因：Spring 为方法创建代理、添加事务通知，前提条件都是该方法是 public 的
- 解法：方法改为 public

5️⃣父子容器导致的事务失效

- 原因：子容器扫描范围过大，把未加事务配置的 service 扫描进来，子容器查询 bean 的时候查询到的是自己容器中未加事务配置的 bean，而非父容器中加了事务配置的 bean。
- 解法 1：各扫各的，不要图方便
- 解法 2：不使用父子容器，所有 bean 放在同一容器

6️⃣调用本类方法导致传播行为失效

- 原因：本类方法调用不经过代理，因此无法增强
- 解法 1：依赖注入自己（代理）来调用
- 解法 2：通过 AopContext 拿到代理对象来调用
- 解法 3：通过 CTW，LTW 来实现功能增强

7️⃣@Transactional 没有保证原子性行为

- 原因：事务的原子性仅涵盖 insert、update、delete、select...for update 语句，select 方法并不阻塞。

8️⃣@Transactional 方法导致的 synchronized 失效

- 原因：sync 保证的只是目标方法的原子性，环绕目标方法的还有 commit 等操作，没有为 commit（提交事务的方法）加上锁。
- 解法 1：加大锁的范围，覆盖到 commit，如，将范围扩大到代理方法的调用
- 解法 2：使用 select...for update 替换 select，为 select 操作加锁。

#### 代码示例

[SpringBoot事务失效场景、事务正确使用姿势_林邵晨的博客-CSDN博客_springboot 事务 应用场景](https://blog.csdn.net/qq_54429571/article/details/126814655)

> <b>抛出检查异常导致事务失效</b>

Spring 默认只会回滚非检查异常，发生检查异常时不会回滚。

```java
@Transactional
public void testTX() throws FileNotFoundException {
    String update = "update tb_u set age = ? where id = ?";
    userDao.update(update, 20, 1);
    new FileInputStream("xxx");
    userDao.update(update, 20, 2);
}
```

解决办法，配置 rollbackFor 属性，可以配置成最大的 Exception（此处配置的 FileNotFoundException），这样不管发生什么检查异常都会进行回滚。

```java
@Transactional(rollbackFor = FileNotFoundException.class)
public void testTX() throws FileNotFoundException {
    String update = "update tb_u set age = ? where id = ?";
    userDao.update(update, 30, 1);
    new FileInputStream("xxx");
    userDao.update(update, 30, 2);
}
```

><b>业务方法内自己 try-catche 异常导致事务不能正确回滚</b>

事务内部捕捉了异常，导致事务管理器无法知晓是否发生异常，事务失效

```java
@Transactional
public void testTX() {
    String update = "update tb_u set age = ? where id = ?";
    try {
        userDao.update(update, 10, 1);
        int i = 1 / 0;
        userDao.update(update, 10, 2);
    } catch (Exception e) {
        System.out.println(e.getMessage());
    }
}
```

不 try-catch 或异常原样抛出。

```java
@Transactional()
public void testTX() {
    String update = "update tb_u set age = ? where id = ?";
    try {
        userDao.update(update, 30, 1);
        int i = 1 / 0;
        userDao.update(update, 30, 2);
    } catch (Exception e) {
        throw new RuntimeException(e);
    }
}
```

手动设置 TransactionStatus.setRollbackOnly()，即在 catch 块添加 TransactionInterceptor.currentTransactionStatus().setRollbackOnly();

```java
@Transactional()
public void testTX() {
    String update = "update tb_u set age = ? where id = ?";
    try {
        userDao.update(update, 30, 1);
        int i = 1 / 0;
        userDao.update(update, 30, 2);
    } catch (Exception e) {
        e.printStackTrace();
        TransactionInterceptor.currentTransactionStatus().setRollbackOnly();
    }
}
```

> <b>aop 切面顺序导致事务不能正常回滚</b>

事务切面优先级最低（最后执行），因此如果有其他切面类捕捉了异常，没有向上抛出而是自行处理了，这样事务切面就捕获不到异常，也就无法回滚事务了。

```java
// 如下面这个切面类，自行处理了异常，没有向上抛出
@Aspect
public void MyAspect{
    public Object around(ProceedingJoinPoint pjo) throws Throwable{
        try{
            return pjp.proceed();
        }catch(Throwable e){
            e.printStackTrace();
            return null;
        }
    }
}
```

在 catch 块添加 `throw new RuntimeException(e);` 或手动设置 TransactionStatus.setRollbackOnly()；也可以调整切面顺序，在 MyAspect 上添加 `@Order(Ordered.LOWEST_PRECEDENCE-1)`（不推荐）

> <b>非 public 方法导致事务失效</b>

```java
@Transactional()
void testTX() {
    String update = "update tb_u set age = ? where id = ?";
    userDao.update(update, 30, 1);
    int i = 1 / 0;
    userDao.update(update, 30, 2);
}
```

改为 public 方法或添加配置，但是添加配置的方式不推荐，因此不做记录。

> <b>类内方法调用导致事务失效</b>

Spring 的事务是通过代理类来实现的，而只有加了 @Transactional 注解的方法才会用代理类去调用该方法。下面这种写法会导致事务失效。

```java
@Service
public class UserService {
    @Autowired
    UserDao userDao;

    public void testTX() {
        synchronized (this) {
            testAopContext();
        }
    }

    @Transactional
    public void testAopContext() {
        String update = "update tb_u set age = ? where id = ?";
        userDao.update(update, 30, 1);
        int i = 1 / 0;
        userDao.update(update, 30, 2);
    }
}
```

因为 testAopContext() 是在没有加 @Transactional 注解的方法 testTX() 内调用的，该方法的调用不会走代理，相当于 this.testAopContext()，没有走代理对象，因此事务会失效。

解决办法是，获取代理对象，使用代理对象调用方法。可以自己注入自己，也可以用 AopContext 获取当前代理对象。

```java
@Service
public class UserService {
    @Autowired
    UserDao userDao;

    public void testTX() {
        synchronized (this) {
            UserService o = (UserService) AopContext.currentProxy();
            o.testAopContext();
        }
    }

    @Transactional
    public void testAopContext() {
        String update = "update tb_u set age = ? where id = ?";
        userDao.update(update, 300, 1);
        int i = 1 / 0;
        userDao.update(update, 300, 2);
    }
}
```

## 源码分析

### 分析思路

和之前分析 @EnableAspectJAutoProxy 类似。为了简单起见，去除多余的 @EnableXX 注解，只开启事务必备的内容。因此，采用的配置类如下

```java
@Configuration
@ComponentScan(basePackages = "com.review.spring.service")
@EnableTransactionManagement
public class DruidConfig {
    @Bean
    public DataSource dataSource() {
        DruidDataSource druidDataSource = new DruidDataSource();
        druidDataSource.setUsername("root");
        druidDataSource.setPassword("root");
        druidDataSource.setDriverClassName("com.mysql.cj.jdbc.Driver");
        druidDataSource.setUrl("jdbc:mysql://localhost:3306/mybatis_plus");
        return druidDataSource;
    }

    @Bean
    public JdbcTemplate jdbcTemplate() {
        // 虽然写的是调用 dataSource 方法获取数据源，但是实际上是从 IoC 容器中取的对象
        return new JdbcTemplate(dataSource());
    }

    @Bean
    public PlatformTransactionManager transactionManager() {
        return new DataSourceTransactionManager(dataSource());
    }
}
```

简单看下 @EnableAspectJAutoProxy 这个注解，可以发现它用 @Import 注解导入了一个配置类 `TransactionManagementConfigurationSelector`。在这个配置类中会导入两个组件

- AutoProxyRegistrar，用于注册 BeanDefinitions，会给容器注册一个 InfrastructureAdvisorAutoProxyCreator，该对象对应的 beanName 为 org.springframework.aop.config.internalAutoProxyCreator，也是个后置处理器。
- ProxyTransactionManagementConfiguration，用于注册启用基于代理的事务管理所需的 bean。如事务拦截器（TransactionInterceptor），用于拦截方法的执行，在方法的执行前后会涉及到事务的相关操作。

我们在这些关键的类上打上断点。简单过一遍执行流程，就可以猜到那些是关键类，那些是关键方法。简单 debug 梳理后可以发现，下面这些类非常关键

- AutoProxyRegistrar#registerBeanDefinitions 方法
    - AopConfigUtils#registerOrEscalateApcAsRequired 方法
- ProxyTransactionManagementConfiguration#transactionInterceptor 方法
- TransactionInterceptor#invoke 方法
- TransactionAspectSupport#invokeWithinTransaction 方法

<b>实现原理</b>

1. @EnableTransactionManagement 利用 TransactionManagementConfigurationSelector 给 spring 容器中导入两个组件：AutoProxyRegistrar 和 ProxyTransactionManagementConfiguration
2. AutoProxyRegistrar 给 spring 容器中注册一个 InfrastructureAdvisorAutoProxyCreator，而该类实现了 InstantiationAwareBeanPostProcessor,InstantiationAwareBeanPostProcessor 是一个 BeanPostProcessor。它可以拦截 spring 的 Bean 初始化 (Initialization) 前后和实例化 (Initialization) 前后。利用后置处理器机制在被拦截的 bean 创建以后包装该 bean 并返回一个代理对象代理对象执行方法利用拦截器链进行调用（同 Spring AOP 的原理）
3. ProxyTransactionManagementConfiguration：是一个 spring 的配置类，它为 spring 容器注册了一个 BeanFactoryTransactionAttributeSourceAdvisor，是一个事务事务增强器。它有两个重要的字段：AnnotationTransactionAttributeSource 和 TransactionInterceptor。
    - AnnotationTransactionAttributeSource：用于解析事务注解的相关信息
    - TransactionInterceptor：事务拦截器，在事务方法执行时，都会调用 TransactionInterceptor  的 invoke->invokeWithinTransaction 方法，这里面通过配置的 PlatformTransactionManager 控制着事务的提交和回滚。

### 源码阅读

按照分析思路，对关键的类打上断点。然后分析 AutoProxyRegistrar 和 ProxyTransactionManagementConfiguration 的功能。本质就是利用 AOP 在方法执行前关闭事务自动提交，在方法执行后提交事务/回滚事务。

1️⃣AutoProxyRegistrar

- 给容器注册一个 InfrastructureAdvisorAutoProxyCreator 组件，这个组件是 SmartInstantiationAwareBeanPostProcessor 类型的，也是一个后置处理器。
- InfrastructureAdvisorAutoProxyCreator 利用后置处理器机制，在对象创建以后包装对象，返回一个代理对象（增强器），代理对象执行方法，利用拦截器链进行调用。和 AOP 的逻辑类似。

2️⃣ProxyTransactionManagementConfiguration

- 利用 @Bean 给容器注册各种组件。
- 会给容器中注入事务增强器
    - AnnotationTransactionAttributeSource 解析事务注解
    - TransactionInterceptor 保存了事务的属性信息，事务管理器，本质上是一个 MethodInterceptor，代理对象要执行目标方法时，拦截器就会开始工作。在目标方法执行的时候执行拦截器链，这个拦截器链中只有一个拦截器，就是事务拦截器。
    - 事务拦截器：先获取事务属性，再获取 PlatformTranscationManager，如果实现没有指定，最终会从容器中按照类型获取一个 PlatformTranscationManager。然后执行事务方法。

```java
@Nullable
protected Object invokeWithinTransaction(Method method, @Nullable Class<?> targetClass,
                                         final InvocationCallback invocation) throws Throwable {
	// some code...
    PlatformTransactionManager ptm = asPlatformTransactionManager(tm);
    final String joinpointIdentification = methodIdentification(method, targetClass, txAttr);

    if (txAttr == null || !(ptm instanceof CallbackPreferringPlatformTransactionManager)) {
        // Standard transaction demarcation with getTransaction and commit/rollback calls.
        TransactionInfo txInfo = createTransactionIfNecessary(ptm, txAttr, joinpointIdentification);

        Object retVal;
        try {
			// 对这个方法 debug 发现，它就是执行的 proceed() 方法，逐个执行拦截器链
            // 中的拦截器。此处拦截器只有一个 TransactionInterceptor。
            retVal = invocation.proceedWithInvocation();
        }
        catch (Throwable ex) {
            // 拿到事务管理器，进行回滚。
            completeTransactionAfterThrowing(txInfo, ex);
            throw ex;
        }
        finally {
            cleanupTransactionInfo(txInfo);
        }
		// some code...
    }
	// some code...
}
```





## 其他

### 编程式事务

```java
// 用过滤器控制事务！妙啊！
TransactionFilter{
    try{
        // 获取连接
        // 设置非自动提交
        chain.doFilter();
        // 提交
    }catch(Exception e){
        // 回滚
    }finally{
        // 提交
    }
}
```

事务管理代码的固定模式作为一种横切关注点，可以通过 AOP 方法模块化，进而借助 Spring AOP 框架实现声明式事务管理。

自己要写这个切面还是很麻烦；且这个切面已经有了；（事务切面，事务管理）

### 事务控制

> Spring 支持的事务控制

```java
import java.io.FileInputStream;
import java.io.FileNotFoundException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Isolation;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import com.study.dao.BookDao;

@Service
public class BookService {
	
	@Autowired
	BookDao bookDao;
	
//	@Autowired
//	BookService bookService;
	
	/**
	 * 事务细节：
	 * isolation-Isolation：事务的隔离级别;
	 * 
	 * noRollbackFor-Class[]：哪些异常事务可以不回滚
	 * noRollbackForClassName-String[]（String全类名）:
	 * 
	 * rollbackFor-Class[]:哪些异常事务需要回滚；
	 * rollbackForClassName-String[]:
	 * 
	 * 异常分类：
	 * 		运行时异常（非检查异常）：可以不用处理；默认都回滚；
	 * 		编译时异常（检查异常）：要么try-catch，要么在方法上声明throws
	 * 				默认不回滚；
	 * 
	 * 事务的回滚：默认发生运行时异常都 回滚，发生编译时异常不会回滚；
	 * noRollbackFor:哪些异常事务可以不回滚;（可以让原来默认回滚的异常给他不回滚）
	 * 	noRollbackFor={ArithmeticException.class,NullPointerException.class}
	 * noRollbackForClassName
	 * 
	 * rollbackFor：原本不回滚（原本编译时异常是不回滚的）的异常指定让其回滚；
	 * 
	 * readOnly-boolean：设置事务为只读事务：
	 * 		可以进行事务优化；
	 * 		readOnly=true：加快查询速度；不用管事务那一堆操作了。
	 * 
	 * timeout-int（秒为单位）：超时：事务超出指定执行时长后自动终止并回滚
	 * @throws FileNotFoundException 
	 * 
	 * 
	 * propagation-Propagation：事务的传播行为;
	 * 	传播行为（事务的传播+事务的行为）；
	 * 		如果有多个事务进行嵌套运行，子事务是否要和大事务共用一个事务；
	 * 传播行为:
	 * AService{
	 * 		tx_a(){
	 * 			//a的一些方法
	 * 			tx_b(){
	 * 			}
	 * 			tx_c(){
	 * 			}
	 * 		}
	 * }
	 */
	@Transactional(propagation=Propagation.REQUIRES_NEW)
	public void checkout(String username,String isbn){
		//1、减库存
		bookDao.updateStock(isbn);
		
		int price = bookDao.getPrice(isbn);
//		try {
//			Thread.sleep(3000);
//		} catch (InterruptedException e) {
//			e.printStackTrace();
//		}
		//2、减余额
		bookDao.updateBalance(username, price);
		
		//int i = 10/0;
		//new FileInputStream("D://hahahahha.aa");
	}
	
	@Transactional(propagation=Propagation.REQUIRES_NEW)
	public void updatePrice(String isbn,int price){
		bookDao.updatePrice(isbn, price);
	}
	
	/**
	 * 根据业务的特性；进行调整
	 * isolation=Isolation.READ_UNCOMMITTED:读出脏数据
	 * 
	 * 		READ_COMMITTED；实际上业务逻辑中用的最多的也是这个；
	 * 		REPEATABLEP_READ；
	 * @param isbn
	 * @return
	 */
	@Transactional(readOnly=true)
	public int getPrice(String isbn){
		return bookDao.getPrice(isbn);
	}
	
	@Transactional
	public void mulTx(){
		
		//ioc.getBean("BookSerice");
		checkout("Tom", "ISBN-001");
		
		updatePrice("ISBN-002", 998);
		
		int i=10/0;
	}
}

//===============================
@Service
public class MulService {
	
	@Autowired
	private BookService bookService;
	
	@Transactional
	public void mulTx(){
		//都是可以设置的；
		//传播行为来设置这个事务方法是不是和之前的大事务共享一个事务（使用同一条连接）；
		//REQUIRED  
		bookService.checkout("Tom", "ISBN-001");
		
		//REQUIRED   REQUIRES_NEW
		bookService.updatePrice("ISBN-002", 998);
		
		//int i = 10/0;
	}
}
```

### 事务隔离级别

事务的隔离级别有四种：读未提交、读已提交、可重复的、串行化。

<span style="color:red">数据库事务并发问题有如下三种：</span>

1️⃣<b>脏读：</b>读到了未提交的数据

2️⃣<b>不可重复读：</b>两次读取数据不一样（第一次读到了原来的数据；接下来数据更新了；第二次又读了这个数据，数据不一样了，因为更新了）

3️⃣<b>幻读：</b>多读了，或少读了数据

事务的隔离级别是需要根据业务的特性进行调整

```java
@Transactional(isolation+Isolation.READ_UNCOMMITTED)
```

> 嵌套事务

<div align="center"><img src="img/spring/shiwu.png"></div>

本类方法的嵌套调用是一个事务

# 扩展原理

## BeanFactoryPostProcessor

### 概述

注意与 BeanPostProcessor 进行区分。BeanPostProcessor 是 bean 后置处理器，bean 创建对象初始化前后进行拦截工作的。

BeanFactoryPostProcessor 是 beanFactory 的后置处理器，在 BeanFactory 标准初始化之后调用。根据调用时机的特点，可以在 `beanFactory` 初始化后进行一些操作。

- 在 BeanFactory 标准初始化之后调用；所有的 bean 定义已经保存加载到 beanFactory 中，<b>但是 bean 的实例还未创建。</b>

- BeanFactoryPostProcessor 的源码注释

    ```
    Modify the application context's internal bean factory after its standard
    initialization. All bean definitions will have been loaded, but no beans
    will have been instantiated yet.
    ```

- 在 bean factory 标准初始化后执行。所有的 bean 的定义信息已经加载了，但是 bean 没有初始化！

### 原理

1️⃣IOC 容器创建对象

2️⃣invokeBeanFactoryPostProcessor（beanFactory）

- 如何找到所有的 BeanFactoryPostProcessor 并执行他们的方法？
    - 直接在 BeanFactory 中找到所有类型是 BeanFactoryPostProcessor 的组件，并执行他们的方法
    - 在初始化创建其他组件前面执行

3️⃣学到的一种写法，用接口表示排序规则，获取类时，查看它是否实现了 xxx 接口，以此判断执行顺序。

## BeanDefinitionegistryPostProcessor

对标准 {@link BeanFactoryPostProcessor} SPI 的扩展，允许在常规 BeanFactoryPostProcessor 检测开始之前注册更多的 bean 定义。开发者可以通过该类实现扩展，在类初始之前对 beanDefinition 进行修改以及新增注册。

### 概述

BeanDefinitionegistryPostProcessor 是 BeanFactoryPostProcessor 的子接口

```java
public interface BeanDefinitionRegistryPostProcessor extends BeanFactoryPostProcessor {
	void postProcessBeanstDefinitionRegistry(BeanDefinitionRegistry registry) throws BeansException;
}
```

postProcessBeanstDefinitionRegistry() 在所有 bean 定义信息将要被加载，bean 实例还未创建的时候执行。

<b>先给结论</b>

- BeanDefinitionRegistryPostProcessor() 优于 BeanFactoryPostProcessor 执行。
- 我们可以利用 BeanDefinitionRegistryPostProcessor() 给容器中再额外添加一些组件。
- 可以在如下代码的两个方法中打断点，看看执行流程。

验证代码如下

```java
@Configuration
public class BeanDefinitionRegistryPostProcessorConfig {

    @Bean
    public MyBeanDefinitionRegistryPostProcessor get() {
        return new MyBeanDefinitionRegistryPostProcessor();
    }

    public static void main(String[] args) {
        /**
         * 这个测试流程如下：
            postProcessBeanDefinitionRegistry拥有的类数量为 8
            postProcessBeanDefinitionRegistry又注册了一个bean blue
            此时postProcessBeanDefinitionRegistry拥有的类数量为 9
            postProcessBeanFactory拥有的bean数量 9
         *  这说明了  Registry先执行于Factory
         */
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanDefinitionRegistryPostProcessorConfig.class);
        context.close();
    }
}

class MyBeanDefinitionRegistryPostProcessor implements BeanDefinitionRegistryPostProcessor {

    @Override
    // BeanDefinitionRegistry 中保存了 beanDefinition，以后 BeanFactory 就是按照
    // BeanDefinitionRegistry 里面保存的每一个 bean 定义信息创建 bean 实例
    public void postProcessBeanDefinitionRegistry(BeanDefinitionRegistry registry) throws BeansException {
        System.out.println(String.format("postProcessBeanDefinitionRegistry拥有的类数量为 %d", registry.getBeanDefinitionCount()));
        // 可在这里进行bean的注册
        RootBeanDefinition beanDefinition = new RootBeanDefinition(Blue.class);
        registry.registerBeanDefinition("blue", beanDefinition);
        System.out.println(String.format("postProcessBeanDefinitionRegistry又注册了一个bean %s", "blue"));
        System.out.println(String.format("此时postProcessBeanDefinitionRegistry拥有的类数量为 %d", registry.getBeanDefinitionCount()));

    }

    @Override
    public void postProcessBeanFactory(ConfigurableListableBeanFactory beanFactory) throws BeansException {
        System.out.println(String.format("postProcessBeanFactory拥有的bean数量 %d", beanFactory.getBeanDefinitionCount()));
    }
}

/*
postProcessBeanDefinitionRegistry拥有的类数量为 8
postProcessBeanDefinitionRegistry又注册了一个bean blue
此时postProcessBeanDefinitionRegistry拥有的类数量为 9
postProcessBeanFactory拥有的bean数量 9
*/
```

### 原理

1️⃣IOC 创建对象

2️⃣refresh()-->invokeBeanFactoryPostProcssors(beanFactory)

3️⃣从容器中获取所有的 BeanDefinitionRegistryPostProcessor 组件

- 依次触发所有的 postProessBeanDefinitionRegistry() 方法
- 再来触发 postProcessBeanFactory() 方法【该方法位于 BeanFactoryPostProcessor 类里】

4️⃣再来从容器中找到 BeanFactoryPostProcessor 组件，然后依次触发 postProcessBeanFactory() 方法

为什么他要先于 BeanFactoryPostProcessor 执行呢？为了注册一些 BeanDefinition，做扩展呀。 

## ApplicationListener

### 概述

监听容器中发布的事件。事件驱动模型开发。

- 容器关闭事件

- 容器刷新事件

- 容器开始事件

- 容器停止事件

要想实现事件监听机制，我们需要这样做，写一个类实现如下监听器接口 

public interface ApplicationListener\<E extends ApplicationEvent\> extends EventListener {}

这个接口，它所带的泛型就是我们要监听的事件。即它会监听 ApplicationEvent 及下面的子事件。

然后重写接口中的 onApplicationEvent() 方法即可

<b>容器事件监听步骤</b>

1️⃣写一个监听器来监听某个事件（ApplicationEvent 及其子类）

2️⃣把监听器加入到容器

3️⃣只要容器中有相关事件的发布，我们就能监听到这个事件

- ContextRefreshedEvent：容器刷新完成（所有 bean 都完全创建）会发布这个事件。
- ContextClosedEvent：关闭容器会发布这个事件。
- 我们也可以自定义事件！

4️⃣发布一个事件

```java
package org.example.configuration.ext;

import org.springframework.context.ApplicationEvent;
import org.springframework.context.ApplicationListener;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
// 代码中含义Spring自己定义的一些事件的发布，也有我们自定义事件的发布。
@Configuration
public class ApplicationEventConfig {

    @Bean
    public MyApplicationEvent event() {
        return new MyApplicationEvent();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(ApplicationEventConfig.class);
        // 自定义事件发布
        context.publishEvent(new ApplicationEvent(new String("123")) {
        });
        context.close();
    }

}

// Spring中的事件的发布
class MyApplicationEvent implements ApplicationListener<ApplicationEvent> {
    // 当容器中发布此事件以后，方法触发
    @Override
    public void onApplicationEvent(ApplicationEvent event) {
        System.out.println(String.format("收到事件 %s", event));
    }
}
```

<b>自己发布事件</b>

1️⃣写一个监听器来监听这个事件（ApplicationEvent 及其子类）。

2️⃣把监听器加入到容器。

3️⃣只要容器中有相关事件的发布，我们就能监听到这个事件，比如监听 ApplicationEvent，监听 ContextClosedEvent 事件。

```java

@Configuration
public class MyApplicationEvent {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(MyApplicationEvent.class);
        context.publishEvent(new ApplicationEvent(new String("hello")) {
        });
        context.close();
    }

    @Bean
    public ApplicationListener applicationListener() {
        return new MyApplicationListener();
    }

    static class MyApplicationListener implements ApplicationListener<ApplicationEvent> {

        @Override
        public void onApplicationEvent(ApplicationEvent event) {
            System.out.println("接受到事件=====>" + event);
        }
    }
}
/*
接受到事件=====>org.springframework.context.event.ContextRefreshedEvent[source=org.springframework.context.annotation.AnnotationConfigApplicationContext@5383967b]
接受到事件=====>com.review.spring.ext.MyApplicationEvent$1[source=hello]
接受到事件=====>org.springframework.context.event.ContextClosedEvent[source=org.springframework.context.annotation.AnnotationConfigApplicationContext@5383967b]
*/
```

### 原理

1-12 再学。今天摸鱼去了。

## @EventListener

使用 `@EventListener` 注解监听事件。

标记在方法上，使用 classes 属性声明要监听的事件类型。ApplicationEvent 类型的方法参数可以获得到该事件。

```java
@Service
public class UserService {

    @EventListener(classes = {ApplicationEvent.class})
    public void listener(ApplicationEvent event) {
        System.out.println("得到事件:" + event);
    }
}
```

使用 `EventListenerMethodProcessor` 处理器来解析方法上的 `@EventListener` 注解。

`EventListenerMethodProcessor` 实现了 `SmartInitializingSingleton` 接口。

## SmartInitializingSingleton

在所有单实例 bean 都创建完成之后调用，调用的时机类似 `ContextRefreshedEvent`。

```java
public interface SmartInitializingSingleton {
	void afterSingletonsInstantiated();
}
```

<b>调用过程</b>

1. ioc 创建对象并刷新容器：refresh() 
2. refresh() 调用 `finishBeanFactoryInitialization()`
3. finishBeanFactoryInitialization() 初始化剩下的单实例 bean
    - 遍历所有待创建的单实例 bean，调用 `getBean()` 创建所有的单实例 bean 

1. - 获取所有创建好的单实例 bean，判断是否是 `SmartInitializingSingleton` 类型。
        如果是该类型，就调用其 `afterSingletonsInstantiated()` 方法 

## Spring容器的创建过程

Spring 的 `refresh()` 方法进行容器的创建和刷新，进入 `AbstractApplicationcontext` 类的 `refresh()` 方法中。

```java
public void refresh() throws BeansException, IllegalStateException {
    synchronized (this.startupShutdownMonitor) {
        // 1.进行容器的预处理
        prepareRefresh();

        // 2.获取beanFactory
        ConfigurableListableBeanFactory beanFactory = obtainFreshBeanFactory();

        // 3.进行beanFactory的预处理工作
        prepareBeanFactory(beanFactory);

        try {
            // 4.beanFactory准备工作完成之后的后置处理工作
            postProcessBeanFactory(beanFactory);

            // 5.执行BeanFactoryPostProcessors（包括BeanFactoryPostProcessor 和 BeanDefinitionRegistryPostProcessor）
            invokeBeanFactoryPostProcessors(beanFactory);
	         // 6.注册bean后置处理器，拦截bean的创建过程
            registerBeanPostProcessors(beanFactory);

            // 7.初始化MessageSource，做国际化功能
            initMessageSource();

            // 8.初始化事件派发器
            initApplicationEventMulticaster();

            // 9.初始化其他的一些特殊Bean，默认为空实现，留给子类进行自定义重写
            onRefresh();

            // 10.将所有的ApplicationListener注册进容器中
            registerListeners();

            // 11.初始化所有剩下的单实例bean
            finishBeanFactoryInitialization(beanFactory);

            // Last step: publish corresponding event.
            finishRefresh();
        }

        catch (BeansException ex) {
            if (logger.isWarnEnabled()) {
                logger.warn("Exception encountered during context initialization - " +
                            "cancelling refresh attempt: " + ex);
            }

            // Destroy already created singletons to avoid dangling resources.
            destroyBeans();

            // Reset 'active' flag.
            cancelRefresh(ex);

            // Propagate exception to caller.
            throw ex;
        }

        finally {
            // Reset common introspection caches in Spring's core, since we
            // might not ever need metadata for singleton beans anymore...
            resetCommonCaches();
        }
    }
}
```

# Servlet3.0

## Servlet的运行时插件能力

<b>概述</b>

Shared libraries（共享库） / `runtimes pluggability`（运行时插件能力）

1）`Servlet `容器启动会扫描，当前应用里面每一个 jar 包的 `ServletContainerInitServletContainerIntiializer` 的实现

2）提供 `ServletContainerInitializer` 的实现类；

- 必须绑定在，`META-INF/services/javax.servlet.ServletContainerInitializer `文件中
    - maven 项目中，META-INF/services 这个目录是以 resources 为根目录的。即目录全名为：`resources/META-INF/services`
    - `javax.servlet.ServletContainerInitializer` 是一个没有后缀的文件哦！
- 文件的内容就是 `ServletContainerInitServletContainerIntiializer` 实现类的全类名

<b>总结</b>

容器在启动应用的时候，会扫描当前应用每一个 jar 包里面 `META-INF/services/javax.servlet.ServletContainerInitializer` 指定的实现类，启动并运行这个实现类的方法；传入感兴趣的类型。`SpringMVC` 也是通过这种原理来实现的。

<b>代码示例</b>

maven工厂，`JavaWeb` 项目，工程目录结构如下：

<div align="center"><img src="img/spring/Spring_ann_runtimes_pluggability.png"></div>

```java
import javax.servlet.ServletContainerInitializer;
import javax.servlet.ServletContext;
import javax.servlet.ServletException;
import javax.servlet.annotation.HandlesTypes;
import java.util.Set;

// 容器启动的时候会将@HandlesTypes指定的这个类型下面的子类（实现类）传递过来
// 写一个感兴趣的类型.
@HandlesTypes(value = {Hello.class})
public class MyServletContainerInitializer implements ServletContainerInitializer {
    /**
     * @param c   感兴趣的类型的所有子类型
     * @param ctx 代表当前web应用的ServletContext：一个web应用一个ServletContext
     * 使用 ServletContext注册web组件（Servlet、Filter、Listener）
     * @throws ServletException
     */
    @Override
    public void onStartup(Set<Class<?>> c, ServletContext ctx) throws ServletException {
        System.out.println("感兴趣的类型");
        for (Class<?> clazz : c) {
            // 会输出  class lg.Demo
            System.out.println(clazz);
        }
    }
}

interface Hello {}

class Demo implements Hello {
    public Demo() {
        System.out.println("Demo");
    }
}
```

## 用上述代码注册JavaWeb三大组件

PS：Servlet，Filter，XxxListener 的实现类要是 public 修饰的！！！不然会失败！！

例子：你直接 `class Servlet xxx` 这样注册组件，添加范围路径访问的话，浏览器会显示 `no this function`！

> ServletContainerInitializer 实现类

```java
import javax.servlet.*;
import javax.servlet.annotation.HandlesTypes;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.EnumSet;
import java.util.Set;

// 容器启动的时候会将@HandlesTypes指定的这个类型下面的子类（实现类）传递过来
// 写一个感兴趣的类型.
@HandlesTypes(value = {Hello.class})
public class MyServletContainerInitializer implements ServletContainerInitializer {
    /**
     * @param c   感兴趣的类型的所有子类型
     * @param ctx 代表当前web应用的ServletContext：一个web应用一个ServletContext
     * @throws ServletException
     */
    @Override
    public void onStartup(Set<Class<?>> c, ServletContext ctx) throws ServletException {
        System.out.println("感兴趣的类型");
        for (Class<?> clazz : c) {
            System.out.println(clazz);
        }
        // 注册 Servlet 組件
        ServletRegistration.Dynamic userServlet = ctx.addServlet("userServlet", UserServlet.class);
        // 配置Servlet的映射信息
        userServlet.addMapping("/userServlet");

        ServletRegistration.Dynamic demo = ctx.addServlet("demo", Demos.class);
        demo.addMapping("/demo");
        // 注冊监听器
        ctx.addListener("lg.UserListener");
        FilterRegistration.Dynamic userFilter = ctx.addFilter("userFilter", UserFilter.class);
        // userFilter.addMappingForServletNames(); // 专门拦截xxx Servlet
        userFilter.addMappingForUrlPatterns(EnumSet.of(DispatcherType.REQUEST), true, "/*"); // 按路径拦截
    }
}

interface Hello {}

class Demo implements Hello {
    public Demo() {
        System.out.println("Demo");
    }
}
// 这是演示错误的servlet。正确的需要public 修饰！！
class Demos extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        resp.getWriter().write("!@#");
    }
}
```

> Servlet 实现类

```java
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class UserServlet extends HttpServlet implements Hello {
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        resp.getWriter().write("UserServlet");
    }
}
```

> Filter 实现类

```java
import javax.servlet.*;
import java.io.IOException;

public class UserFilter implements Filter {

    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        System.out.println("UserFilter init");
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) throws IOException, ServletException {
        System.out.println("UserFilter doFilter");
        chain.doFilter(request, response);
    }

    @Override
    public void destroy() {
        System.out.println("UserFilter destroy");
    }
}
```

> ServletContextListener 实现类

```java
import javax.servlet.ServletContextEvent;
import javax.servlet.ServletContextListener;

public class UserListener implements ServletContextListener, Hello {

    @Override
    public void contextInitialized(ServletContextEvent sce) {
        System.out.println("UserListener init");
    }

    @Override
    public void contextDestroyed(ServletContextEvent sce) {
        System.out.println("UserListener destroy");
    }
}
```

# SpringMVC

## 概述

SpringMVC 文件中指定了 SpringServletContainerInitializer

<div align="center"><img src="img/spring/SpringMVC_config.png"></div>

用监听器启动 Spring 的配置（配置 ContextLoaderListener 加载 Spring 的配置启动 Spring 容器）

启动 SpringMVC 的配置（配置 DispatcherServlet 启动 SpringMVC，配好映射）

看一下 SpringServletContainerInitializer 的源码：

```java
@HandlesTypes(WebApplicationInitializer.class) // 对WebApplicationInitializer及其子类感兴趣
public class SpringServletContainerInitializer implements ServletContainerInitializer {

	@Override
	public void onStartup(Set<Class<?>> webAppInitializerClasses, ServletContext servletContext) throws ServletException {

		List<WebApplicationInitializer> initializers = new LinkedList<WebApplicationInitializer>();

		if (webAppInitializerClasses != null) {
            // 拿到感兴趣的类型集合，挨个遍历
			for (Class<?> waiClass : webAppInitializerClasses) {
				// Be defensive: Some servlet containers provide us with invalid classes,
                //【防止抽象类，接口啥的吧】
				// no matter what @HandlesTypes says...
				if (!waiClass.isInterface() && !Modifier.isAbstract(waiClass.getModifiers()) &&
						WebApplicationInitializer.class.isAssignableFrom(waiClass)) {
					try {
						initializers.add((WebApplicationInitializer) waiClass.newInstance());
					}
					catch (Throwable ex) {
						throw new ServletException("Failed to instantiate WebApplicationInitializer class", ex);
					}
				}
			}
		}

		if (initializers.isEmpty()) {
			servletContext.log("No Spring WebApplicationInitializer types detected on classpath");
			return;
		}

		servletContext.log(initializers.size() + " Spring WebApplicationInitializers detected on classpath");
		AnnotationAwareOrderComparator.sort(initializers);
		for (WebApplicationInitializer initializer : initializers) {
			initializer.onStartup(servletContext);
		}
	}
}
```

<b>梳理一下</b>

1️⃣web 容器在启动的时候，会扫描每个 jar 包下的 META-INFO/services/javax.servlet.ServletContainerInitializer

2️⃣加载这个文件指定的类 SpringServletContainerInitializer

3️⃣Spring 应用一启动就会加载感兴趣的 WebAppleicationInitializer 下的所有组件

4️⃣并且为这些组件创建对象（组件不是接口，不是抽象类，从源码里看的哦），下面让我看看 WebAppleicationInitializer 的子类。

```java
public abstract class AbstractContextLoaderInitializer{}
// 作用是createRootApplicationContext() 创建根容器
```

```java
public abstract class AbstractContextLoaderInitializer{}
```

```java
public abstract class AbstractDispatcherServletInitializer{} 
// 看registerDispatcherServlet方法里的代码
// 创建一个web的ioc容器：createServletApplicationContext
// 创建一个DispatcherServlet：createDispatcherServlet
// 然后根据ServletContext的api，把创建的Servlet添加到web容器中/ 将创建的DispatcherServlet添加到Servletcontext中
```

```java
// 注解方式的配置的DispatcherServlet初始化器
public abstract class AbstractAnnotationConfigDispatcherServletInitializer{
    // 创建根容器：createRootApplicationContext
    protected WebApplicationContext createRootApplicationContext() {
        // 获得配置类
        Class<?>[] configClasses = getRootConfigClasses();
        if (!ObjectUtils.isEmpty(configClasses)) {
            AnnotationConfigWebApplicationContext rootAppContext = new AnnotationConfigWebApplicationContext();
            // 把配置类注册到根容器中
            rootAppContext.register(configClasses);
            return rootAppContext;
        }
        else {
            return null;
        }
    }
    // 创建Web的ioc容器
    protected WebApplicationContext createServletApplicationContext() {
        AnnotationConfigWebApplicationContext servletAppContext = new AnnotationConfigWebApplicationContext();
        Class<?>[] configClasses = getServletConfigClasses();
        if (!ObjectUtils.isEmpty(configClasses)) {
            servletAppContext.register(configClasses);
        }
        return servletAppContext;
    }
}
```

<b>总结</b>

以注解方式来启动 SpringMVC；

- 继承 AbstractAnnotationConfigDispatcherServletInitializer；
- 实现抽象方法指定 DispatcherServlet 的配置信息。

## 基本整合

[SpringMVC文档](https://docs.spring.io/spring-framework/docs/current/reference/html/web.html#mvc-servlet-context-hierarchy)

> <b>简单介绍</b>

- org.example.config
    - AppConfig.java ==> 配置 controller 的扫描
    - MyWebApplicationInitializer ==> Web 容器启动的时候创建对象；调用方法来初始化容器前端控制器
    - RootConfig ==> 根容器的配置。也就是 Spring 的，如配置 datasource，service，middle-tier
- controller
    - HelloController.java
- service
    - HelloService.java

### 配置文件代码

> <b>AppConfig 代码</b>

```java
package org.example.config;

import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.FilterType;
import org.springframework.stereotype.Controller;

// Spring容器不扫描 controller
// useDefaultFilters = false 禁用默认的过滤规则，默认是扫描所有的。
@ComponentScan(basePackages = "org.example", includeFilters = {
        @ComponentScan.Filter(type = FilterType.ANNOTATION, classes = Controller.class)
},useDefaultFilters = false)
public class AppConfig {
}
```

><b>MyWebApplicationInitializer 代码</b>

```java
package org.example.config;

import org.springframework.web.servlet.support.AbstractAnnotationConfigDispatcherServletInitializer;

// Web容器启动的时候创建对象；调用方法来初始化容器前端控制器
public class MyWebApplicationInitializer extends AbstractAnnotationConfigDispatcherServletInitializer {
    // 获取根容器的配置类; （以前是利用Spring的配置文件的方式，创建出一个父容器）
    protected Class<?>[] getRootConfigClasses() {

        return new Class[]{RootConfig.class};
    }

    // 获取web容器的配置类，相当于SpringMVC配置文件。
    protected Class<?>[] getServletConfigClasses() {
        return new Class[]{AppConfig.class};
    }

    // 获取DispatcherServlet的映射信息
    protected String[] getServletMappings() {
        // /    拦截所有资源，包括静态文件，但是不包括*.jsp
        // /*    拦截所有资源，包括静态文件和*.jsp；jsp页面是tomcat的jsp引擎解析的。
        return new String[]{"/"};
    }
}
```

> <b>RootConfig 代码</b>

```java
package org.example.config;

import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.FilterType;
import org.springframework.stereotype.Controller;

/**
 * 这个是 Root WebApplicationContext；根容器的配置。也就是Spring的
 * 如datasource、services、middle-tier
 */
@ComponentScan(basePackages = "org.example", excludeFilters = {
        // 排除所有的Controller
        @ComponentScan.Filter(type = FilterType.ANNOTATION, classes = {Controller.class})
})
public class RootConfig {
}
```

### 其他代码

> <b>HelloService 代码</b>

```java
package org.example.service;

import org.springframework.stereotype.Service;

@Service
public class HelloService {
    public String sayHello(String name) {
        return "Hello " + name;
    }
}
```

> <b>HelloController 代码</b>

```java
package org.example.controller;

import org.example.service.HelloService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class HelloController {

    @Autowired
    HelloService helloService;

    @ResponseBody
    @RequestMapping("/hello")
    public String hello() {
        String tomcat = helloService.sayHello("tomcat");
        return tomcat;
    }
}
```

## mvc定制整合

### 概述

> <b>xml 配置方式</b>

```xml
<!-- 将SpringMVC处理不了的请求交给tomcat；专门针对静态资源的，用这个配置，静态资源就可以访问了。 -->
<mvc:default-servlet-handler />
<!-- SpringMVC的高级功能开启 -->
<mvc:annotation-drivern />
<!-- 拦截器 -->
<mvc:interceptors></mvc:interceptors>
<mvc:view-controller path="" />
```

> <b>注解配置方式</b>

[SpringMVC 注解配置官方文档](https://docs.spring.io/spring-framework/docs/current/reference/html/web.html#mvc-config)

1️⃣@EnableWebMvc：开启 SpringMVC 定制配置功能；相当于 xml 中的 <mvc:annotation-drivern />

2️⃣配置组件（视图解析器、视图映射、静态资源映射、拦截器...）

3️⃣实现 WebMvcConfigurer 类，但是这个类的所有方法都要实现，有时候我们用不了这么多方法！怎么办？？SpringMVC 在这里用了一个设计模式，有一个实现了 WebMvcConfigurer 的抽象子类 WebMvcConfigurerAdapter，这个子类实现了它的所有方法，不过都是空方法！我们可以继承这个类哦！

具体代码看 github 吧。不贴代码了。

> <b>SpringMVC maven 目录结构说明</b>

1）java 目录放的 java 文件；最后都是输出到 classes 文件夹下

2）resources 放的资源文件；最后也是输出到 classes 文件夹下

3）webapp 是 web 目录；WEB-INF 目录下的最后是输出到 WEB-INF。static 与 webapp 的 WEB-INF 同级，那么它也会与最终输出文件的 WEB-INF 同级。

<div align="center"><img src="img/spring/maven_mvc.png"></div>

<div align="center"><img src="img/spring/maven_mvc2.png"></div>

# 异步请求

## 原生异步请求

Servlet 3.0 异步请求

### 概述

<div align="center"><img src="img/spring/servlet3.0_async.png"></div>

### 代码

```java
package org.example;

import javax.servlet.AsyncContext;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.concurrent.TimeUnit;

@WebServlet(urlPatterns = "/async", asyncSupported = true)
public class AsyncController extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
         // 不设置响应类型就无法异步
        resp.setContentType("text/html");
        
        // 1. 支持异步处理 asyncSupported = true
        // 2. 开启异步模式
        AsyncContext asyncContext = req.startAsync(req, resp);
        asyncContext.start(() -> {
            try {
                PrintWriter writer = asyncContext.getResponse().getWriter();
                for (int i = 0; i < 10; i++) {
                    TimeUnit.SECONDS.sleep(1);
                    writer.write("123"); writer.flush();
                }
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            } finally {
                asyncContext.complete();
            }
        });
    }
}
```

## MVC异步请求

### 返回 Callable

```java
@Controller
public class AsyncController {

    @ResponseBody
    @RequestMapping("/async")
    /**
     * 1、控制器返回Callable
     * 2、Spring异步处理，将Callable 提交道TaskExecutor 使用一个隔离的线程进行执行。
     * 3、DispatcherServlet和所有的Filter退出web容器的线程，但是response 保持打开状态。
     * 4、Callable返回结果，SpringMVC将重新发送请求。
     * 5、根据Callable返回的结果。SpringMVC继续进行视图渲染流程等。（从收请求 -> 视图渲染）
     *
     * 控制台输出 验证了上述的说法
     * preHandle
     * 主线程开始是...http-nio-8080-exec-2 ==> 1611740780382
     * 主线程结束是...http-nio-8080-exec-2 ==> 1611740780382
     * ===============DispatcherServlet及所有的Filter退出线程===============
     *
     * ===============等待Callable执行完成===============
     * 副线程是...MvcAsync1 ==> 1611740780394
     * 副线程是...MvcAsync1 ==> 1611740782395
     *
     * ===============Callable执行完成后又发送了一次请求===============
     * preHandle
     * postHandle
     * afterCompletion
     *
     * -----------------------------
     * 异步请求拦截器：
     *      - 原生api：AsyncListener
     *      - SpringMVC；实现AsyncHandlerInterceptor
     */
    public Callable<String> async() {
        System.out.println(String.format("主线程开始是...%s ==> %s", Thread.currentThread().getName(), System.currentTimeMillis()));
        Callable<String> callable = () -> {
            System.out.println(String.format("副线程是...%s ==> %s", Thread.currentThread().getName(), System.currentTimeMillis()));
            TimeUnit.SECONDS.sleep(2);
            System.out.println(String.format("副线程是...%s ==> %s", Thread.currentThread().getName(), System.currentTimeMillis()));
            return "Callable<String> async";
        };
        System.out.println(String.format("主线程结束是...%s ==> %s", Thread.currentThread().getName(), System.currentTimeMillis()));
        return callable;
    }
}
```

### 真实场景用法

```java
package org.example.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.context.request.async.DeferredResult;

import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.ConcurrentLinkedDeque;

@Controller
public class GroundTrueAsyncController {

    @RequestMapping("/createOrder")
    @ResponseBody
    public DeferredResult<Object> createOrder() {
        // 假设我们是指定了3秒内要完成，没完成就抛出错误 create fail
        // 他需要有人设置值才算成功         deferredResult.setResult("value")
        DeferredResult<Object> deferredResult = new DeferredResult<>(10000L, "create fail");
        DeferredResultQueue.save(deferredResult);
        return deferredResult;
    }

    @RequestMapping("/create")
    @ResponseBody
    public String create() {
        // 生成订单id
        String s = UUID.randomUUID().toString();
        DeferredResult<Object> deferredResult = DeferredResultQueue.get();
        // 存入订单id
        deferredResult.setResult(s);
        return "success==>" + s;
    }
}

class DeferredResultQueue {
    private static Queue<DeferredResult<Object>> queue = new ConcurrentLinkedDeque<>();

    public static void save(DeferredResult<Object> deferredResult) {
        queue.add(deferredResult);
    }

    public static DeferredResult<Object> get() {
        return queue.poll();
    }
}
```

# Spring5新特性

## 新功能

- 代码基于 Java8，运行时兼容 Java 9，把不建议的代码、库删除了。
- 自带通用日志封装
  - Spring 5 移除了 Log4jConfigListener，官方建议使用 Log4j2
  - Spring 5 框架整合 Log4j2
- 核心容器
  - 支持 @Nullable 注解
    - 可以使用在方法、属性、参数上
    - 方法：返回值可以为空
    - 属性：属性可以为空
    - 参数值：参数值可以为空

## Spring WebFlux

Spring 5 新功能，用于 web 开发，与 Spring MVC 类似，但是 WebFlux 是一种响应式编程框架。

WeblFlux 时一种异步非阻塞框架，Servlet 3.1 开始支持的。核心是基于 Reactor 相关的 API 实现的。

不扩充硬件资源的情况下，可以提升系统的吞吐量和伸缩性（秒杀系统用 WebFlux 试试）。

WebFlux 使用函数式编程实现路由请求。（观察者模式，数据发生变化就通知）

响应式流规范可以总结为 4 个接口：Publisher、Subscriber、Subscription 和 Processor。Publisher 负责生成数据，并将数据发送给 Subscription（每个 Subscriber 对应一个 Subscription）。Publisher 接口声明了一个方法subscribe()，Subscriber 可以通过该方法向 Publisher 发起订阅。

1️⃣命令式编程，假定有一批数据需要处理，每个数据都需要经过若干步骤才能完成。使用命令式编程模型，每行代码执行一个步骤，按部就班，并且肯定在同一个线程中进行。每一步在执行完成之前都会阻止执行线程执行下一步。

```java
String name = "xxx";
String cap = name.toUpperCase();
String out = "DFS "+ cap;
System.out.println(out);
```

2️⃣响应式编程，看起来依然保持着按步骤执行的模型，但实际是数据会流经处理管线。在处理管线的每一步，都对数据进行了某种形式的加工，但是我们不能判断数据会在哪个线程上执行操作。它们既可能在同一个线程，也可能在不同的线程。

```java
Mono.just("Craig")
    .map(n->n.toUpperCase())
    .map(cn->"DFS "+cn)
    .subscribe(System.out::println);
```

有点 Stream 流的并行编程的意思。

Reactor 有两个核心的类，Mono 和 Flux，这两个类实现了接口 Publisher，提供了丰富的操作接口。两者都实现了反应式流的 Publisher 接口。Flux 代表具有零个、一个或者多个（可能是无限个）数据项的管道。Mono 是一种特殊的反应式类型，针对数据项不超过一个的场景，它进行了优化。

```java
Mono.just("Craig")
    .map(n->n.toUpperCase())
    .map(cn->"DFS "+cn)
    .subscribe(System.out::println);
```

在这个例子中，有 3 个 Mono。just() 操作创建了第一个Mono。map 创建了第二个 Mono，map 创建了第三个 Mono。最后，对第三个 Mono 上的 subscribe() 方法调用时，会接收数据并将数据打印出来。

# Spring扩展点

- [Spring常用扩展点_星夜孤帆的博客-CSDN博客_spring扩展点](https://blog.csdn.net/qq_38826019/article/details/117389466)

- [Spring-MVC配置和扩展 - CodeAntenna](https://codeantenna.com/a/OO1sywTUnO)
