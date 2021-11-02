# 概述

## Spring Boot 简介

-  快速创建可运行的 Spring 项目并与主流框架进行了集成
-  使用嵌入式的 Servlet 容器，应用无需打成 war 包
-  starters 自动依赖与版本控制
-  大量的自动配置，简化开发
-  无需配置 xml，无代码生成，开箱即用
-  准生成环境的运行时应用监控
-  与云计算天然集成

## Spring Boot 优点

- Create stand-alone Spring applications
    - 创建独立Spring应用
- Embed Tomcat, Jetty or Undertow directly (no need to deploy WAR files)
    - 内嵌 web 服务器
- Provide opinionated 'starter' dependencies to simplify your build configuration
    - 自动 starter 依赖，简化构建配置
- Automatically configure Spring and 3rd party libraries whenever possible
    - 自动配置 Spring 以及第三方功能
- Provide production-ready features such as metrics, health checks, and externalized configuration
    - 提供生成级别的监控、健康检查以及外部化配置
- Absolutely no code generation and no requirement for XML configuration
    - 无代码生成、无需编写 XML

## Spring Boot 缺点

- 迭代快，需要时刻关注变化
- 封装太深，内部原理复杂，不易精通

## 微服务

- 微服务是一种架构风格（服务微化）
- 一个应用拆分为一组小型服务；可以通过HTTP的方式进行互通；
- 每个服务运行在自己的进程内，也就是可独立部署和升级
- 服务围绕业务功能拆分
- 可以由全自动部署机制独立部署
- 去中心化，服务自治。服务可以使用不同的语言、不同的存储技术

## 分布式

### 需要解决的问题

- 远程调用
- 服务发现
- 负载均衡
- 服务容错
- 配置管理
- 服务监控
- 链路追踪
- 日志管理
- 任务调度
- ......

### 分布式技术组合

Spring Boot + Spring Cloud

# 快速上手

## 文档概述

<img src="../pics/Spring Boot/doc_1.png">

<img src="../pics/Spring Boot/doc_2.png">

查看版本新特性；

https://github.com/spring-projects/spring-boot/wiki#release-notes

![image-20211003101958201](..\pics\Spring Boot\doc_3.png)

## 创建

- IDEA 直接创建
- 官网骨架创建

使用 Spring Initializer快速创建，直接选择它，不选择maven。给 maven 的 settings.xml配置文件的 profiles  标签添加

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>1.5.9.RELEASE</version>
</parent>
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

> 编写一个主程序；启动Spring Boot应用

```java
/**
 *  @SpringBootApplication 来标注一个主程序类，说明这是一个Spring Boot应用
 */
@SpringBootApplication
public class HelloWorldMainApplication {

    public static void main(String[] args) {

        // Spring应用启动起来
        SpringApplication.run(HelloWorldMainApplication.class,args);
    }
}
```

> 编写相关的Controller、Service，运行主程序测试

```java
@Controller
public class HelloController {

    @ResponseBody
    @RequestMapping("/hello")
    public String hello(){
        return "Hello World!";
    }
}

```

> Spring Boot 的配置

所有的配置信息全部写在 application.porperties 或 application.yml 中

[Common Application properties (spring.io)](https://docs.spring.io/spring-boot/docs/2.4.11/reference/html/appendix-application-properties.html#common-application-properties-web)

> 简化部署

引入插件，就可以将项目打成 jar 包。

```xml
<!-- 这个插件，可以将应用打包成一个可执行的jar包；-->
<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

将这个应用打成jar包，直接使用 java -jar 的命令进行执行；打成 jar 包可以直接用 maven的 package 命令或者控制台 mvn package 使用 `mvn spring-boot:run` 启动

**注意：**如果引入了JDBC相关的依赖，需要我们在配置文件中配置连接数据库相关的操作。

## 目录解释

> resources文件夹中目录结构

- static：保存静态资源；js css img
- templates：保存页面资源；springboot默认不支持jsp
- mybatis：的配置文件之类的需要放在resources文件夹下面。resources是资源的根路径。就把resources当成编译后的classes文件夹吧。

## 基本原理

### Spring Boot 特点

#### 依赖管理

- 父项目做依赖管理
- 开发导入starter场景启动器
- 无需关注版本号，自动版本仲裁
- 可以修改默认版本号

> 依赖管理

<span style="color:red">spring-boot-starter-parent --> spring-boot-dependencies --> 维护了版本号</span>

`spring-boot-dependencies` 中维护了常见 jar 的版本号，我们添加依赖的时候就不必添加版本了。

```xml
<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
</dependency>
```

如果版本仲裁的版本不合适，那么我们可以自己修改，在 Spring Boot 工程的 pom 中修改 properties

```xml
<properties>
    <java.version>1.8</java.version>
    <mysql.version>5.1.43</mysql.version>
</properties>
```

> 场景启动器 starter

starter 是一组依赖的集合描述。一般引入一个 starter 整个完整的开发场景就会被引入。官方 starter 的命名 `spring-boot-starter-*`，自定义 starter 的命名推荐 `*-spring-boot-starter`

Spring Boot 的 starter [Developing with Spring Boot](https://docs.spring.io/spring-boot/docs/current/reference/html/using.html#using.build-systems.starters)

所有场景启动器的最底层的依赖就是 spring-boot-starter

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter</artifactId>
    <version>2.5.5</version>
    <scope>compile</scope>
</dependency>
```

> 无需关注版本号，自动版本仲裁

- 引入依赖默认都可以不写版本 
- 引入非版本仲裁的jar，要写版本号。

>可以修改默认版本号

如果版本仲裁的版本不合适，那么我们可以自己修改，在 Spring Boot 工程的 pom 中修改 properties

```xml
<properties>
    <java.version>1.8</java.version>
    <mysql.version>5.1.43</mysql.version>
</properties>
```

#### 自动配置

- 自动配好Tomcat

    - 引入Tomcat依赖。
    - 配置 Tomcat

- 自动配好 Spring MVC

    - 引入 SpringMVC 全套组件
    - 自动配好 Spring MVC 常用组件（功能）

- 自动配好 web 常见功能，如：字符编码问题

    - Spring Boot 帮我们配置好了所有 web 开发的常见场景

- 默认的包结构

    - 主程序所在包及其下面的所有子包里面的组件都会被默认扫描进来
    - 无需以前的包扫描配置
    - 想改变扫描路径的话 @SpringBootApplication(scanBasePackages="xxx")
    - 或者 @ComponentScan 指定扫描路径

- 各种配置拥有默认值

    - 默认配置最终都是映射到某个类上，如：MultipartProperties 类
    - 配置文件的值最终会绑定每个类上，这个类会在容器中创建对象

- 按需加载所有自动配置项

    - 非常多的starte
    - 引入了哪些场景这个场景的自动配置才会开启
    - Spring Boot 所有的自动配置功能都在 `spring-boot-autoconfigure` 包里面

> POM文件

父项目

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>1.5.9.RELEASE</version>
</parent>

他的父项目是
<parent>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-dependencies</artifactId>
  <version>1.5.9.RELEASE</version>
  <relativePath>../../spring-boot-dependencies</relativePath>
</parent>
他来真正管理Spring Boot应用里面的所有依赖版本；
```

Spring Boot的版本仲裁中心；

以后我们导入依赖默认是不需要写版本；（没有在dependencies里面管理的依赖自然需要声明版本号）

启动器

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
```

**spring-boot-starter**-web：

spring-boot-starter：spring-boot场景启动器；帮我们导入了web模块正常运行所依赖的组件；

Spring Boot将所有的功能场景都抽取出来，做成一个个的starters（启动器），只需要在项目里面引入这些starter相关场景的所有依赖都会导入进来。要用什么功能就导入什么场景的启动器

> 主程序类，主入口类

```java
/**
 *  @SpringBootApplication 来标注一个主程序类，说明这是一个Spring Boot应用
 */
@SpringBootApplication
public class HelloWorldMainApplication {

    public static void main(String[] args) {

        // Spring应用启动起来
        SpringApplication.run(HelloWorldMainApplication.class,args);
    }
}
```

@**SpringBootApplication**：Spring Boot应用标注在某个类上说明这个类是SpringBoot的主配置类，SpringBoot就应该运行这个类的main方法来启动SpringBoot应用；

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(excludeFilters = {
      @Filter(type = FilterType.CUSTOM, classes = TypeExcludeFilter.class),
      @Filter(type = FilterType.CUSTOM, classes = AutoConfigurationExcludeFilter.class) })
public @interface SpringBootApplication {
```

@**SpringBootConfiguration**:Spring Boot的配置类；

		标注在某个类上，表示这是一个Spring Boot的配置类；
	
		@**Configuration**:配置类上来标注这个注解；
	
			配置类 -----  配置文件；配置类也是容器中的一个组件；@Component

@**EnableAutoConfiguration**：开启自动配置功能；

以前我们需要配置的东西，Spring Boot帮我们自动配置；@**EnableAutoConfiguration**告诉SpringBoot开启自动配置功能；这样自动配置才能生效；

```java
@AutoConfigurationPackage
@Import(EnableAutoConfigurationImportSelector.class)
public @interface EnableAutoConfiguration {
```

```java
  	@**AutoConfigurationPackage**：自动配置包

	@**Import**(AutoConfigurationPackages.Registrar.class)：

	Spring的底层注解@Import，给容器中导入一个组件；导入的组件由AutoConfigurationPackages.Registrar.class;
```

**将主配置类（@SpringBootApplication标注的类）的所在包及下面所有子包里面的所有组件扫描到Spring容器；**

```java
@**Import**(EnableAutoConfigurationImportSelector.class)；

	给容器中导入组件？

	**EnableAutoConfigurationImportSelector**：导入哪些组件的选择器；

	将所有需要导入的组件以全类名的方式返回；这些组件就会被添加到容器中；

	会给容器中导入非常多的自动配置类（xxxAutoConfiguration）；就是给容器中导入这个场景需要的所有组件，并配置好这些组件；		![自动配置类](images/搜狗截图20180129224104.png)
```

有了自动配置类，免去了我们手动编写配置注入功能组件等的工作；

```java
SpringFactoriesLoader.loadFactoryNames(EnableAutoConfiguration.class,classLoader)；
```

**Spring Boot 在启动的时候从类路径下的 META-INF/spring.factories 中获取EnableAutoConfiguration 指定的值，将这些值作为自动配置类导入到容器中，自动配置类就生效，帮我们进行自动配置工作；**以前我们需要自己配置的东西，自动配置类都帮我们；J2EE 的整体整合解决方案和自动配置都在 spring-boot-autoconfigure-1.5.9.RELEASE.jar；

### 容器功能

> 常用注解

- @Configuration、告诉 Spring Boot 这是一个配置类，配置文件。
- @Bean、标注在方法上给容器注册组件，默认是单实例，名称为方法名。
- @Component、
- @Controller、
- @Service、
- @Repository、
- @ComponentScan、
- @Import、
- @Conditional、
- @ImportResource、
- @ConfigurationProperties、
- @EnableConfigurationProperties
- @EnableAutoConfiguration
- @AutoConfigurationPackage

#### 组件添加

> @Configuration+@Bean
Spring Boot2.0 @Configuration 注解添加了一个新的属性 `proxyBeanMethods`，代理 bean 的方法。

```java
@Configuration(proxyBeanMethods = true)  // 默认是 true
// proxyBeanMethods 代理 bean 的方法
// 配置类也是注解
public class MyConfig {
    @Bean
    public Pet tomcat() {
        return new Pet();
    }
}

@SpringBootApplication
public class SpringBoot2Application {

    public static void main(String[] args) {
        ConfigurableApplicationContext run = SpringApplication.run(SpringBoot2Application.class, args);
        // MyConfig$$EnhancerBySpringCGLIB$$cf24ac66@2c54c0c0 代理的
        MyConfig bean = run.getBean(MyConfig.class);
        System.out.println(bean);
		// 如果 MyConfig 是代理对象，则两次拿到的 tomcat 都是一样的，都是从容器中拿的
        // 如果不是代理对象，则每次调用方法生成的都是新的对象
        Pet tomcat1 = bean.tomcat();
        Pet tomcat2 = bean.tomcat();
        System.out.println(tomcat1 == tomcat2);
        run.close();
    }

}
/* 
是代理对象
com.example.springboot2.config.MyConfig$$EnhancerBySpringCGLIB$$5bb12bc0@40d71db3
true

不是代理对象
com.example.springboot2.config.MyConfig@389720c3
false
*/
```

**Full 模式与 Lite 模式**

- 示例：最佳实战
    - 配置类组件之间无依赖关系用 Lite 模式加速容器启动过程，不会检查容器中是否这个对象，减少判断
    - 配置类组件之间有依赖关系，方法会被调用得到之前单实例组件（IOC 容器中的组件），用 Full 模式

> @Import 导入组件

@Import({User.class, DBHelper.class}) 给容器中自动创建出这两个类型的组件、默认组件的名字就是全类名

> Conditional

条件装配：满足 Condition 指定的条件，则进行组件注入

![image-20211003163043774](..\pics\Spring Boot\conditional_1.png)

@Conditional 派生注解（ Spring 注解版原生的 @Conditional作用 ）

作用：必须是 @Conditional 指定的条件成立，才给容器中添加组件，配置配里面的所有内容才生效；

| @Conditional扩展注解            | 作用（判断是否满足当前指定条件）                 |
| ------------------------------- | ------------------------------------------------ |
| @ConditionalOnJava              | 系统的java版本是否符合要求                       |
| @ConditionalOnBean              | 容器中存在指定Bean；                             |
| @ConditionalOnMissingBean       | 容器中不存在指定Bean；                           |
| @ConditionalOnExpression        | 满足SpEL表达式指定                               |
| @ConditionalOnClass             | 系统中有指定的类                                 |
| @ConditionalOnMissingClass      | 系统中没有指定的类                               |
| @ConditionalOnSingleCandidate   | 容器中只有一个指定的Bean，或者这个Bean是首选Bean |
| @ConditionalOnProperty          | 系统中指定的属性是否有指定的值                   |
| @ConditionalOnResource          | 类路径下是否存在指定资源文件                     |
| @ConditionalOnWebApplication    | 当前是web环境                                    |
| @ConditionalOnNotWebApplication | 当前不是web环境                                  |
| @ConditionalOnJndi              | JNDI存在指定项                                   |

测试条件装配。两个类一个 Person 一个 Pet。有 Pet 对象了，Person 才实例化

```java
public class Pet {

}

public class Person {
    public Pet per;
}

@Configuration(proxyBeanMethods = true)
public class MyConfig {
    @Bean
    public Pet tomcat() {
        return new Pet();
    }

    @Bean
    @ConditionalOnBean(name = {"tomcat22"}) // 没有 tomcat22 名字的 bean 则不注入 person
    public Person person() {
        Person person = new Person();
        person.per = tomcat();
        return new Person();
    }
}

@SpringBootApplication
public class TestConditional {
    public static void main(String[] args) {
        ConfigurableApplicationContext run = SpringApplication.run(TestConditional.class, args);
        System.out.println(run.getBean(Pet.class));
        String[] names = run.getBeanDefinitionNames();
        Arrays.stream(names)
                .filter(s -> s.contains("person") || s.contains("pet"))
                .forEach(System.out::println);
    }
}

/*
只有 pet ，没有 person
com.example.springboot2.config.Pet@4b3b442a
*/
```

#### xml 引入

> @ImportResource 导入并解析 xml 配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean id="mypet" class="com.example.springboot2.config.Pet"></bean>
</beans>
```

```java
@Configuration(proxyBeanMethods = true)
@ImportResource("classpath:bean.xml")// 会导入 resources 下的 xml 文件作为配置
public class MyConfig {
}

@SpringBootApplication
public class TestImportResource {
    public static void main(String[] args) {
        ConfigurableApplicationContext run = SpringApplication.run(TestImportResource.class, args);
        System.out.println(run.getBean(Pet.class));
    }
}
```

#### 配置信息绑定

我们在 properties 文件中书写了一些配置信息，那 Spring Boot 如何得到这些配置信息呢？

> Java 代码读取 properties 文件

```java
public class getProperties {
     public static void main(String[] args) throws FileNotFoundException, IOException {
         Properties pps = new Properties();
         pps.load(new FileInputStream("a.properties"));
         Enumeration enum1 = pps.propertyNames();//得到配置文件的名字
         while(enum1.hasMoreElements()) {
             String strKey = (String) enum1.nextElement();
             String strValue = pps.getProperty(strKey);
             System.out.println(strKey + "=" + strValue);
             //封装到JavaBean。
         }
     }
 }
```

> Spring Boot 获得 properties & yml 中的信息

@ConfigurationProperties + @Component

@ConfigurationProperties 注解，通过 set/get 方法进行值的注入的。注意：只有这个 bean 被注入到容器中了，才可以拥有所提供的值注入。

```java
@Component
@ConfigurationProperties(prefix = "mycar")
public class MyCar {
    String bound;
    int price;

    public String getBound() {
        return bound;
    }

    public void setBound(String bound) {
        this.bound = bound;
    }

    public int getPrice() {
        return price;
    }

    public void setPrice(int price) {
        this.price = price;
    }
}


@SpringBootApplication
public class TestMyCar {
    public static void main(String[] args) {
        ConfigurableApplicationContext run = SpringApplication.run(TestMyCar.class, args);
        MyCar bean = run.getBean(MyCar.class);
        System.out.println(bean.price);
    }
}
/*
100
*/
```

@EnableConfigurationProperties + @ConfigurationProperties

### 自动配置原理

#### 引导加载自动配置类

- @SpringBootConfiguration 表示是一个配置类
- @ComponentScan 指定扫描哪些，Spring注解；
- @EnableAutoConfiguration，开启自动配置 ★★★★

```java
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(excludeFilters = { @Filter(type = FilterType.CUSTOM, classes = TypeExcludeFilter.class),
    @Filter(type = FilterType.CUSTOM, classes = AutoConfigurationExcludeFilter.class) })
public @interface SpringBootApplication{
    // 是一个合成注解
}
```

> @EnableAutoConfiguration

- @AutoConfigurationPackage  自动配置包
- @Import(AutoConfigurationImportSelector.class)

```java
@AutoConfigurationPackage
@Import(AutoConfigurationImportSelector.class)
public @interface EnableAutoConfiguration {
}
```

@AutoConfigurationPackage

```java
@Import(AutoConfigurationPackages.Registrar.class)
public @interface AutoConfigurationPackage {
    // 将指定的一个包下的所有组件导入进来，就是我们加了 @SpringBootApplication 类所在的包
}

// 批量注册
static class Registrar implements ImportBeanDefinitionRegistrar, DeterminableImports {

    @Override
    public void registerBeanDefinitions(AnnotationMetadata metadata, BeanDefinitionRegistry registry) {
        // new PackageImports(metadata).getPackageNames() 得到一个包名，把包下的组件批量注入进来
        register(registry, new PackageImports(metadata).getPackageNames().toArray(new String[0]));
    }

    @Override
    public Set<Object> determineImports(AnnotationMetadata metadata) {
        return Collections.singleton(new PackageImports(metadata));
    }

}
```

@Import(AutoConfigurationImportSelector.class)

```java
// AutoConfigurationImportSelector 源码中的 selectImports 方阿飞
@Override
public String[] selectImports(AnnotationMetadata annotationMetadata) {
    if (!isEnabled(annotationMetadata)) {
        return NO_IMPORTS;
    }
    // 给容器批量导入一些组件，我们自习看下 getXX方法的源码
    AutoConfigurationEntry autoConfigurationEntry = getAutoConfigurationEntry(annotationMetadata);
    return StringUtils.toStringArray(autoConfigurationEntry.getConfigurations());
}

protected AutoConfigurationEntry getAutoConfigurationEntry(AnnotationMetadata annotationMetadata) {
    if (!isEnabled(annotationMetadata)) {
        return EMPTY_ENTRY;
    }
    AnnotationAttributes attributes = getAttributes(annotationMetadata);
    // 获取候选的，需要装配的 bean。
    List<String> configurations = getCandidateConfigurations(annotationMetadata, attributes);
    configurations = removeDuplicates(configurations);
    Set<String> exclusions = getExclusions(annotationMetadata, attributes);
    checkExcludedClasses(configurations, exclusions);
    configurations.removeAll(exclusions);
    configurations = getConfigurationClassFilter().filter(configurations);
    fireAutoConfigurationImportEvents(configurations, exclusions);
    return new AutoConfigurationEntry(configurations, exclusions);
}
```

![image-20211003173551017](..\pics\Spring Boot\AutoConfigurationImportSelector.png)

```java
protected List<String> getCandidateConfigurations(AnnotationMetadata metadata, AnnotationAttributes attributes) {
    // 利用工厂加载，得到所有加载。loadFactoryNames 调用了 loadSpringFactories 方法
    List<String> configurations = SpringFactoriesLoader.loadFactoryNames(getSpringFactoriesLoaderFactoryClass(),
                                                                         getBeanClassLoader());
    Assert.notEmpty(configurations, "No auto configuration classes found in META-INF/spring.factories. If you "
                    + "are using a custom packaging, make sure that file is correct.");
    return configurations;
}

private static Map<String, List<String>> loadSpringFactories(ClassLoader classLoader) {
    Map<String, List<String>> result = cache.get(classLoader);
    if (result != null) {
        return result;
    }

    result = new HashMap<>();
    try {
        // 从 FACTORIES_RESOURCE_LOCATION=META-INF/spring.factories 加载数据
        // 默认扫描我们当前系统里面所有 META-INF/spring.factories 位置的文件
        // 文件里写死了，spring-boot一启动就要给容器中加载的所有配置类
        Enumeration<URL> urls = classLoader.getResources(FACTORIES_RESOURCE_LOCATION);
        while (urls.hasMoreElements()) {
            URL url = urls.nextElement();
            UrlResource resource = new UrlResource(url);
            Properties properties = PropertiesLoaderUtils.loadProperties(resource);
            for (Map.Entry<?, ?> entry : properties.entrySet()) {
                String factoryTypeName = ((String) entry.getKey()).trim();
                String[] factoryImplementationNames =
                    StringUtils.commaDelimitedListToStringArray((String) entry.getValue());
                for (String factoryImplementationName : factoryImplementationNames) {
                    result.computeIfAbsent(factoryTypeName, key -> new ArrayList<>())
                        .add(factoryImplementationName.trim());
                }
            }
        }

        // Replace all lists with unmodifiable lists containing unique elements
        result.replaceAll((factoryType, implementations) -> implementations.stream().distinct()
                          .collect(Collectors.collectingAndThen(Collectors.toList(), Collections::unmodifiableList)));
        cache.put(classLoader, result);
    }
    catch (IOException ex) {
        throw new IllegalArgumentException("Unable to load factories from location [" +
                                           FACTORIES_RESOURCE_LOCATION + "]", ex);
    }
    return result;
}
```

#### 按需开启自动配置项

虽然我们 134 个场景的所有自动配置启动的时候默认全部加载。xxxxAutoConfiguration 按照条件装配规则（@Conditional），最终会按需配置。如何按需配置呢？就是通过前面介绍的 @Conditional 注解

```java
@Configuration(proxyBeanMethods = false)
@ConditionalOnProperty(prefix = "spring.aop", name = "auto", havingValue = "true", matchIfMissing = true)
public class AopAutoConfiguration {

	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass(Advice.class)
	// some code
}
```

#### 修改默认配置

配置文件中是否存在 spring.aop 的配置，如果存在 spring.aop.auto ，且值为 true，就失效。即便没有配，也认为配置了。

```java
@Configuration(proxyBeanMethods = false)
@ConditionalOnProperty(prefix = "spring.aop", name = "auto", havingValue = "true", matchIfMissing = true)
public class AopAutoConfiguration {
    	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass(Advice.class) // 存在 Advice.class 才失效
	static class AspectJAutoProxyingConfiguration {
	}

	@Configuration(proxyBeanMethods = false)
	@ConditionalOnMissingClass("org.aspectj.weaver.Advice") // 系统里没有这个类才生效。
	@ConditionalOnProperty(prefix = "spring.aop", name = "proxy-target-class", havingValue = "true",
			matchIfMissing = true)
	static class ClassProxyingConfiguration {

		@Bean
		static BeanFactoryPostProcessor forceAutoProxyCreatorToUseClassProxying() {
			return (beanFactory) -> {
				if (beanFactory instanceof BeanDefinitionRegistry) {
					BeanDefinitionRegistry registry = (BeanDefinitionRegistry) beanFactory;
					AopConfigUtils.registerAutoProxyCreatorIfNecessary(registry);
					AopConfigUtils.forceAutoProxyCreatorToUseClassProxying(registry);
				}
			};
		}

	}

}
```

容器组件注入（重命名，不按 Spring Boot 约定的名字来）

```java
@Bean
@ConditionalOnBean(MultipartResolver.class)  //容器中有这个类型组件
@ConditionalOnMissingBean(name = DispatcherServlet.MULTIPART_RESOLVER_BEAN_NAME) //容器中没有这个名字 multipartResolver 的组件
public MultipartResolver multipartResolver(MultipartResolver resolver) {
    //给@Bean标注的方法传入了对象参数，这个参数的值就会从容器中找。
    //SpringMVC multipartResolver。防止有些用户配置的文件上传解析器不符合规范
    // Detect if the user has created a MultipartResolver but named it incorrectly
    return resolver;
}
// 给容器中加入了文件上传解析器；
```

SpringBoot 默认会在底层配好所有的组件。但是如果用户自己配置了以用户的优先

```java
@Bean
@ConditionalOnMissingBean
public CharacterEncodingFilter characterEncodingFilter() {
}
```

#### 自动配置原理

以**HttpEncodingAutoConfiguration（Http编码自动配置）**为例解释自动配置原理；

```java
@Configuration   //表示这是一个配置类，以前编写的配置文件一样，也可以给容器中添加组件
@EnableConfigurationProperties(HttpEncodingProperties.class)  //启动指定类的ConfigurationProperties功能；将配置文件中对应的值和HttpEncodingProperties绑定起来；并把HttpEncodingProperties加入到ioc容器中

@ConditionalOnWebApplication //Spring底层@Conditional注解（Spring注解版），根据不同的条件，如果满足指定的条件，整个配置类里面的配置就会生效；    判断当前应用是否是web应用，如果是，当前配置类生效

@ConditionalOnClass(CharacterEncodingFilter.class)  //判断当前项目有没有这个类CharacterEncodingFilter；SpringMVC中进行乱码解决的过滤器；

@ConditionalOnProperty(prefix = "spring.http.encoding", value = "enabled", matchIfMissing = true)  //判断配置文件中是否存在某个配置  spring.http.encoding.enabled；如果不存在，判断也是成立的
//即使我们配置文件中不配置pring.http.encoding.enabled=true，也是默认生效的；
public class HttpEncodingAutoConfiguration {
  
  	//他已经和SpringBoot的配置文件映射了
  	private final HttpEncodingProperties properties;
  
   //只有一个有参构造器的情况下，参数的值就会从容器中拿
  	public HttpEncodingAutoConfiguration(HttpEncodingProperties properties) {
		this.properties = properties;
	}
  
    @Bean   //给容器中添加一个组件，这个组件的某些值需要从properties中获取
	@ConditionalOnMissingBean(CharacterEncodingFilter.class) //判断容器没有这个组件？
	public CharacterEncodingFilter characterEncodingFilter() {
		CharacterEncodingFilter filter = new OrderedCharacterEncodingFilter();
		filter.setEncoding(this.properties.getCharset().name());
		filter.setForceRequestEncoding(this.properties.shouldForce(Type.REQUEST));
		filter.setForceResponseEncoding(this.properties.shouldForce(Type.RESPONSE));
		return filter;
	}
```

根据当前不同的条件判断，决定这个配置类是否生效？

一但这个配置类生效；这个配置类就会给容器中添加各种组件；这些组件的属性是从对应的properties类中获取的，这些类里面的每一个属性又是和配置文件绑定的；

所有在配置文件中能配置的属性都是在xxxxProperties类中封装者‘；配置文件能配置什么就可以参照某个功能对应的这个属性类

```java
@ConfigurationProperties(prefix = "spring.http.encoding")  //从配置文件中获取指定的值和bean的属性进行绑定
public class HttpEncodingProperties {
   public static final Charset DEFAULT_CHARSET = Charset.forName("UTF-8");
}
```

#### 总结

- SpringBoot先加载所有的自动配置类  xxxxxAutoConfiguration
- 每个自动配置类按照条件进行生效，默认都会绑定配置文件指定的值。xxxxProperties里面拿。xxxProperties和配置文件进行了绑定

- 生效的配置类就会给容器中装配很多组件
- 只要容器中有这些组件，相当于这些功能就有了
- 定制化配置
    - 用户直接自己@Bean替换底层的组件
    - 用户去看这个组件是获取的配置文件什么值就去修改。

**xxxxxAutoConfiguration ---> 把组件装配进去  --->** **组件从xxxxProperties里面拿值 ----> application.properties**

## 实践方式

- 引入场景依赖
    - https://docs.spring.io/spring-boot/docs/current/reference/html/using-spring-boot.html#using-boot-starter


- 查看自动配置了哪些


  - 自己分析，引入场景对应的自动配置一般都生效了
  - <span style="color:red">配置文件中 debug=true 开启自动配置报告，让控制台打印自动配置报告。Negative（不生效）\Positive（生效）</span>

  ```shell
  =========================
  AUTO-CONFIGURATION REPORT
  =========================
  Positive matches:（自动配置类启用的）
  -----------------
     DispatcherServletAutoConfiguration matched:
        - @ConditionalOnClass found required class 'org.springframework.web.servlet.DispatcherServlet'; @ConditionalOnMissingClass did not find unwanted class (OnClassCondition)
        - @ConditionalOnWebApplication (required) found StandardServletEnvironment (OnWebApplicationCondition)
      
  Negative matches:（没有启动，没有匹配成功的自动配置类）
  -----------------
     ActiveMQAutoConfiguration:
        Did not match:
           - @ConditionalOnClass did not find required classes 'javax.jms.ConnectionFactory', 'org.apache.activemq.ActiveMQConnectionFactory' (OnClassCondition)
  
     AopAutoConfiguration:
        Did not match:
           - @ConditionalOnClass did not find required classes 'org.aspectj.lang.annotation.Aspect', 'org.aspectj.lang.reflect.Advice' (OnClassCondition)    
  ```


- 是否需要修改


  - 参照文档修改配置项


    - https://docs.spring.io/spring-boot/docs/current/reference/html/appendix-application-properties.html#common-application-properties
    - 自己分析。xxxxProperties 绑定了配置文件的哪些。
  - 自定义加入或者替换组件

    - @Bean、@Component。。。
  - 自定义器  **XXXXXCustomizer**；
  - ......

## 开发小工具

### Lombok

Lombok 简化 JavaBean 开发；IDEA 中搜索安装 lombok 插件

```xml
<dependency>
    <groupId>org.projectlombok</groupId>
    <artifactId>lombok</artifactId>
</dependency>
```

简单使用

```java
@NoArgsConstructor // 无参构造器
//@AllArgsConstructor // 全参构造器
@Data // get set
@ToString // toString 方法
@EqualsAndHashCode // equals 和 hashCode 方法
public class User {

    private String name;
    private Integer age;

    private Pet pet;

    public User(String name,Integer age){
        this.name = name;
        this.age = age;
    }

}
```

lombok 可以简化日志开发

```java
@Slf4j
@RestController
public class HelloController {
    @RequestMapping("/hello")
    public String handle01(@RequestParam("name") String name){
        log.info("请求进来了....");
        return "Hello, Spring Boot 2!"+"你好："+name;
    }
}
```

### dev-tools

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-devtools</artifactId>
    <optional>true</optional>
</dependency>
```

项目或者页面修改以后：只需要 Ctrl+F9 就可以实时生效；如果想要真正的热更新，需要用付费的 jrebel。

### 配置文件提示

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-configuration-processor</artifactId>
    <optional>true</optional>
</dependency>
```

# 配置文件

SpringBoot使用一个全局的配置文件，配置文件名是固定的；

- application.properties

- application.yml

配置文件的作用：修改SpringBoot自动配置的默认值；SpringBoot在底层都给我们自动配置好；

YAML（YAML Ain't Markup Language）

```yaml
YAML  A Markup Language：是一个标记语言

YAML   isn't Markup Language：不是一个标记语言；
```

标记语言：

```yaml
以前的配置文件；大多都使用的是  **xxxx.xml**文件;
YAML：**以数据为中心**，比json、xml等更适合做配置文件;
YAML：配置例子
```

```yaml
server:
  port: 8081
```

xml

```xml
<server>
	<port>8081</port>
</server>
```

## YAML

### 基本语法

k:(空格)v：表示一对键值对（空格必须有）；以**空格**的缩进来控制层级关系；只要是左对齐的一列数据，都是同一个层级的。

注意点：yaml 中单引号会将 \n 作为字符串输出；双引号会解析 \n 的含义换行输出。

```yaml
server:
    port: 8081
    path: /hello
```

属性和值也是大小写敏感

### 值的写法

**字面量：普通的值（数字，字符串，布尔）**

**k: v：字面直接来写；**

- 字符串默认不用加上单引号或者双引号；
    - ""：双引号；不会转义字符串里面的特殊字符；特殊字符会作为本身想表示的意思
        name:   "zhangsan \n lisi"：输出；zhangsan 换行  lisi
    - ''：单引号；会转义特殊字符，特殊字符最终只是一个普通的字符串数据
        name:   ‘zhangsan \n lisi’：输出；zhangsan \n  lisi

> **对象、Map（属性和值）（键值对）：**

k: v在下一行来写对象的属性和值的关系；注意缩进，对象还是k: v的方式

```yaml
friends:
		lastName: zhangsan
		age: 20
```

行内写法：

```yaml
friends: {lastName: zhangsan,age: 18}
```

> **数组（List、Set）：**

用- 值表示数组中的一个元素

```yaml
pets:
 - cat
 - dog
 - pig
```

行内写法

```yaml
pets: [cat,dog,pig]
```

### 具体代码

**yaml：**

```yaml
person:
    lastName: hello
    age: 18
    boss: false
    birth: 2017/12/12
    maps: {k1: v1,k2: 12}
    lists:
      - lisi
      - zhaoliu
    dog:
      name: 小狗
      age: 12
```

**JavaBean：**

```java
/**
 * 将配置文件中配置的每一个属性的值，映射到这个组件中
 * @ConfigurationProperties：告诉Spring Boot将本类中的所有属性和配置文件中相关的配置进行绑定；
 *      prefix = "person"：配置文件中哪个下面的所有属性进行一一映射
 * 只有这个组件是容器中的组件，才能容器提供的@ConfigurationProperties功能；
 */
@Component
@ConfigurationProperties(prefix = "person")
public class Person {

    private String lastName;
    private Integer age;
    private Boolean boss;
    private Date birth;

    private Map<String,Object> maps;
    private List<Object> lists;
    private Dog dog;
    
}
```

我们可以导入配置文件处理器，以后编写配置就有提示了。

```xml
<!--导入配置文件处理器，配置文件进行绑定就会有提示-->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-configuration-processor</artifactId>
    <optional>true</optional>
</dependency>
```

### 配置提示

自定义的类和配置文件绑定一般没有提示。引入一下的插件会有提示

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-configuration-processor</artifactId>
    <optional>true</optional>
</dependency>


<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
            <configuration>
                <excludes>
                    <!-- 打包时排除这个 -->
                    <exclude>
                        <groupId>org.springframework.boot</groupId>
                        <artifactId>spring-boot-configuration-processor</artifactId>
                    </exclude>
                </excludes>
            </configuration>
        </plugin>
    </plugins>
</build>
```

## Properties

```properties
# 配置person的值
person.name=liuhanboy
person.age=233
person.date=1997/11/11
person.girls.name=lh
person.habby=a,b,c
person.maps.k1=v1
person.maps.k2=v2
```

> **properties 配置文件在 idea 中默认 utf-8 可能会乱码**

`idea中找到File Encodings，勾选 Transparent native-to-ascii conversion`

> **@Value赋值**

- Value("${xx}") 表达式语法，获取配置文件中的xx的值
- Value("#{11*22}") 表达式语法，可以进行计算

```java
@Component
// @ConfigurationProperties(prefix = "person")
public class Person {
    @Value("${person.name}")
    private String name;
    @Value("#{11*2}")
    private String age;
    @Value("${person.date}")
    private Date date;
}
```

> **@Validated**及其使用

- 有@ConfigurationProperties @Validated的那些什么校验的注解才有作用

```java
@Component
@ConfigurationProperties(prefix = "person")
@Validated
public class Person {
    @Value("${person.name}")
    private String name;
    @Value("#{11*2}")
    @Email
    private String age;
    @Value("${person.date}")
    private Date date;
    private List<String> habby;
    private Map<String, String> girls;
}
```

> **@Value获取值和@ConfigurationProperties获取值比较**

|                      | @ConfigurationProperties | @Value     |
| -------------------- | ------------------------ | ---------- |
| 功能                 | 批量注入配置文件中的属性 | 一个个指定 |
| 松散绑定（松散语法） | 支持                     | 不支持     |
| SpEL                 | 不支持                   | 支持       |
| JSR303数据校验       | 支持                     | 不支持     |
| 复杂类型封装         | 支持                     | 不支持     |

配置文件yml还是properties他们都能获取到值；

如果说，我们只是在某个业务逻辑中需要获取一下配置文件中的某项值，使用@Value；

如果说，我们专门编写了一个 JavaBean 来和配置文件进行映射，我们就直接使用@ConfigurationProperties；

> **配置文件注入值数据校验**

```java
@Component
@ConfigurationProperties(prefix = "person")
@Validated
public class Person {

    /**
     * <bean class="Person">
     *      <property name="lastName" value="字面量/${key}从环境变量、配置文件中获取值/#{SpEL}"></property>
     * <bean/>
     */

   //lastName必须是邮箱格式
    @Email
    //@Value("${person.last-name}")
    private String lastName;
    //@Value("#{11*2}")
    private Integer age;
    //@Value("true")
    private Boolean boss;

    private Date birth;
    private Map<String,Object> maps;
    private List<Object> lists;
    private Dog dog;
```

## 其他注解

- **@ConfigurationProperties** 默认是从全局配置文件中获取值
- **@PropertySource(value{"classpath:person.properties"})**
    - 加载指定的配置文件
- **@ImportResource**：导入Spring的配置文件，让配置文件里面的内容生效。

@**PropertySource**：加载指定的配置文件；

```java
/**
 * 将配置文件中配置的每一个属性的值，映射到这个组件中
 * @ConfigurationProperties：告诉SpringBoot将本类中的所有属性和配置文件中相关的配置进行绑定；
 *      prefix = "person"：配置文件中哪个下面的所有属性进行一一映射
 * 只有这个组件是容器中的组件，才能容器提供的@ConfigurationProperties功能；
 *  @ConfigurationProperties(prefix = "person")默认从全局配置文件中获取值；
 *
 */
@PropertySource(value = {"classpath:person.properties"})
@Component
public class Person {
    @Value("${person.last-name}")
    private String lastName;
    @Value("#{11*2}")
    private Integer age;
    @Value("true")
    private Boolean boss;
}
```

@**ImportResource**：导入Spring的配置文件，让配置文件里面的内容生效；

Spring Boot里面没有Spring的配置文件，我们自己编写的配置文件，也不能自动识别；

想让Spring的配置文件生效，加载进来；@**ImportResource**标注在一个配置类上

```xml
-- resource下的xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans
                           https://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean class="com.xxbb.boot_01.pojo.Student" id="student"></bean>
</beans>
```

SpringBoot启动文件

```java
@ImportResource(locations = {"classpath:beans.xml"})
@SpringBootApplication
public class DemoApplication {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

}
```

SpringBoot测试文件

```java
import com.xxbb.boot_01.pojo.Person;

import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
@SpringBootTest
class DemoApplicationTests {

	@Autowired
	ApplicationContext ioc;

	@Test
	void contextLoads() {
		boolean containsBean = ioc.containsBean("student");
		System.err.println(containsBean);
	}

}
```

**SpringBoot推荐给容器中添加组件的方式；推荐使用全注解的方式**

1、配置类**@Configuration**------>Spring配置文件

2、使用**@Bean**给容器中添加组件,就这样加上注解就行了！

```java
/**
 * @Configuration：指明当前类是一个配置类；就是来替代之前的Spring配置文件
 * 在配置文件中用<bean><bean/>标签添加组件
 */
@Configuration
public class MyAppConfig {
    
    @Bean
    public HelloService helloService02(){
        //将方法的返回值添加到容器中；容器中这个组件默认的id就是方法名
        System.out.println("配置类@Bean给容器中添加组件了...");
        return new HelloService();
    }
}
```

**配置文件占位符**

RandomValuePropertySource：配置文件中可以使用随机数。yaml和properties都支持！

```java
${random.value}
${random.uuid}
${person.name:hello} // 从已有配置中获取，如果person.name没有值则用默认值hello
${random.int}
${random.long}
${random.int(10)}
${random.int[1024,65536]}

```

**占位符获取之前配置的值，如果没有可以是用:指定默认值**

```properties
person.last-name=张三${random.uuid}
person.age=${random.int}
person.birth=2017/12/15
person.boss=false
person.maps.k1=v1
person.maps.k2=14
person.lists=a,b,c
person.dog.name=${person.hello:hello}_dog
person.dog.age=15
```

----

## Profile

### 多 Profile 文件

我们在主配置文件编写的时候，文件名可以是   **application-{profile}.properties/yml**

**创建多个properties文件，默认使用application.properties的配置；**

可以在 application.properties 中使用 spring.profiles.active=xxx;

**例如：**

- application.properties
- application-dev.properties

```properties
server.port=8080
spring.profiles.active=dev
```

### yml多文档块方式

```yml
server:
    port: 9999
spring:
  profiles:
    active: dev
---
server:
    port: 8080
spring:
  profiles: dev
---
server:
  port: 8856
spring:
  profiles: prod # 指定属于那个环境 记得要用---分割开来
```

### 激活指定profile

- 在配置文件中指定  spring.profiles.active=dev
- 命令行：
    	java -jar spring-boot-02-config-0.0.1-SNAPSHOT.jar --spring.profiles.active=dev；
        	可以直接在测试的时候，配置传入命令行参数
- 虚拟机参数；
    	-Dspring.profiles.active=dev
- Program agruments中填写
    	--spring.profiles.active=dev

# Web 开发

![yuque_diagram](..\pics\Spring Boot\yuque_diagram.jpg)



## Spring MVC 自动配置概览

Spring Boot provides auto-configuration for Spring MVC that **works well with most applications.(大多场景我们都无需自定义配置)**

The auto-configuration adds the following features on top of Spring’s defaults:

- Inclusion of `ContentNegotiatingViewResolver` and `BeanNameViewResolver` beans.
    - 内容协商视图解析器和BeanName视图解析器
- Support for serving static resources, including support for WebJars (covered [later in this document](https://docs.spring.io/spring-boot/docs/current/reference/html/spring-boot-features.html#boot-features-spring-mvc-static-content))).
    - 静态资源（包括webjars）
- Automatic registration of `Converter`, `GenericConverter`, and `Formatter` beans.
    - 自动注册 `Converter，GenericConverter，Formatter `
- Support for `HttpMessageConverters` (covered [later in this document](https://docs.spring.io/spring-boot/docs/current/reference/html/spring-boot-features.html#boot-features-spring-mvc-message-converters)).
    - 支持 `HttpMessageConverters` （后来我们配合内容协商理解原理）
- Automatic registration of `MessageCodesResolver` (covered [later in this document](https://docs.spring.io/spring-boot/docs/current/reference/html/spring-boot-features.html#boot-features-spring-message-codes)).
    - 自动注册 `MessageCodesResolver` （国际化用）
- Static `index.html` support.
    - 静态index.html 页支持
- Custom `Favicon` support (covered [later in this document](https://docs.spring.io/spring-boot/docs/current/reference/html/spring-boot-features.html#boot-features-spring-mvc-favicon)).
    - 自定义 `Favicon`  
- Automatic use of a `ConfigurableWebBindingInitializer` bean (covered [later in this document](https://docs.spring.io/spring-boot/docs/current/reference/html/spring-boot-features.html#boot-features-spring-mvc-web-binding-initializer)).
    - 自动使用 `ConfigurableWebBindingInitializer` ，（DataBinder 负责将请求数据绑定到 JavaBean 上）

> If you want to keep those Spring Boot MVC customizations and make more [MVC customizations](https://docs.spring.io/spring/docs/5.2.9.RELEASE/spring-framework-reference/web.html#mvc) (interceptors, formatters, view controllers, and other features), you can add your own `@Configuration` class of type `WebMvcConfigurer` but **without** `@EnableWebMvc`.
>
> **不用@EnableWebMvc注解。使用** `@Configuration` **+** `WebMvcConfigurer` **自定义规则**

> If you want to provide custom instances of `RequestMappingHandlerMapping`, `RequestMappingHandlerAdapter`, or `ExceptionHandlerExceptionResolver`, and still keep the Spring Boot MVC customizations, you can declare a bean of type `WebMvcRegistrations` and use it to provide custom instances of those components.
>
> **声明** `WebMvcRegistrations` **改变默认底层组件**

> If you want to take complete control of Spring MVC, you can add your own `@Configuration` annotated with `@EnableWebMvc`, or alternatively add your own `@Configuration`-annotated `DelegatingWebMvcConfiguration` as described in the Javadoc of `@EnableWebMvc`.
>
> **使用** `@EnableWebMvc+@Configuration+DelegatingWebMvcConfiguration 全面接管SpringMVC`

## 简单功能分析

### 静态资源访问

#### 静态资源目录

默认情况下，Spring Boot 下的静态资源都统一放在类路径下：名为 /static (or /public or /resources or /META-INF/resources）目录下。

访问 url 为：<span style="color:green">当前项目根路径/ + 静态资源名</span>

> 原理： 静态映射/**。

请求进来，先去找 Controller 看能不能处理。不能处理的所有请求又都交给静态资源处理器。静态资源也找不到则响应 404 页面

> 改变默认的静态资源路径

```yaml
spring:
  mvc:
    static-path-pattern: /res/**

  resources:
    static-locations: [classpath:/haha/]
```

#### 静态资源访问前缀

默认无前缀，为其增加前缀后可以方便的拦截静态资源了（资源的拦截是先匹配 Controller、Controller 匹配不到再匹配静态资源）

```yaml
spring:
  mvc:
    static-path-pattern: /res/**
```

<span style="color:green">当前项目 + static-path-pattern + 静态资源名 = 静态资源文件夹下找（只是修改了下 url 的访问规则，资源的存放位置并没有改变）</span>

#### 改变默认静态资源

```yaml
spring:
	resources:
		static-locations: classpath:/haha/ # 2.4.x 无效了。
```

### 欢迎页

Spring Boot supports both static and templated welcome pages. It first looks for an **index.html file in the configured static content locations.** If one is not found, it then looks for an index template. If either is found, it is automatically used as the welcome page of the application.

- 静态资源路径下  index.html
    - 可以配置静态资源路径
    - 但是不可以配置静态资源的访问前缀。否则导致 index.html不能被默认访问

```yaml
spring:
#  mvc:
#    static-path-pattern: /res/**   这个会导致welcome page功能失效
  resources:
    static-locations: [classpath:/haha/]
```

## 静态资源配置原理

- SpringBoot启动默认加载  xxxAutoConfiguration 类（自动配置类）
- SpringMVC功能的自动配置类 WebMvcAutoConfiguration，生效

```java
@Configuration(proxyBeanMethods = false)
@ConditionalOnWebApplication(type = Type.SERVLET) // 典型的 Servlet
@ConditionalOnClass({ Servlet.class, DispatcherServlet.class, WebMvcConfigurer.class })
@ConditionalOnMissingBean(WebMvcConfigurationSupport.class) // 缺失这个 WebMvcConfigurationSupport 才生效。用户自定义了，就用自定义的，没有就用默认的。
@AutoConfigureOrder(Ordered.HIGHEST_PRECEDENCE + 10)
@AutoConfigureAfter({ DispatcherServletAutoConfiguration.class, TaskExecutionAutoConfiguration.class,
		ValidationAutoConfiguration.class })
public class WebMvcAutoConfiguration {
    
    /* 内部配置类 */
    @Configuration(proxyBeanMethods = false)
	@Import(EnableWebMvcConfiguration.class)
	@EnableConfigurationProperties({ WebMvcProperties.class,
			org.springframework.boot.autoconfigure.web.ResourceProperties.class, WebProperties.class })
	@Order(0)
	public static class WebMvcAutoConfigurationAdapter implements WebMvcConfigurer, ServletContextAware {}
}
```

> 给容器中配了什么

```java
@Configuration(proxyBeanMethods = false)
@Import(EnableWebMvcConfiguration.class)
// WebMvcProperties 
@EnableConfigurationProperties({ WebMvcProperties.class, ResourceProperties.class })
@Order(0)
public static class WebMvcAutoConfigurationAdapter implements WebMvcConfigurer {
    public WebMvcAutoConfigurationAdapter(
        org.springframework.boot.autoconfigure.web.ResourceProperties resourceProperties,
        WebProperties webProperties, WebMvcProperties mvcProperties, ListableBeanFactory beanFactory,
        ObjectProvider<HttpMessageConverters> messageConvertersProvider,
        ObjectProvider<ResourceHandlerRegistrationCustomizer> resourceHandlerRegistrationCustomizerProvider,
        ObjectProvider<DispatcherServletPath> dispatcherServletPath,
        ObjectProvider<ServletRegistrationBean<?>> servletRegistrations) {
        this.resourceProperties = resourceProperties.hasBeenCustomized() ? resourceProperties
            : webProperties.getResources();
        this.mvcProperties = mvcProperties;
        this.beanFactory = beanFactory;
        this.messageConvertersProvider = messageConvertersProvider;
        this.resourceHandlerRegistrationCustomizer = resourceHandlerRegistrationCustomizerProvider.getIfAvailable();
        this.dispatcherServletPath = dispatcherServletPath;
        this.servletRegistrations = servletRegistrations;
        this.mvcProperties.checkConfiguration();
    }

}
```

- EnableConfigurationProperties，会和配置文件的属性关联起来。
    - WebMvcProperties 和 spring.mvc 前缀相关的配置文件进行匹配。
    - 一个配置类如果只有一个有参构造器，有参构造器所有参数的值都会从容器中确定。

### 资源管理的默认规则

高版本 Spring Boot 重构了下里面的代码

```java
@Override
public void addResourceHandlers(ResourceHandlerRegistry registry) {
    if (!this.resourceProperties.isAddMappings()) { // 可以禁用静态资源的访问
        logger.debug("Default resource handling disabled");
        return;
    }
    Duration cachePeriod = this.resourceProperties.getCache().getPeriod(); // 可以获取缓存策略
    CacheControl cacheControl = this.resourceProperties.getCache().getCachecontrol().toHttpCacheControl();
    //webjars的规则
    if (!registry.hasMappingForPattern("/webjars/**")) {
        customizeResourceHandlerRegistration(registry.addResourceHandler("/webjars/**")
                                             .addResourceLocations("classpath:/META-INF/resources/webjars/")
                                             .setCachePeriod(getSeconds(cachePeriod)).setCacheControl(cacheControl));
    }

    //
    String staticPathPattern = this.mvcProperties.getStaticPathPattern();
    if (!registry.hasMappingForPattern(staticPathPattern)) {
        customizeResourceHandlerRegistration(registry.addResourceHandler(staticPathPattern) // staticPathPattern 静态资源路径，有默认值
                                             .addResourceLocations(getResourceLocations(this.resourceProperties.getStaticLocations()))
                                             .setCachePeriod(getSeconds(cachePeriod)).setCacheControl(cacheControl));
    }
}
```

### 欢迎页处理

```java
// HandlerMapping：处理器映射。保存了每一个Handler能处理哪些请求。
@Bean
public WelcomePageHandlerMapping welcomePageHandlerMapping(ApplicationContext applicationContext,
                                                           FormattingConversionService mvcConversionService, ResourceUrlProvider mvcResourceUrlProvider) {
    WelcomePageHandlerMapping welcomePageHandlerMapping = new WelcomePageHandlerMapping(
        new TemplateAvailabilityProviders(applicationContext), applicationContext, getWelcomePage(),
        this.mvcProperties.getStaticPathPattern());
    welcomePageHandlerMapping.setInterceptors(getInterceptors(mvcConversionService, mvcResourceUrlProvider));
    welcomePageHandlerMapping.setCorsConfigurations(getCorsConfigurations());
    return welcomePageHandlerMapping;
}

final class WelcomePageHandlerMapping extends AbstractUrlHandlerMapping {

	private static final Log logger = LogFactory.getLog(WelcomePageHandlerMapping.class);

	private static final List<MediaType> MEDIA_TYPES_ALL = Collections.singletonList(MediaType.ALL);

	WelcomePageHandlerMapping(TemplateAvailabilityProviders templateAvailabilityProviders,
			ApplicationContext applicationContext, Resource welcomePage, String staticPathPattern) {
		if (welcomePage != null && "/**".equals(staticPathPattern)) { // 原始静态路径没改的话，就转发到 forward:index.html
			logger.info("Adding welcome page: " + welcomePage);
			setRootViewName("forward:index.html");
		}
		else if (welcomeTemplateExists(templateAvailabilityProviders, applicationContext)) {
			logger.info("Adding welcome page template: index");
			setRootViewName("index"); // 改了就到 Controller 的 index 页面 ，看 哪个 Controller 可以处理这个页面
		}
	}
    
    private void setRootViewName(String viewName) {
        ParameterizableViewController controller = new ParameterizableViewController();
        controller.setViewName(viewName);
        setRootHandler(controller);
        setOrder(2);
	}
}
```

## 请求参数处理

### 请求映射

#### rest使用与原理

- @xxxMapping；
- Rest风格支持（使用**HTTP**请求方式动词来表示对资源的操作）
    - 以前：/getUser  获取用户    /deleteUser 删除用户   /editUser  修改用户   /saveUser  保存用户
    - 现在： /user  GET-获取用户    DELETE-删除用户    PUT-修改用户  POST-保存用户
    - 核心Filter；HiddenHttpMethodFilter
        - 用法： 表单method=post，隐藏域 _method=put
        - Spring Boot 中手动开启，`spring.mvc.hiddenmethod.filter.enabled=true`
    - 扩展：如何把_method 这个名字换成我们自己喜欢的

> 原理

```java
// 测试代码
@RestController
public class HelloController {
    @GetMapping(path = "/demo")
    public String getUser() {
        return "GET-张三";
    }

    @PutMapping(path = "/demo")
    public String saveUser() {
        return "PUT-张三";
    }
}
```

装配了一个 bean，HiddenHttpMethodFilter 需要我们手动开启

```java
@Bean
@ConditionalOnMissingBean(HiddenHttpMethodFilter.class)
@ConditionalOnProperty(prefix = "spring.mvc.hiddenmethod.filter", name = "enabled")
public OrderedHiddenHttpMethodFilter hiddenHttpMethodFilter() {
    return new OrderedHiddenHttpMethodFilter();
}
```

Rest 原理（表单提交需要使用 Rest的时候，debug 看 HiddenHttpMethodFilter 的 doFilter 方法；如果是自己实现，也可以考虑用 Filter 对请求进行拦截，判断，然后选择对应的 Controller 方法）

- 表单提交会带上 \_method=PUT
- 请求过来被 HiddenHttpMethodFilter 拦截
    - 请求是否正常，并且是 POST
        - 获取到 \_method 的值
        - 兼容以下请求：PUT、DELETE、PATCH
        - 原生 request (post)，包装模式 requestWarpper 重写了 getMethod 方法，返回的是传入的值。
        - 过滤器链放行的时候放行的 Warpper，以后的方法调用 getMethod 是调用 requestWarpper的。

```java
protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain) throws ServletException, IOException {
    HttpServletRequest requestToUse = request;
    if ("POST".equals(request.getMethod()) && request.getAttribute("javax.servlet.error.exception") == null) {
        String paramValue = request.getParameter(this.methodParam);
        if (StringUtils.hasLength(paramValue)) {
            String method = paramValue.toUpperCase(Locale.ENGLISH);
            if (ALLOWED_METHODS.contains(method)) {
                // Warpper，继承于 HttpRequestWarpper，起始就是原生的
                // HttpRequest，重写了 getMethod 方法
                requestToUse = new HiddenHttpMethodFilter.HttpMethodRequestWrapper(request, method);
            }
        }
    }

    filterChain.doFilter((ServletRequest)requestToUse, response);
}
```

> 自定义 HiddenHttpMethodFilter  可以修改 _method 变量的名称

methodParam 可以通过 set 方法进行修改

```java
@Configuration(proxyBeanMethods = false)
public class WebMvcConfig {

    @Bean
    public HiddenHttpMethodFilter hiddenHttpMethodFilter() {
        HiddenHttpMethodFilter hiddenHttpMethodFilter = new HiddenHttpMethodFilter();
        hiddenHttpMethodFilter.setMethodParam("_m");
        return hiddenHttpMethodFilter;
    }
}
```

#### 请求映射原理

请求映射原理和原先 Spring MVC 的原理是一致的。稍微用到了一点设计模式 “模板方法”。具体用到了如下几个类：

- HttpServlet：
    - service 方法
    - doGet、doPost、doPut、doDelete 等
- HttpServletBean
    - 对 HttpServlet 做一些信息的补充
- FrameworkServlet
    - 重写了 HttpServlet 的 service、doGet、doPost 等方法。重写的方法调用了 processRequest 方法，processRequest 调用了 doService（空实现）。
    - service 中调用了 基类（HttpServlet）的 service 方法
    - 而基类的 service 方法调用了 doGet、doPost 等
- DispatcherServlet
    - 实现了父类的 doService
    - doService 调用 doDispatch 方法

![image-20211023111242760](..\pics\Spring Boot\image-20211023111242760.png)

总结：Spring MVC 功能分析都从 org.springframework.web.servlet.DispatcherServlet ---> doDispatch() 开始

```java
protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
    HttpServletRequest processedRequest = request;
    HandlerExecutionChain mappedHandler = null;
    boolean multipartRequestParsed = false;
    WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);

    try {
        try {
            ModelAndView mv = null;
            Object dispatchException = null;

            try {
                processedRequest = this.checkMultipart(request);
                multipartRequestParsed = processedRequest != request;
                // 找到当前请求使用哪个 Handler（Controller的方法）处理
                mappedHandler = this.getHandler(processedRequest);
                if (mappedHandler == null) {
                    this.noHandlerFound(processedRequest, response);
                    return;
                }

                HandlerAdapter ha = this.getHandlerAdapter(mappedHandler.getHandler());
                String method = request.getMethod();
                boolean isGet = HttpMethod.GET.matches(method);
                if (isGet || HttpMethod.HEAD.matches(method)) {
                    long lastModified = ha.getLastModified(request, mappedHandler.getHandler());
                    if ((new ServletWebRequest(request, response)).checkNotModified(lastModified) && isGet) {
                        return;
                    }
                }

                if (!mappedHandler.applyPreHandle(processedRequest, response)) {
                    return;
                }

                mv = ha.handle(processedRequest, response, mappedHandler.getHandler());
                if (asyncManager.isConcurrentHandlingStarted()) {
                    return;
                }

                this.applyDefaultViewName(processedRequest, mv);
                mappedHandler.applyPostHandle(processedRequest, response, mv);
            } catch (Exception var20) {
                dispatchException = var20;
            } catch (Throwable var21) {
                dispatchException = new NestedServletException("Handler dispatch failed", var21);
            }

            this.processDispatchResult(processedRequest, response, mappedHandler, mv, (Exception)dispatchException);
        } catch (Exception var22) {
            this.triggerAfterCompletion(processedRequest, response, mappedHandler, var22);
        } catch (Throwable var23) {
            this.triggerAfterCompletion(processedRequest, response, mappedHandler, new NestedServletException("Handler processing failed", var23));
        }

    } finally {
        if (asyncManager.isConcurrentHandlingStarted()) {
            if (mappedHandler != null) {
                mappedHandler.applyAfterConcurrentHandlingStarted(processedRequest, response);
            }
        } else if (multipartRequestParsed) {
            this.cleanupMultipart(processedRequest);
        }

    }
}
```

----

getHandler 方法；RequestMappingHandlerMapping 中存储了所有@RequestMapping 和handler的映射规则。

![image-20211023114049369](..\pics\Spring Boot\image-20211023114049369.png)

![image-20211023114945974](..\pics\Spring Boot\image-20211023114945974.png)

```java
@Nullable
protected HandlerExecutionChain getHandler(HttpServletRequest request) throws Exception {
    // 一共有五个
    /*
    RequestMappingHandlerMapping
    WelcomePageHandlerMapping
    BeanNameUrlHandlerMapping
    RouterFunctionMapping
    SimpleUrlHandlerMapping
    */
    if (this.handlerMappings != null) {
        Iterator var2 = this.handlerMappings.iterator();

        while(var2.hasNext()) {
            HandlerMapping mapping = (HandlerMapping)var2.next();
            HandlerExecutionChain handler = mapping.getHandler(request);
            if (handler != null) {
                return handler;
            }
        }
    }
    return null;
}
```

所有的请求映射都在HandlerMapping中。

- Spring Boot 自动配置欢迎页的 WelcomePageHandlerMapping 。访问 /能访问到 index.html；
- Spring Boot 自动配置了默认的 RequestMappingHandlerMapping

- 请求进来，挨个尝试所有的HandlerMapping看是否有请求信息。

- - 如果有就找到这个请求对应的handler
    - 如果没有就是下一个 HandlerMapping

- 我们需要一些自定义的映射处理，我们也可以自己给容器中放**HandlerMapping**。自定义 **HandlerMapping**

```java
protected HandlerExecutionChain getHandler(HttpServletRequest request) throws Exception {
    if (this.handlerMappings != null) {
        for (HandlerMapping mapping : this.handlerMappings) {
            HandlerExecutionChain handler = mapping.getHandler(request);
            if (handler != null) {
                return handler;
            }
        }
    }
    return null;
}
```

### 普通参数与基本注解

#### 注解

- @PathVariable
- @RequestHeader
- @ModelAttribute
- @RequestParam
- @MatrixVariable：矩阵变量，需要额外配置一个属性。
- @CookieValue
- @RequestBody

```java
@RestController
public class ParameterTestController {


    //  car/2/owner/zhangsan
    @GetMapping("/car/{id}/owner/{username}")
    public Map<String,Object> getCar(@PathVariable("id") Integer id,
                                     @PathVariable("username") String name,
                                     @PathVariable Map<String,String> pv,
                                     @RequestHeader("User-Agent") String userAgent,
                                     @RequestHeader Map<String,String> header,
                                     @RequestParam("age") Integer age,
                                     @RequestParam("inters") List<String> inters,
                                     @RequestParam Map<String,String> params,
                                     @CookieValue("_ga") String _ga,
                                     @CookieValue("_ga") Cookie cookie){


        Map<String,Object> map = new HashMap<>();

//        map.put("id",id);
//        map.put("name",name);
//        map.put("pv",pv);
//        map.put("userAgent",userAgent);
//        map.put("headers",header);
        map.put("age",age);
        map.put("inters",inters);
        map.put("params",params);
        map.put("_ga",_ga);
        System.out.println(cookie.getName()+"===>"+cookie.getValue());
        return map;
    }


    @PostMapping("/save")
    public Map postMethod(@RequestBody String content){
        Map<String,Object> map = new HashMap<>();
        map.put("content",content);
        return map;
    }


    //1、语法： 请求路径：/cars/sell;low=34;brand=byd,audi,yd
    //2、SpringBoot默认是禁用了矩阵变量的功能
    //      手动开启：原理。对于路径的处理。UrlPathHelper进行解析。
    //              removeSemicolonContent（移除分号内容）支持矩阵变量的
    //3、矩阵变量必须有url路径变量才能被解析
    @GetMapping("/cars/{path}")
    public Map carsSell(@MatrixVariable("low") Integer low,
                        @MatrixVariable("brand") List<String> brand,
                        @PathVariable("path") String path){
        Map<String,Object> map = new HashMap<>();

        map.put("low",low);
        map.put("brand",brand);
        map.put("path",path);
        return map;
    }

    // /boss/1;age=20/2;age=10

    @GetMapping("/boss/{bossId}/{empId}")
    public Map boss(@MatrixVariable(value = "age",pathVar = "bossId") Integer bossAge,
                    @MatrixVariable(value = "age",pathVar = "empId") Integer empAge){
        Map<String,Object> map = new HashMap<>();

        map.put("bossAge",bossAge);
        map.put("empAge",empAge);
        return map;

    }
}
```

----

```java
@Override
// 配置矩阵参数
public void configurePathMatch(PathMatchConfigurer configurer) {
    if (this.mvcProperties.getPathmatch()
        .getMatchingStrategy() == WebMvcProperties.MatchingStrategy.PATH_PATTERN_PARSER) {
        configurer.setPatternParser(new PathPatternParser());
    }
    configurer.setUseSuffixPatternMatch(this.mvcProperties.getPathmatch().isUseSuffixPattern());
    configurer.setUseRegisteredSuffixPatternMatch(
        this.mvcProperties.getPathmatch().isUseRegisteredSuffixPattern());
    this.dispatcherServletPath.ifAvailable((dispatcherPath) -> {
        String servletUrlMapping = dispatcherPath.getServletUrlMapping();
        if (servletUrlMapping.equals("/") && singleDispatcherServlet()) {
            UrlPathHelper urlPathHelper = new UrlPathHelper();
            urlPathHelper.setAlwaysUseFullPath(true);
            configurer.setUrlPathHelper(urlPathHelper);
        }
    });
}

@Configuration(proxyBeanMethods = false)
public class WebConfig implements WebMvcCofnigure{
    @Bean
	public void configurePathMatch(PathMatchConfigurer configurer) {
		UrlPathHelper urlPathHelper = new UrlPathHelper();
        urlPathHelper.setRemoveSemicolonContent(false); // 不移除逗号分隔符，就可以启用矩阵参数解析了。
        configure.setUrlPathHelper(urlPathHelper)
    }
}
```

#### 各种类型参数解析原理

大致源码流程分析

```java
@SuppressWarnings("deprecation")
protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
    HttpServletRequest processedRequest = request;
    HandlerExecutionChain mappedHandler = null;
    boolean multipartRequestParsed = false;

    WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);

    try {
        ModelAndView mv = null;
        Exception dispatchException = null;

        try {
            processedRequest = checkMultipart(request);
            multipartRequestParsed = (processedRequest != request);

            // Determine handler for the current request.
            // 查找出当前请求对应上的处理器
            // 发现这个可以处理对应的请求 HandlerExecutionChain with [com.example.bootweb.controller.HelloController#saveUser()] and 2 interceptors
            mappedHandler = getHandler(processedRequest);
            if (mappedHandler == null) {
                noHandlerFound(processedRequest, response);
                return;
            }

            // Determine handler adapter for the current request.
            // 查找可以处理这个处理器的处理器适配器
            HandlerAdapter ha = getHandlerAdapter(mappedHandler.getHandler());

            // Process last-modified header, if supported by the handler.
            String method = request.getMethod();
            boolean isGet = HttpMethod.GET.matches(method);
            if (isGet || HttpMethod.HEAD.matches(method)) {
                long lastModified = ha.getLastModified(request, mappedHandler.getHandler());
                if (new ServletWebRequest(request, response).checkNotModified(lastModified) && isGet) {
                    return;
                }
            }

            if (!mappedHandler.applyPreHandle(processedRequest, response)) {
                return;
            }

            // Actually invoke the handler.
            // 反射工具类执行对应的方法
            mv = ha.handle(processedRequest, response, mappedHandler.getHandler());

            if (asyncManager.isConcurrentHandlingStarted()) {
                return;
            }

            applyDefaultViewName(processedRequest, mv);
            mappedHandler.applyPostHandle(processedRequest, response, mv);
        }
        catch (Exception ex) {
            dispatchException = ex;
        }
        catch (Throwable err) {
            // As of 4.3, we're processing Errors thrown from handler methods as well,
            // making them available for @ExceptionHandler methods and other scenarios.
            dispatchException = new NestedServletException("Handler dispatch failed", err);
        }
        processDispatchResult(processedRequest, response, mappedHandler, mv, dispatchException);
    }
    catch (Exception ex) {
        triggerAfterCompletion(processedRequest, response, mappedHandler, ex);
    }
    catch (Throwable err) {
        triggerAfterCompletion(processedRequest, response, mappedHandler,
                               new NestedServletException("Handler processing failed", err));
    }
    finally {
        if (asyncManager.isConcurrentHandlingStarted()) {
            // Instead of postHandle and afterCompletion
            if (mappedHandler != null) {
                mappedHandler.applyAfterConcurrentHandlingStarted(processedRequest, response);
            }
        }
        else {
            // Clean up any resources used by a multipart request.
            if (multipartRequestParsed) {
                cleanupMultipart(processedRequest);
            }
        }
    }
}
```



#### Servlet API

#### 复杂参数

#### 自定义对象参数

### POJO

### 参数处理原理

#### HandlerAdapter

- HandlerMapping 中找到能处理请求的 Handler（Controller#method）
- 为当前 Handler 找一个适配器 HandlerAdapter
    - HandlerAdapter 是 Spring 设计的一个接口，
        - supports 支持那些方法
        - handle 定义了如何处理的逻辑

![image-20211024200517771](..\pics\Spring Boot\image-20211024200517771-16350771187701.png)

#### 执行目标方法

`mv = ha.handle(processedRequest, response, mappedHandler.getHandler());`

```java
mav = invokeHandlerMethod(request, response, handlerMethod); //执行目标方法

//ServletInvocableHandlerMethod
Object returnValue = invokeForRequest(webRequest, mavContainer, providedArgs); // 真正执行目标方法
// 获取方法的参数值
Object[] args = getMethodArgumentValues(request, mavContainer, providedArgs);
```



```java
protected ModelAndView handleInternal(HttpServletRequest request,
                                      HttpServletResponse response, HandlerMethod handlerMethod) throws Exception {

    ModelAndView mav;
    checkRequest(request);

    // Execute invokeHandlerMethod in synchronized block if required.
    if (this.synchronizeOnSession) {
        HttpSession session = request.getSession(false);
        if (session != null) {
            Object mutex = WebUtils.getSessionMutex(session);
            synchronized (mutex) {
                // 执行目标方法
                mav = invokeHandlerMethod(request, response, handlerMethod);
            }
        }
        ...
    }
    ...
    return mav;
}
```

#### 参数解析器

确定将要执行的目标方法的每一个参数的值是什么;

SpringMVC目标方法能写多少种参数类型。取决于参数解析器。

![image-20211024201053219](..\pics\Spring Boot\image-20211024201053219.png)

参数解析器

```java
public interface HandlerMethodArgumentResolver {

	// 当前解析器是否支持这种解析
	boolean supportsParameter(MethodParameter parameter);

	// 进行参数解析
	@Nullable
	Object resolveArgument(MethodParameter parameter, @Nullable ModelAndViewContainer mavContainer,
			NativeWebRequest webRequest, @Nullable WebDataBinderFactory binderFactory) throws Exception;

}
```

#### 返回值处理器

![image-20211024201539502](..\pics\Spring Boot\image-20211024201539502.png)

### 确定目标方法每一个参数值

#### 解析器

#### 解析参数

#### 自定义类型参数

### 目标方法执行完成

### 处理派发结果



# 日志

## 日志框架

> 市面上的日志框架；

JUL、JCL、Jboss-logging、logback、log4j、log4j2、slf4j ....

| 日志门面  （日志的抽象层）                                   | 日志实现                                             |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| ~~JCL（Jakarta  Commons Logging）~~    SLF4j（Simple  Logging Facade for Java）    **~~jboss-logging~~** | Log4j  JUL（java.util.logging）  Log4j2  **Logback** |

左边选一个门面（抽象层）、右边来选一个实现；

- 日志门面：  SLF4J；
- 日志实现：Logback；

SpringBoot：底层是 Spring 框架，Spring 框架默认是用 JCL；Spring Boot 选用 SLF4j 和 logback；

## SLF4j

### 如使用SLF4j   

https://www.slf4j.org

以后开发的时候，日志记录方法的调用，不应该来直接调用日志的实现类，而是调用日志抽象层里面的方法；

> 给系统里面导入 slf4j 的 jar 和  logback的实现 jar

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HelloWorld {
  public static void main(String[] args) {
    Logger logger = LoggerFactory.getLogger(HelloWorld.class);
    logger.info("Hello World");
  }
}
```

每一个日志的实现框架都有自己的配置文件。使用slf4j以后，**配置文件还是做成日志实现框架自己本身的配置文件；**

### 遗留问题

a（slf4j+logback）: Spring（commons-logging）、Hibernate（jboss-logging）、MyBatis、xxxx

统一日志记录，即使是别的框架和我一起统一使用slf4j进行输出？

**如何让系统中所有的日志都统一到slf4j；**

- 将系统中其他日志框架先排除出去；
- 用中间包来替换原有的日志框架；
- 我们导入slf4j其他的实现

## 日志关系

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter</artifactId>
</dependency>
```

SpringBoot使用它来做日志功能；

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-logging</artifactId>
</dependency>
```

底层依赖关系

- Spring Boot 底层也是使用 slf4j+logback 的方式进行日志记录
- Spring Boot 也把其他的日志都替换成了 slf4j；
- 如果我们要引入其他框架？一定要把这个框架的默认日志依赖移除掉？

比如，Spring 用的 commons-logging，那么我们在 Boot 里面就需要这样

```xml
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-core</artifactId>
    <exclusions>
        <exclusion>
            <groupId>commons-logging</groupId>
            <artifactId>commons-logging</artifactId>
        </exclusion>
    </exclusions>
</dependency>
```

**SpringBoot能自动适配所有的日志，而且底层使用slf4j+logback的方式记录日志，引入其他框架的时候，只需要把这个框架依赖的日志框架排除掉即可；**

## 日志使用；

### 默认配置

SpringBoot默认帮我们配置好了日志；

```java
//记录器
Logger logger = LoggerFactory.getLogger(getClass());
@Test
public void contextLoads() {
    //System.out.println();

    //日志的级别；
    //由低到高   trace<debug<info<warn<error
    //可以调整输出的日志级别；日志就只会在这个级别以以后的高级别生效
    logger.trace("这是trace日志...");
    logger.debug("这是debug日志...");
    //SpringBoot默认给我们使用的是info级别的，没有指定级别的就用SpringBoot默认规定的级别；root级别
    logger.info("这是info日志...");
    logger.warn("这是warn日志...");
    logger.error("这是error日志...");

}
```

```xml
<!--
    日志输出格式：
		%d表示日期时间，
		%thread表示线程名，
		%-5level：级别从左显示5个字符宽度
		%logger{50} 表示logger名字最长50个字符，否则按照句点分割。 
		%msg：日志消息，
		%n是换行符
-->
    %d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] %-5level %logger{50} - %msg%n
```

SpringBoot修改日志的默认配置

```properties
logging.level.com=trace # 设置日志记录的等级。这句话是记录com包下的日志，日志等级为trace
logging.file.name=springboot.log # 输出的日志的名称是springboot.log

#logging.path=
# 不指定路径在当前项目下生成springboot.log日志
# 可以指定完整的路径；
#logging.file.path==G:/springboot.log

# 在当前磁盘的根路径下创建spring文件夹和里面的log文件夹；使用 spring.log 作为默认文件
logging.file.path==/spring/log

#  在控制台输出的日志的格式
logging.pattern.console=%d{yyyy-MM-dd} [%thread] %-5level %logger{50} - %msg%n
# 指定文件中日志输出的格式
logging.pattern.file=%d{yyyy-MM-dd} === [%thread] === %-5level === %logger{50} ==== %msg%n
```

| logging.file | logging.path | Example  | Description                        |
| ------------ | ------------ | -------- | ---------------------------------- |
| (none)       | (none)       |          | 只在控制台输出                     |
| 指定文件名   | (none)       | my.log   | 输出日志到my.log文件               |
| (none)       | 指定目录     | /var/log | 输出到指定目录的 spring.log 文件中 |

### 指定配置

给类路径下放上每个日志框架自己的配置文件即可；Spring Boot 就不使用他默认配置的了

| Logging System          | Customization                                                |
| ----------------------- | ------------------------------------------------------------ |
| Logback                 | `logback-spring.xml`, `logback-spring.groovy`, `logback.xml` or `logback.groovy` |
| Log4j2                  | `log4j2-spring.xml` or `log4j2.xml`                          |
| JDK (Java Util Logging) | `logging.properties`                                         |

logback.xml：直接就被日志框架识别了；

**logback-spring.xml**：日志框架就不直接加载日志的配置项，由 Spring Boot 解析日志配置，可以使用 Spring Boot 的高级Profile功能

```xml
<springProfile name="staging">
    <!-- configuration to be enabled when the "staging" profile is active -->
  	可以指定某段配置只在某个环境下生效
</springProfile>
```

如：

```xml
<appender name="stdout" class="ch.qos.logback.core.ConsoleAppender">
<!--
   日志输出格式：
   %d表示日期时间，
   %thread表示线程名，
   %-5level：级别从左显示5个字符宽度
   %logger{50} 表示logger名字最长50个字符，否则按照句点分割。 
   %msg：日志消息，
   %n是换行符
-->
    <layout class="ch.qos.logback.classic.PatternLayout">
        <springProfile name="dev">
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} ----> [%thread] ---> %-5level %logger{50} - %msg%n</pattern>
        </springProfile>
        <springProfile name="!dev">
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} ==== [%thread] ==== %-5level %logger{50} - %msg%n</pattern>
        </springProfile>
    </layout>
</appender>
```

如果使用logback.xml作为日志配置文件，还要使用profile功能，会有以下错误

 `no applicable action for [springProfile]`

## 切换日志框架

可以按照slf4j的日志适配图，进行相关的切换；

slf4j+log4j的方式；

```xml
<dependency>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-starter-web</artifactId>
  <exclusions>
    <exclusion>
      <artifactId>logback-classic</artifactId>
      <groupId>ch.qos.logback</groupId>
    </exclusion>
    <exclusion>
      <artifactId>log4j-over-slf4j</artifactId>
      <groupId>org.slf4j</groupId>
    </exclusion>
  </exclusions>
</dependency>

<dependency>
  <groupId>org.slf4j</groupId>
  <artifactId>slf4j-log4j12</artifactId>
</dependency>
```

切换为log4j2

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
    <exclusions>
        <exclusion>
            <artifactId>spring-boot-starter-logging</artifactId>
            <groupId>org.springframework.boot</groupId>
        </exclusion>
    </exclusions>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-log4j2</artifactId>
</dependency>
```

# Docker

## 简介

**Docker**是一个开源的应用容器引擎；是一个轻量级容器技术；

Docker支持将软件编译成一个镜像；然后在镜像中各种软件做好配置，将镜像发布出去，其他使用者可以直接使用这个镜像；

运行中的这个镜像称为容器，容器启动是非常快速的。

## 核心概念

docker主机(Host)：安装了Docker程序的机器（Docker直接安装在操作系统之上）；

docker客户端(Client)：连接docker主机进行操作；

docker仓库(Registry)：用来保存各种打包好的软件镜像；

docker镜像(Images)：软件打包好的镜像；放在docker仓库中；

docker容器(Container)：镜像启动后的实例称为一个容器；容器是独立运行的一个或一组应用

> 使用Docker的步骤：

- 安装Docker
- 去Docker仓库找到这个软件对应的镜像；
- 使用Docker运行这个镜像，这个镜像就会生成一个Docker容器；
- 对容器的启动停止就是对软件的启动停止；

## Linux安装Docker

```shell
1、检查内核版本，必须是3.10及以上
uname -r
2、安装docker
yum install docker
3、输入y确认安装
4、启动docker
[root@localhost ~]# systemctl start docker
[root@localhost ~]# docker -v
Docker version 1.12.6, build 3e8e77d/1.12.6
5、开机启动docker
[root@localhost ~]# systemctl enable docker
Created symlink from /etc/systemd/system/multi-user.target.wants/docker.service to /usr/lib/systemd/system/docker.service.
6、停止docker
systemctl stop docker
```

## Docker常用命令&操作

### 镜像操作

| 操作 | 命令                                            | 说明                                                     |
| ---- | ----------------------------------------------- | -------------------------------------------------------- |
| 检索 | docker  search 关键字  eg：docker  search redis | 我们经常去docker  hub上检索镜像的详细信息，如镜像的TAG。 |
| 拉取 | docker pull 镜像名:tag                          | :tag是可选的，tag表示标签，多为软件的版本，默认是latest  |
| 列表 | docker images                                   | 查看所有本地镜像                                         |
| 删除 | docker rmi image-id                             | 删除指定的本地镜像                                       |

https://hub.docker.com/

### 容器操作

软件镜像（QQ安装程序）----> 运行镜像 ----> 产生一个容器（正在运行的软件，运行的QQ）；

````shell
1、搜索镜像
[root@localhost ~]# docker search tomcat
2、拉取镜像
[root@localhost ~]# docker pull tomcat
3、根据镜像启动容器
docker run --name mytomcat -d tomcat:latest
4、docker ps  
查看运行中的容器
5、 停止运行中的容器
docker stop  容器的id
6、查看所有的容器
docker ps -a
7、启动容器
docker start 容器id
8、删除一个容器
 docker rm 容器id
9、启动一个做了端口映射的tomcat
[root@localhost ~]# docker run -d -p 8888:8080 tomcat
-d：后台运行
-p: 将主机的端口映射到容器的一个端口    主机端口:容器内部的端口

10、为了演示简单关闭了linux的防火墙
service firewalld status ；查看防火墙状态
service firewalld stop：关闭防火墙
11、查看容器的日志
docker logs container-name/container-id

更多命令参看
https://docs.docker.com/engine/reference/commandline/docker/
可以参考每一个镜像的文档
````

### 安装MySQL示例

```shell
docker pull mysql
```

错误的启动

```shell
[root@localhost ~]# docker run --name mysql01 -d mysql
42f09819908bb72dd99ae19e792e0a5d03c48638421fa64cce5f8ba0f40f5846

mysql退出了
[root@localhost ~]# docker ps -a
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS                           PORTS               NAMES
42f09819908b        mysql               "docker-entrypoint.sh"   34 seconds ago      Exited (1) 33 seconds ago                            mysql01
538bde63e500        tomcat              "catalina.sh run"        About an hour ago   Exited (143) About an hour ago                       compassionate_
goldstine
c4f1ac60b3fc        tomcat              "catalina.sh run"        About an hour ago   Exited (143) About an hour ago                       lonely_fermi
81ec743a5271        tomcat              "catalina.sh run"        About an hour ago   Exited (143) About an hour ago                       sick_ramanujan


//错误日志
[root@localhost ~]# docker logs 42f09819908b
error: database is uninitialized and password option is not specified 
  You need to specify one of MYSQL_ROOT_PASSWORD, MYSQL_ALLOW_EMPTY_PASSWORD and MYSQL_RANDOM_ROOT_PASSWORD；这个三个参数必须指定一个
```

正确的启动

```shell
[root@localhost ~]# docker run --name mysql01 -e MYSQL_ROOT_PASSWORD=123456 -d mysql
b874c56bec49fb43024b3805ab51e9097da779f2f572c22c695305dedd684c5f
[root@localhost ~]# docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS               NAMES
b874c56bec49        mysql               "docker-entrypoint.sh"   4 seconds ago       Up 3 seconds        3306/tcp            mysql01
```

做了端口映射

```shell
[root@localhost ~]# docker run -p 3306:3306 --name mysql02 -e MYSQL_ROOT_PASSWORD=123456 -d mysql
ad10e4bc5c6a0f61cbad43898de71d366117d120e39db651844c0e73863b9434
[root@localhost ~]# docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS              PORTS                    NAMES
ad10e4bc5c6a        mysql               "docker-entrypoint.sh"   4 seconds ago       Up 2 seconds        0.0.0.0:3306->3306/tcp   mysql02
```

几个其他的高级操作

```
docker run --name mysql03 -v /conf/mysql:/etc/mysql/conf.d -e MYSQL_ROOT_PASSWORD=my-secret-pw -d mysql:tag
把主机的/conf/mysql文件夹挂载到 mysqldocker容器的/etc/mysql/conf.d文件夹里面
改mysql的配置文件就只需要把mysql配置文件放在自定义的文件夹下（/conf/mysql）

docker run --name some-mysql -e MYSQL_ROOT_PASSWORD=my-secret-pw -d mysql:tag --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci
指定mysql的一些配置参数
```

# 数据访问

## JDBC

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-jdbc</artifactId>
</dependency>
<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <scope>runtime</scope>
</dependency>
```

```yaml
spring:
  datasource:
    username: root
    password: 123456
    url: jdbc:mysql://192.168.15.22:3306/jdbc
    driver-class-name: com.mysql.jdbc.Driver
```

> **效果：**

默认是用org.apache.tomcat.jdbc.pool.DataSource作为数据源；

数据源的相关配置都在DataSourceProperties里面；

> **自动配置原理：**

org.springframework.boot.autoconfigure.jdbc：

1、参考DataSourceConfiguration，根据配置创建数据源，默认使用Tomcat连接池；可以使用spring.datasource.type指定自定义的数据源类型；

2、SpringBoot默认可以支持；

org.apache.tomcat.jdbc.pool.DataSource、HikariDataSource、BasicDataSource、

3、自定义数据源类型

```java
/**
 * Generic DataSource configuration.
 */
@ConditionalOnMissingBean(DataSource.class)
@ConditionalOnProperty(name = "spring.datasource.type")
static class Generic {

   @Bean
   public DataSource dataSource(DataSourceProperties properties) {
       //使用DataSourceBuilder创建数据源，利用反射创建响应type的数据源，并且绑定相关属性
      return properties.initializeDataSourceBuilder().build();
   }

}
```

4、**DataSourceInitializer：ApplicationListener**；

	作用：
	
		1）、runSchemaScripts();运行建表语句；
	
		2）、runDataScripts();运行插入数据的sql语句；

默认只需要将文件命名为：

```properties
schema-*.sql、data-*.sql
默认规则：schema.sql，schema-all.sql；
可以使用   
	schema:
      - classpath:department.sql
      指定位置
```

5、操作数据库：自动配置了JdbcTemplate操作数据库

## 整合 Druid 数据源

这个配置文件的前缀一定要一致！

```properties
# 禁用缓存
spring.thymeleaf.cache=false
# 基础的jdbc配置
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.url=jdbc:mysql:///blog
spring.datasource.driver-class-name=com.mysql.jdbc.Driver
spring.datasource.type=com.alibaba.druid.pool.DruidDataSource
# druid相关配置
spring.datasource.initialSize=5
spring.datasource.minIdle=5
spring.datasource.maxActive=20
spring.datasource.maxWait=60000
spring.datasource.timeBetweenEvictionRunsMillis=60000
spring.datasource.minEvictableIdleTimeMillis=300000
spring.datasource.validationQuery=SELECT 1 FROM DUAL
spring.datasource.testWhileIdle=true
spring.datasource.testOnBorrow=false
spring.datasource.testOnReturn=false
spring.datasource.poolPreparedStatements=true
#   配置监控统计拦截的filters，去掉后监控界面sql无法统计，'wall'用于防火墙
spring.datasource.filters=stat,wall,log4j
spring.datasource.maxPoolPreparedStatementPerConnectionSize=20
spring.datasource.useGlobalDataSourceStat=true
spring.datasource.connectionProperties=druid.stat.mergeSql=true;druid.stat.slowSqlMillis=500
```

```java
@Configuration
public class DruidConfig {

    @ConfigurationProperties(prefix = "spring.datasource")
    @Bean
    public DataSource getDatasource() {
        return new DruidDataSource();
    }

    @Bean
    // 注册servlet管理druid
    public ServletRegistrationBean tatViewServlet() {
        // 注册那个servlet 管理那些url请求
        ServletRegistrationBean bean = new ServletRegistrationBean(new StatViewServlet(), "/druid/*");
        Map<String, String> init = new HashMap();
        init.put("loginUsername", "root");
        init.put("loginPassword", "root");
        init.put("allow", "");
        bean.setInitParameters(init);
        return bean;
    }

    @Bean
    // 注册过滤器
    public FilterRegistrationBean webStatFilter() {
        FilterRegistrationBean bean = new FilterRegistrationBean(new WebStatFilter());

        Map<String, String> init = new HashMap();
        init.put("exclusions", "*.js,*.css,/druid/*");

        bean.setInitParameters(init);

        bean.setUrlPatterns(Arrays.asList("/*"));
        return bean;
    }

}
```

## 整合 MyBatis

```xml
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>2.1.2</version>
</dependency>
```

步骤：

- 配置数据源相关属性（见上一节Druid）
- 给数据库建表
- 创建JavaBean

### 	注解版

```java
//指定这是一个操作数据库的mapper
@Mapper
public interface DepartmentMapper {

    @Select("select * from department where id=#{id}")
    public Department getDeptById(Integer id);

    @Delete("delete from department where id=#{id}")
    public int deleteDeptById(Integer id);

    @Options(useGeneratedKeys = true,keyProperty = "id")
    @Insert("insert into department(departmentName) values(#{departmentName})")
    public int insertDept(Department department);

    @Update("update department set departmentName=#{departmentName} where id=#{id}")
    public int updateDept(Department department);
}
```

> 问题：自定义MyBatis的配置规则；给容器中添加一个ConfigurationCustomizer；

```java
@org.springframework.context.annotation.Configuration
public class MyBatisConfig {

    @Bean
    public ConfigurationCustomizer configurationCustomizer(){
        return new ConfigurationCustomizer(){

            @Override
            public void customize(Configuration configuration) {
                configuration.setMapUnderscoreToCamelCase(true);
            }
        };
    }
}
```

```java
// 使用MapperScan批量扫描所有的Mapper接口；
@MapperScan(value = "com.atguigu.springboot.mapper")
@SpringBootApplication
public class SpringBoot06DataMybatisApplication {

	public static void main(String[] args) {
		SpringApplication.run(SpringBoot06DataMybatisApplication.class, args);
	}
}
```

### 配置文件版

```yaml
mybatis:
  config-location: classpath:mybatis/mybatis-config.xml # 指定全局配置文件的位置
  mapper-locations: classpath:mybatis/mapper/*.xml  # 指定sql映射文件的位置
```

更多使用参照

http://www.mybatis.org/spring-boot-starter/mybatis-spring-boot-autoconfigure/

## 整合SpringData JPA

### SpringData JPA

JPA:ORM（Object Relational Mapping）；

编写一个实体类（bean）和数据表进行映射，并且配置好映射关系；

```java
//使用JPA注解配置映射关系
@Entity //告诉JPA这是一个实体类（和数据表映射的类）
@Table(name = "tbl_user") //@Table来指定和哪个数据表对应;如果省略默认表名就是user；
public class User {

    @Id //这是一个主键
    @GeneratedValue(strategy = GenerationType.IDENTITY)//自增主键
    private Integer id;

    @Column(name = "last_name",length = 50) //这是和数据表对应的一个列
    private String lastName;
    @Column //省略默认列名就是属性名
    private String email;
```

编写一个Dao接口来操作实体类对应的数据表（Repository）

```java
//继承JpaRepository来完成对数据库的操作
public interface UserRepository extends JpaRepository<User,Integer> {
}

```

基本的配置JpaProperties

```yaml
spring:  
 jpa:
    hibernate:
#     更新或者创建数据表结构
      ddl-auto: update
#    控制台显示SQL
    show-sql: true
```

# SpringBoot整合示例

https://github.com/spring-projects/spring-boot/tree/master/spring-boot-samples

# 其他

ymal 读数据

- Java代码中 读数组 `${xx[1]}$`
- 使用 `${属性名}$` 引用 在 yml 种引用数据
    - 如果用引号包裹起来了，会进行转义，比如  \t 被识别为水平制表符
- yml 引用对象赋值
    - @Component + @ConfigurationProperties
    - @EnableConfigurationProperties

## 整合 Junit

Spring + Junit 需要指定引导类

-  @RunWith(设置运行器) + @ContextConfiguration