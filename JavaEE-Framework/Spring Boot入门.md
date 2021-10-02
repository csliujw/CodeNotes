# 快速入门

## Spring与SpringBoot

### Spring

### SpringBoot

能快速创建出生产级别的Spring应用

#### 优点

- Create stand-alone Spring applications
    - 创建独立 Spring 应用
- Embed Tomcat, Jetty or Undertow directly (no need to deploy WAR files)
    - 内嵌web服务器
- Provide opinionated 'starter' dependencies to simplify your build configuration
    - 自动starter依赖，简化构建配置
- Automatically configure Spring and 3rd party libraries whenever possible
    - 自动配置 Spring 以及第三方功能
- Provide production-ready features such as metrics, health checks, and externalized configuration
    - 提供生产级别的监控、健康检查及外部化配置
- Absolutely no code generation and no requirement for XML configuration
    - 无代码生成、无需编写 XML

Spring Boot 是整合 Spring 技术栈的一站式框架，是简化 Spring 技术栈的快速开发脚手架。

####  缺点

- 迭代快，需要时刻关注变化

- 封装太深，内部原理复杂，不易精通

### 时代背景

#### 微服务

[James Lewis and Martin Fowler (2014)](https://martinfowler.com/articles/microservices.html)  提出微服务完整概念。https://martinfowler.com/microservices/

> In short, the **microservice architectural style** is an approach to developing a single application as a **suite of small services**, each **running in its own process** and communicating with **lightweight** mechanisms, often an **HTTP** resource API. These services are **built around business capabilities** and **independently deployable** by fully **automated deployment** machinery. There is a **bare minimum of centralized management** of these services, which may be **written in different programming languages** and use different data storage technologies.-- [James Lewis and Martin Fowler (2014)](https://martinfowler.com/articles/microservices.html)

- 微服务是一种架构风格
- 一个应用拆分为一组小型服务
- 每个服务运行在自己的进程内，也就是可独立部署和升级
- 服务之间使用轻量级HTTP交互
- 服务围绕业务功能拆分
- 可以由全自动部署机制独立部署
- 去中心化，服务自治。服务可以使用不同的语言、不同的存储技术

#### 分布式

![image](https://cdn.nlark.com/yuque/0/2020/png/1613913/1599562347965-a617a866-4270-44e9-9c5b-ced552683eda.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_14%2Ctext_YXRndWlndS5jb20g5bCa56GF6LC3%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10)

> 分布式的困难

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

> 分布式的解决

- SpringBoot + SpringCloud

![image](https://cdn.nlark.com/yuque/0/2020/png/1613913/1599799119457-841ef47a-6585-4ca4-8e3d-8298e796012c.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_10%2Ctext_YXRndWlndS5jb20g5bCa56GF6LC3%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10)

#### 云原生

原生应用如何上云。 Cloud Native

> 上云的困难

- 服务自愈
- 弹性伸缩
- 服务隔离
- 自动化部署
- 灰度发布
- 流量治理
- ......

## 创建SpringBoot项目

### IDEA 创建项目

使用Spring Initializer快速创建，直接选择它，不选择maven。

给maven 的settings.xml配置文件的profiles标签添加

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

编写一个主程序；启动Spring Boot应用

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

编写相关的Controller、Service

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

运行主程序测试

> 简化部署

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

将这个应用打成 jar 包，直接使用 java -jar 的命令进行执行；

打成 jar 包可以直接用 maven 的 package 命令或者控制台 mvn package

使用 `mvn spring-boot:run` 启动

<span style="color:red">**注意：**如果引入了JDBC相关的依赖，需要我们在配置文件中配置连接数据库相关的操作。</span>

### 官方骨架创建

> 官方骨架创建遇到的问题

- 有时候 IDEA 创建项目总是出错，于是直接取官网选好依赖，下载过来，导入到IDEA中。

- 导入后用 maven 的 Reload All Maven Projects 导入所有的依赖，导入后发现下面这个配置文件报错

```xml
<plugin>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-maven-plugin</artifactId>
</plugin>  
```

加个对应的版本号就可以了

```xml
  <plugin>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-maven-plugin</artifactId>
      <version>2.4.2</version>
  </plugin>  
```

### 目录结构说明

> SpringBoot目录解析

- resources文件夹中目录结构
    - static：保存静态资源；js css img
    - templates：保存页面资源；springboot默认不支持jsp
    - mybatis 的配置文件之类的需要放在resources文件夹下面。resources是资源的根路径。就把resources当成编译后的classes文件夹吧。

resources下的文件最终都会被部署到classpath文件下

### 启动类&热部署设置

> 启动类说明

```java
@SpringBootApplication // 标明这是一个引导类。也可以不在这里的，没事。 
@MapperScan("cn.baobaoxuxu.community.mapper") // 配置MyBatis的Mapper扫描
public class CommunityApplication {
    // 程序的启动入口。
    public static void main(String[] args) {
        SpringApplication.run(CommunityApplication.class, args);
    }
}
```

- 热部署(IDEA进行SpringBoot热部署失败的原因是，IDEA默认情况下不会自动编译，需要对IDEA进行自动编译的设置)
  - Settings -->Compiler
  - Ctrl + Shift + Alt + / -->选择Registry-->compiler.automake.allow.when.app.running ✔

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-devtools</artifactId>
    <scope>runtime</scope>
    <optional>true</optional>
</dependency>
```

## SpringBoot原理分析

父坐标parent

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.4.2</version>
    <relativePath/> <!-- lookup parent from repository -->
</parent>
```

parent的父坐标 dependencies，这里面都是当前版本的 Spring Boot 所集成的一些依赖

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-dependencies</artifactId>
    <version>2.4.2</version>
</parent>
```

## yml语法

通过空格隔开key和value的

```yaml
# 普通数据配置
name: zhangsan

# 对象配置
person:
    name: zhangsan
    age: 13

# 行内对象配置
person2: {name: zhangsan}

# 配置数据、集合
city:
    - beijing
    - tianjin
    - chongqing

city2: [beijing,tianjin]

# 配置数据、集合（对象数据）
student:
    - name: tom
      age: 18
      addr: beijing
    - name: lucy
      age: 19
      addr: nanchang
student2: [{name: tom,age: 18},{name: tom2,age: 18}] 

# map配置
map:
    key1: value1
    key2: value2
```

## 获取配置数据

使用@Value映射

```java
package cn.baobaoxuxu.community.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController 
// 加了这个配置文件，会自动从配置文件中，把person对应的属性字段赋值完毕。
// 字段要一致才能匹配成功~~
@ConfigurationProperties(prefix = "person") 
public class YamlController {
    // Spring的表达式语言
    
    @Value("${name}")
    private String name;

    @GetMapping("/name")
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setPersonName(String personName) {
        this.personName = personName;
    }
}
```

## 配置信息的加载顺序

- 先加载 yml
- 再加载 yaml
- 再加载 properties
- <span style="color:red">后加载的覆盖先加载的哦~<

```xml
<resource>
    <directory>${basedir}/src/main/resources</directory>
    <filtering>true</filtering>
    <includes>
        <include>**/application*.yml</include>
        <include>**/application*.yaml</include>
        <include>**/application*.properties</include>
    </includes>
</resource>
```

# 自动配置原理

## 概述

Spring Boot 父项目做依赖管理，配置了常用的三方 jar 的版本。

Spring Boot 注解的层次关系，从最开始指定启动类那个 `@SpringBootApplication` 开始.

`@SpringBootApplication` 上有这些注解

- `@SpringBootConfiguration`
    - @Configuration,不就是Spring的那个注解吗？就是指定了他是配置类！！
- `@EnableAutoConfiguration`，开启自动配置
    - `@AutoConfigurationPackage`
    - `@Import(AutoConfigurationImportSelector.class)`，Import的作用是当前配置文件引入其他配置类
    - `AutoConfigurationImportSelector` 这个类很重要。单独拎出来讲！
- `@ComponentScan`，组件扫描。有这个注解的，会以加了这个注解的类所在的包为基础路径，进行类的扫描。`@SpringBootApplication` 类上打了`@ComponentScan` 注解，相当于 `@SpringBootApplication` 也有这个扫描的功能。

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(excludeFilters = { @Filter(type = FilterType.CUSTOM, classes = TypeExcludeFilter.class),
      @Filter(type = FilterType.CUSTOM, classes = AutoConfigurationExcludeFilter.class) })
public @interface SpringBootApplication {
}
```

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@AutoConfigurationPackage
@Import(AutoConfigurationImportSelector.class)
public @interface EnableAutoConfiguration {
}
```

```java
public class AutoConfigurationImportSelector implements DeferredImportSelector, BeanClassLoaderAware,
      ResourceLoaderAware, BeanFactoryAware, EnvironmentAware, Ordered {

   @Override
   public String[] selectImports(AnnotationMetadata annotationMetadata) {
      if (!isEnabled(annotationMetadata)) {
         return NO_IMPORTS;
      }
      AutoConfigurationEntry autoConfigurationEntry = getAutoConfigurationEntry(annotationMetadata);
      return StringUtils.toStringArray(autoConfigurationEntry.getConfigurations());
   }

    // getAutoConfigurationEntry方法
   protected AutoConfigurationEntry getAutoConfigurationEntry(AnnotationMetadata annotationMetadata) {
      if (!isEnabled(annotationMetadata)) {
         return EMPTY_ENTRY;
      }
      AnnotationAttributes attributes = getAttributes(annotationMetadata);
        // 加载某些配置。应该是一个全包名。
      List<String> configurations = getCandidateConfigurations(annotationMetadata, attributes);
      configurations = removeDuplicates(configurations);
      Set<String> exclusions = getExclusions(annotationMetadata, attributes);
      checkExcludedClasses(configurations, exclusions);
      configurations.removeAll(exclusions);
      configurations = getConfigurationClassFilter().filter(configurations);
      fireAutoConfigurationImportEvents(configurations, exclusions);
      return new AutoConfigurationEntry(configurations, exclusions);
   }

    // getCandidateConfigurations
   protected List<String> getCandidateConfigurations(AnnotationMetadata metadata, AnnotationAttributes attributes) {
      List<String> configurations = SpringFactoriesLoader.loadFactoryNames(getSpringFactoriesLoaderFactoryClass(),
            getBeanClassLoader());
        // META-INF/spring.factories  一般这个META-INF是当前类所在的那个jar包的META-INF下
        // 我们去看META-INF下的这个类org.springframework.boot.autoconfigure.web.servlet.ServletWebServerFactoryAutoConfiguration,\
      // 找这个类的注解 @EnableConfigurationProperties(ServerProperties.class)
        // 点进这个类 ServerProperties
        // 定位到 @ConfigurationProperties(prefix = "server", ignoreUnknownFields = true)
        // public class ServerProperties { }
        // 结论 去 spring-configuration-metadata.json里找配置信息！
        Assert.notEmpty(configurations, "No auto configuration classes found in META-INF/spring.factories. If you "
            + "are using a custom packaging, make sure that file is correct.");
      return configurations;
   }

}
```

## 特点

> 依赖管理

父项目做依赖管理

```xml
<!-- 依赖管理 -->    
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.3.4.RELEASE</version>
</parent>

<!-- 他的父项目 -->   
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-dependencies</artifactId>
    <version>2.3.4.RELEASE</version>
</parent>

<!-- 几乎声明了所有开发中常用的依赖的版本号,自动版本仲裁机制 -->   
```

开发导入 starter 场景启动器

- 见到很多 spring-boot-starter-* ： *就某种场景
- 只要引入starter，这个场景的所有常规需要的依赖我们都自动引入
- SpringBoot所有支持的场景
    https://docs.spring.io/spring-boot/docs/current/reference/html/using-spring-boot.html#using-boot-starter
- 见到的  *-spring-boot-starter： 第三方为我们提供的简化开发的场景启动器。
- 所有场景启动器最底层的依赖

```xml
<dependency>
  <groupId>org.springframework.boot</groupId>
  <artifactId>spring-boot-starter</artifactId>
  <version>2.3.4.RELEASE</version>
  <scope>compile</scope>
</dependency>
```

无需关注版本号，自动版本仲裁

- 引入依赖默认都可以不写版本
- 引入非版本仲裁的jar，要写版本号。

可以修改默认版本号

- 查看spring-boot-dependencies里面规定当前依赖的版本 用的 key。
- 在当前项目里面重写配置

```xml
<properties>
	<mysql.version>5.1.43</mysql.version>
</properties>
```

## 自动配置举例

- 自动配好Tomcat

    - 引入Tomcat依赖。
    - 配置Tomcat

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-tomcat</artifactId>
    <version>2.3.4.RELEASE</version>
    <scope>compile</scope>
</dependency>
```

- 自动配好SpringMVC

    - 引入SpringMVC全套组件
    - 自动配好SpringMVC常用组件（功能）

- 自动配好Web常见功能，如：字符编码问题

    - SpringBoot帮我们配置好了所有web开发的常见场景

- 默认的包结构

    - 主程序所在包及其下面的所有子包里面的组件都会被默认扫描进来
    - 无需以前的包扫描配置
    - 想要改变扫描路径，@SpringBootApplication(scanBasePackages=**"com.xx"**
    - 或者@ComponentScan 指定扫描路径

```java
@SpringBootApplication
 // 等同于
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan("com.xx.boot")
```

- 各种配置拥有默认值

    - 默认配置最终都是映射到某个类上，如：MultipartProperties
    - 配置文件的值最终会绑定每个类上，这个类会在容器中创建对象

- 按需加载所有自动配置项

- 非常多的starter
- 引入了哪些场景这个场景的自动配置才会开启
- SpringBoot所有的自动配置功能都在 spring-boot-autoconfigure 包里面
- ......

## 开发小技巧

### Lombok

简化JavaBean开发

```java
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
        </dependency>


idea中搜索安装lombok插件
===============================简化JavaBean开发===================================
@NoArgsConstructor
//@AllArgsConstructor
@Data
@ToString
@EqualsAndHashCode
public class User {

    private String name;
    private Integer age;

    private Pet pet;

    public User(String name,Integer age){
        this.name = name;
        this.age = age;
    }


}



================================简化日志开发===================================
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

项目或者页面修改以后：Ctrl+F9；

# 整合JDBC

## 概述

SpringBoot整合JDBC只要引入对应数据库的连接驱动包（如mysql的驱动），然后再properties/yaml配置文件中书写配置即可

## 依赖

- springboot的jdbc api
- mysql驱动

## properties文件

```properties
# jdbc基础配置
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.url=jdbc:mysql://localhost:3306/blog
spring.datasource.driver-class-name=com.mysql.jdbc.Driver
```



## pom文件

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter</artifactId>
    </dependency>
    <!--jdbc api-->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-jdbc</artifactId>
    </dependency>

    <!--mysql驱动-->
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>

    <!--引入了它 配置文件书写时会有提示-->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-configuration-processor</artifactId>
        <optional>true</optional>
    </dependency>

    <!--单元测试依赖-->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
        <exclusions>
            <exclusion>
                <groupId>org.junit.vintage</groupId>
                <artifactId>junit-vintage-engine</artifactId>
            </exclusion>
        </exclusions>
    </dependency>
</dependencies>

<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```



# 整合 Druid

## 概述

SpringBoot整合Druid，需要导入数据库的连接驱动包（如mysql的驱动），Druid包，然后书写对应的配置文件。注意：由于数据库连接相关的配置文件是在Druid中进行设置的，所以前缀名要一致。

## 依赖

- mysql驱动
- springboot jdbc api
- druid依赖
- springboot-web依赖，用来注册servlet，filter 启用druid的控制台
- log4j，我们使用这个日志框架进行记录

## pom文件

```xml
<dependencies>
    <!-- jdbc aip -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-jdbc</artifactId>
    </dependency>
    <!-- mysql驱动 -->
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>
    <!-- drudi依赖 -->
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid</artifactId>
        <version>1.1.8</version>
    </dependency>
    <!-- log4j druid用的日志框架 -->
    <dependency>
        <groupId>log4j</groupId>
        <artifactId>log4j</artifactId>
        <version>1.2.17</version>
    </dependency>
    <!--web依赖 用于启用druid的后台管理-->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <!-- 配置文件书写提示 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-configuration-processor</artifactId>
        <optional>true</optional>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
        <exclusions>
            <exclusion>
                <groupId>org.junit.vintage</groupId>
                <artifactId>junit-vintage-engine</artifactId>
            </exclusion>
        </exclusions>
    </dependency>
</dependencies>

<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

## properties文件

```properties
# jdbc基础配置
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.url=jdbc:mysql://localhost:3306/blog
spring.datasource.driver-class-name=com.mysql.jdbc.Driver
spring.datasource.type=com.alibaba.druid.pool.DruidDataSource
# druid详细配置
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
#配置监控统计拦截的filters，去掉后监控界面sql无法统计，'wall'用于防火墙
spring.datasource.filters=stat,wall,log4j
spring.datasource.maxPoolPreparedStatementPerConnectionSize=20
spring.datasource.useGlobalDataSourceStat=true
spring.datasource.connectionProperties=druid.stat.mergeSql=true;druid.stat.slowSqlMillis=500
```

## 代码

```java
// 表明这是一个JavaConfig配置类
@Configuration
public class DruidConfig {

    @Bean
    @ConfigurationProperties(prefix = "spring.datasource")
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

# 整合 MyBatis

## 概述

数据库采用MySQL，连接池采用Druid。整合方式有SQL用纯注解和SQL采用xml两个版本。

## 依赖

- mysql驱动
- springboot jdbc api
- druid 依赖
- springboot-web 依赖，用来注册 servlet，filter 启用druid的控制台
- log4j，我们使用这个日志框架进行记录
- mybatis 和 Spring 的整合包
- 其他的会自动帮我们导入的，不必担心

## pom文件

```xml
<dependencies>
    <!-- jdbc aip -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-jdbc</artifactId>
    </dependency>
    <!-- mysql驱动 -->
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>
    <!-- drudi依赖 -->
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid</artifactId>
        <version>1.1.8</version>
    </dependency>
    <!-- log4j druid用的日志框架 -->
    <dependency>
        <groupId>log4j</groupId>
        <artifactId>log4j</artifactId>
        <version>1.2.17</version>
    </dependency>
    <!--web依赖 用于启用druid的后台管理-->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <!--mybatis与spring的整合包-->
    <dependency>
        <groupId>org.mybatis.spring.boot</groupId>
        <artifactId>mybatis-spring-boot-starter</artifactId>
        <version>2.1.2</version>
	</dependency>
    <!-- 配置文件书写提示 -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-configuration-processor</artifactId>
        <optional>true</optional>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
        <exclusions>
            <exclusion>
                <groupId>org.junit.vintage</groupId>
                <artifactId>junit-vintage-engine</artifactId>
            </exclusion>
        </exclusions>
    </dependency>
</dependencies>

<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

## properties件

```properties
# 基础的jdbc配置
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.url=jdbc:mysql://localhost:3306/blog
spring.datasource.driver-class-name=com.mysql.jdbc.Driver
spring.datasource.type=com.alibaba.druid.pool.DruidDataSource
# 初始化druid配置
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
# 开启druid的监控
spring.datasource.filters=stat,wall,log4j
spring.datasource.maxPoolPreparedStatementPerConnectionSize=20
spring.datasource.useGlobalDataSourceStat=true
spring.datasource.connectionProperties=druid.stat.mergeSql=true;druid.stat.slowSqlMillis=500
# mybatis采用xml书写SQL的话需要这一行，指定xml文件的位置
mybatis.mapper_locations=classpath:mapper/*.xml
# 开启驼峰命名
mybatis.configuration.map-underscore-to-camel-case=true
# 配置包别名
mybatis.type-aliases-package=com.bbxx.boot02.pojo
```

## SQL纯注解

每个dao接口都加上注解 mapper 或 在启动引导类位置处用@MapperScan("扫描得包全名")

### SQL书写

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

### 开启驼峰命名

也可以在配置文件中开启

`mybatis.configuration.map-underscore-to-camel-case=true`

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

## SQL xml版本

xml 放在那里？

```java
// 接口
public interface ArticleMapper {
    public List<Article> queryAll();
}

// 启动类
@MapperScan("com.bbxx.boot02.mapper")
@SpringBootApplication
public class DemoApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoApplication.class, args);
    }
}

// controller层
@RestController
public class ArticleController {

    @Autowired
    private ArticleMapper mapper;

    @GetMapping("/select")
    public List<Article> selectAll() {
        return mapper.queryAll();
    }

}
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.bbxx.boot02.mapper.ArticleMapper">
    <select id="queryAll" resultType="Article">
        select * from article;
    </select>
</mapper>
```

# 整合 JPA

```xml
<!--jdk9需要导入这种依赖-->
<dependency>
    <groupId>javax.xml.bind</groupId>
    <artifactId>jaxb-api</artifactId>
    <version>2.3.0</version>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <scope>runtime</scope>
</dependency>
```

```properties
server.port=8080
# 设置当前web应用的名称
server.servlet.context-path=/community
# jdbc相关配置
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.datasource.url=jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC
spring.datasource.username=root
spring.datasource.password=root

# jpa信息配置
spring.jpa.database=MySQL
spring.jpa.show-sql=true
spring.jpa.generate-ddl=true
spring.jpa.hibernate.ddl-auto=update
spring.jpa.hibernate.naming_strategy=org.hibernate.cfg.ImprovedNamingStrategy
```

```java
package cn.baobaoxuxu.community.pojo;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Entity
public class Role {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    private String roleName;
    private String roleDesc;

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getRoleName() {
        return roleName;
    }

    public void setRoleName(String roleName) {
        this.roleName = roleName;
    }

    public String getRoleDesc() {
        return roleDesc;
    }

    public void setRoleDesc(String roleDesc) {
        this.roleDesc = roleDesc;
    }

    @Override
    public String toString() {
        return "Role{" +
                "id=" + id +
                ", roleName='" + roleName + '\'' +
                ", roleDesc='" + roleDesc + '\'' +
                '}';
    }
}
```

```java
package cn.baobaoxuxu.community.repository;

import cn.baobaoxuxu.community.pojo.Role;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface RoleRepository extends JpaRepository<Role, Long> {
    public List<Role> findAll();
}
```



# 整合MVC

## 概述

SpringBoot 有默认的配置，它替我们设置了有关 MVC 的一些默认配置。我们可以不使用这些默认配置，全面接管相关配置（全部由我们自行定义），也可以只修改必要的部分，其他的仍采用 Spring Boot 为我们提供的默认配置。一般是不采用全面接管。

## 依赖

- springboot-web模块

## pom文件

```xml
<dependencies>
   <!--web模块-->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <!-- thymeleaf springboot默认的模板引擎，顺带一起导入了。高版本boot，写这个即可，其他的不用写-->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-thymeleaf</artifactId>
    </dependency>

    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-configuration-processor</artifactId>
        <optional>true</optional>
    </dependency>
    <dependency>
        <groupId>junit</groupId>
        <artifactId>junit</artifactId>
        <version>4.13</version>
        <scope>test</scope>
    </dependency>

    <dependency>
        <groupId>org.mybatis.spring.boot</groupId>
        <artifactId>mybatis-spring-boot-starter</artifactId>
        <version>2.1.2</version>
    </dependency>

    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
        <exclusions>
            <exclusion>
                <groupId>org.junit.vintage</groupId>
                <artifactId>junit-vintage-engine</artifactId>
            </exclusion>
        </exclusions>
    </dependency>
</dependencies>

<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

## 配置代码

### 拦截器

```java
public class LoginInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        Object user = request.getSession().getAttribute("user");
        if(user == null){
            System.err.println("Sorry please login");
            return false;
        }else{
            return true;
        }
    }

    @Override
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {

    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) throws Exception {

    }
}
```

### 注册组件

```java
@Configuration
public class WebMvcConfig implements WebMvcConfigurer {

    /**
     * 重写方法式 的配置
     */
    @Override
    public void addViewControllers(ViewControllerRegistry registry) {
        // 浏览器发送 /demo1 请求到success
        registry.addViewController("/demo1").setViewName("/success");
        registry.addViewController("/demo2").setViewName("/success");
    }

    /**
     * 组件式 配置
     */
    @Bean
    public WebMvcConfigurer webMvcConfigurer() {
        WebMvcConfigurer web = new WebMvcConfigurer() {
            @Override
            public void addInterceptors(InterceptorRegistry registry) {
                registry.addInterceptor(new LoginInterceptor()).addPathPatterns("/**").
                        excludePathPatterns("/index.html", "/login.html","/index","/list");
            }
        };
        return web;
    }
}
```

## 测试代码

```java
// RestController 表明，返回值为json格式的数据！
@RestController
public class DemoController {
    static List<Person> list = new ArrayList<>(8);

    static {
        list.add(new Person("1", "ljw1", "0", "1997/11/11"));
        list.add(new Person("2", "ljw231", "0", "1997/11/11"));
        list.add(new Person("3", "ljw1231", "1", "1997/11/11"));
        list.add(new Person("4", "lj45w1", "0", "1997/11/11"));
        list.add(new Person("5", "lj566w1", "1", "1997/11/11"));
        list.add(new Person("6", "ljw671", "0", "1997/11/11"));
    }

    @GetMapping("/list")
    public List<Person> success(HashMap<String, Object> maps) {
        return list;
    }

}
```

# 整合 MVC&MyBatis

## 概述

SpringBoot的ssm整合配置

## 依赖

- mysql驱动
- springboot jdbc api
- druid依赖
- springboot-web依赖，用来注册servlet，filter 启用druid的控制台
- log4j，我们使用这个日志框架进行记录
- mybatis和spring的整合包
- 其他的会自动帮我们导入的，不必担心

## pom文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.3.0.RELEASE</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>
    <groupId>com.bbxx</groupId>
    <artifactId>boot02</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>demo</name>
    <description>Demo project for Spring Boot</description>

    <properties>
        <java.version>11</java.version>
    </properties>

    <dependencies>
        <!-- 引入jq依赖 -->
        <dependency>
            <groupId>org.webjars</groupId>
            <artifactId>jquery</artifactId>
            <version>3.5.1</version>
        </dependency>

        <!--jdbc aip-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-jdbc</artifactId>
            <optional>true</optional>
        </dependency>

        <!--mysql驱动-->
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>5.1.47</version>
        </dependency>

        <!--druid数据源-->
        <dependency>
            <groupId>com.alibaba</groupId>
            <artifactId>druid</artifactId>
            <version>1.1.22</version>
        </dependency>

        <!--log4j日志-->
        <dependency>
            <groupId>log4j</groupId>
            <artifactId>log4j</artifactId>
            <version>1.2.17</version>
        </dependency>

        <!--web模块-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- thymeleaf导入-->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-thymeleaf</artifactId>
        </dependency>

        <!-- 配置文件提示 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-configuration-processor</artifactId>
            <optional>true</optional>
        </dependency>
        
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13</version>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>org.mybatis.spring.boot</groupId>
            <artifactId>mybatis-spring-boot-starter</artifactId>
            <version>2.1.2</version>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
            <exclusions>
                <exclusion>
                    <groupId>org.junit.vintage</groupId>
                    <artifactId>junit-vintage-engine</artifactId>
                </exclusion>
            </exclusions>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

</project>
```

## properties文件

```properties
# 禁用缓存
spring.thymeleaf.cache=false
# 基础的jdbc配置
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.url=jdbc:mysql://localhost:3306/blog
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
# mybatis xml方式书写SQL
mybatis.mapper_locations=classpath:mapper/*.xml
mybatis.configuration.map-underscore-to-camel-case=true
mybatis.type-aliases-package=com.bbxx.boot02.pojo
```

## Config代码

### druid

```java
@Configuration
public class DruidConfig {

    @ConfigurationProperties(prefix = "spring.datasource")
    @Bean
    public DataSource getDatasource() {
        return new DruidDataSource();
    }

    @Bean
    public ServletRegistrationBean tatViewServlet() {   // 注册servlet管理druid
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
    public FilterRegistrationBean webStatFilter() { // 注册过滤器
        FilterRegistrationBean bean = new FilterRegistrationBean(new WebStatFilter());
        Map<String, String> init = new HashMap();
        // 不拦截这些资源
        init.put("exclusions", "*.js,*.css,/druid/*");
        bean.setInitParameters(init);
        bean.setUrlPatterns(Arrays.asList("/*"));
        return bean;
    }
}
```

### web

```java
// 过滤器
public class LoginInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) throws Exception {
        Object user = request.getSession().getAttribute("user");
        if(user == null){
            System.err.println("Sorry please login");
            return false;
        }else{
            return true;
        }
    }

    @Override
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {

    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) throws Exception {

    }
}

// 更改部分web组件
@Configuration
public class WebMvcConfig implements WebMvcConfigurer {

    //重写方法式的配置
    @Override
    public void addViewControllers(ViewControllerRegistry registry) {
        // 浏览器发送 /demo1 请求到success
        registry.addViewController("/demo1").setViewName("/success");
        registry.addViewController("/demo2").setViewName("/success");
    }
   
    // 组件方式配置
    @Bean
    public WebMvcConfigurer webMvcConfigurer() {
        WebMvcConfigurer web = new WebMvcConfigurer() {
            @Override
            public void addInterceptors(InterceptorRegistry registry) {
                registry.addInterceptor(new LoginInterceptor()).addPathPatterns("/**").
                        excludePathPatterns("/index.html", "/login.html","/index","/list");
            }
        };
        return web;
    }
}
```

## mybatis的sql文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.bbxx.boot02.mapper.ArticleMapper">

</mapper>
```

# 整合 Redis

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
```

```properties
# redis配置信息
spring.redis.host=localhost
spring.redis.port=6379
```

```java
package cn.baobaoxuxu.community;

import cn.baobaoxuxu.community.pojo.Role;
import cn.baobaoxuxu.community.repository.RoleRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.List;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = CommunityApplication.class)
public class RedisTest {

    @Autowired
    RoleRepository dao;

    @Autowired
    RedisTemplate<String, String> redisTemplate;

    @Test
    public void test1() throws JsonProcessingException {
        // 1. 从redis拿数据
        String s = redisTemplate.boundValueOps("user.findAll").get();
        if (s == null) {
            List<Role> all = dao.findAll();
            ObjectMapper objectMapper = new ObjectMapper();
            String s1 = objectMapper.writeValueAsString(all);
            redisTemplate.boundValueOps("user.findAll").set(s1);
            System.out.println("从数据库拿了 存到了redis");
        }else{
            System.out.println("直接從redis拿");
        }
        //2. 判断redis是否存在数据
    }
}
```

# 集成 JUnit

导入测试依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-test</artifactId>
</dependency>
```

别导错包哦！

```java
@RunWith(SpringRunner.class) // 固定写法
@SpringBootTest(classes = CommunityApplication.class) // 启动类的class字节码
public class MyBatisTest {
    @Autowired
    private RoleMapper mapper;

    @Test
    public void test() {
        List<Role> all = mapper.findAll();
        Assert.assertNotNull(all);
    }
}
```
