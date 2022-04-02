# SpringBoot

主要看小马哥的SpringBoot

雷丰阳的作为补充：(https://www.bilibili.com/video/BV19K4y1L7MT?spm_id_from=333.788.b_765f64657363.1)

## SpringBoot 2.0介绍

- 编程语言：Java 8+、Kotlin
- 底层框架：Spring Framwork 5.0.x
- 全新特性：Web Flux， 对Spring MVC的补充。
  - 函数编程：Java 8 Lambda
  - 响应式编程：Reactive Streams
  - 异步编程：Servlet 3.1 或 Asyc NIO

## 学习内容

- SpringBoot如何基于Spring Framework逐步走向自动装配。
- SpringApplication如何掌控Spring应用生命周期。
- SpringBoot外部化配置与Spring Environment抽象之间是生命关系。
- Spring Web MVC向Spring Reactive WebFlux过渡的真实价值和意义。

## 学习方式

- 场景分析，掌握技术选型
- 系统学习
- 重视规范，了解发展趋势
- 源码解读，理解设计思想
- 实战演练，巩固学习成果

## 学习收获

- Spring 全栈技术和实现原理
- Spring Boot核心技术
- 微服务基础设施开发与生产实施经验

## 难精

- 组件自动装配：模式注解、@Enable模块、条件装配、加载机制
- 外部化配置：Environment抽象、生命周期、破坏性变更
- 嵌入式容器：Servlet Web容器、Reactive Web容器
- Spring Boot Starter：依赖管理、装配条件、装配顺序
- Production-Ready：健康检查、数据指标、@Endpoint管控

## Spring Boot的特点

- 组件自动装配：规约大于配置，专注业务
- 外部化配置：一次构建、按需调配、到处运行
- 嵌入式容器：内置容器、无需部署、独立运行
- Spring Boot Starter：简化依赖，按需装配、自我包含
- Production-Ready：一站式运维、生态无缝整合

## Spring Boot与JavaEE规范

- Web：Servlet（JSR-315、JSR-340）
- SQL：JDBC（JSR-221）
- 数据校验：Bean Validation（JSR 303、JSR 349）
- 缓存：Java Caching API（JSR-107）
- WebSockets：Java API for WebSocket（JSR-356）
- Web Services：JAX-WS（JSR-224）
- Java管理：JMX（JSR 3）
- 消息：JMS（JSR-914）

## 主干内容

- 核心特性
- Web应用
- 数据相关
- 功能扩展
- 运维管理

## 快速入门案例

> 场景说明

- 定义用户模型，包括属性：用户ID和名称
- 客户端发送POST请求，创建用户（Web MVC）
- 客户端发送GET请求，获取所有用户（Web Flux）

----

NIO的reactor是同步非阻塞

Web Flux的reactor是异步非阻塞的一个实现

- Flux：0-n的对象
- Mono：0-1的对象

# 理解SpringBoot

SpringBoot应用可以是jar可以是war。jar和war是如何启动的？如何指定那个类为引导类。

## jar文件结构

- BOOT-INF/classes：目录存放应用编译后的 class 文件
- BOOT-INF/lib：存放应用依赖的 jar 包
- META-INF/：存放应用相关元信息，如 MANIFEST.MF 文件。
- org/：存放 SpringBoot 相关的 class 文件

java -jar 命令为何可以执行 FAT JAR文件？

==符合java的标准就可以用 java -jar 执行 jar 包，Java官方文档规定，java -jar 命令引导的具体启动类必须配置在 MANIFEST.MF 资源的 Main-Class 属性中。==

- MANIFEST.MF 文件内容

    ```bash
    Manifest-Version: 1.0
    Created-By: Maven Jar Plugin 3.2.0
    Build-Jdk-Spec: 11
    Implementation-Title: demo
    Implementation-Version: 0.0.1-SNAPSHOT
    Main-Class: org.springframework.boot.loader.JarLauncher
    Start-Class: com.example.demo.DemoApplication
    Spring-Boot-Version: 2.5.2
    Spring-Boot-Classes: BOOT-INF/classes/
    Spring-Boot-Lib: BOOT-INF/lib/
    Spring-Boot-Classpath-Index: BOOT-INF/classpath.idx
    Spring-Boot-Layers-Index: BOOT-INF/layers.idx
    ```

- **Main-Class: org.springframework.boot.loader.JarLauncher** 指定了这是jar运行
- **Main-Class: org.springframework.boot.loader.WarLauncher** 指定了这是war运行
- 这两个类是 jar / war 的启动器，都是 org.springframework.boot.loader 中的类。

# 核心特性

## Spring Boot三大特性

- 组件自动装配：Web MVC、Web Flux、JDBC等
- 嵌入式Web容器：Tomcat、Jetty以及Undertow
- 生产准备特性：指标、健康检查、外部化配置等

## 组件自动装配

- 激活：@EnableAutoConfiguration
- 配置：/META-INF/spring.factories（这个目录是相对于 Classpath 而言，META-INF 元信息；spring.factories 工厂模式，key-value 键值对的配置信息）
- 实现：XXXAutoConfiguration

### quick start

> 新建一个Spring Boot项目，包含Web模块。

```java
// SpringBoot简介中的Demo
@RestController
@EnableAutoConfiguration
public class SpringbootApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringbootApplication.class, args);
    }

    @GetMapping("/hello")
    public String get() {
        return "Hello World Spring Boot";
    }

}
```

---

也可以用以下注解

```java
@SpringBootApplication
@RestController
public class SpringbootApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringbootApplication.class, args);
    }

    @GetMapping("/hello")
    public String get() {
        return "Hello World Spring Boot";
    }

}
```

查看源码可知，`@SpringBootApplication`注解是一个组合注解，包含了`@EnableAutoConfiguration`注解~

### spring.factories

```shell
# ConfigData Location Resolvers
# key-value形式，可以把key看成类名或接口名，value看出他的实现类
org.springframework.boot.context.config.ConfigDataLocationResolver=\
org.springframework.boot.context.config.ConfigTreeConfigDataLocationResolver,\
org.springframework.boot.context.config.StandardConfigDataLocationResolver
```

---

让我们看看EnableAutoConfiguration的配置信息

```shell
# Auto Configure
org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
org.springframework.boot.autoconfigure.admin.SpringApplicationAdminJmxAutoConfiguration,\
org.springframework.boot.autoconfigure.aop.AopAutoConfiguration,\
org.springframework.boot.autoconfigure.amqp.RabbitAutoConfiguration,\
org.springframework.boot.autoconfigure.batch.BatchAutoConfiguration,\
org.springframework.boot.autoconfigure.cache.CacheAutoConfiguration,\
org.springframework.boot.autoconfigure.cassandra.CassandraAutoConfiguration,\
org.springframework.boot.autoconfigure.context.ConfigurationPropertiesAutoConfiguration,\
org.springframework.boot.autoconfigure.context.LifecycleAutoConfiguration,\
org.springframework.boot.autoconfigure.context.MessageSourceAutoConfiguration,\
org.springframework.boot.autoconfigure.context.PropertyPlaceholderAutoConfiguration,\
org.springframework.boot.autoconfigure.couchbase.CouchbaseAutoConfiguration,\
org.springframework.boot.autoconfigure.dao.PersistenceExceptionTranslationAutoConfiguration,\
org.springframework.boot.autoconfigure.data.cassandra.CassandraDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.cassandra.CassandraReactiveDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.cassandra.CassandraReactiveRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.cassandra.CassandraRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.couchbase.CouchbaseDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.couchbase.CouchbaseReactiveDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.couchbase.CouchbaseReactiveRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.couchbase.CouchbaseRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.elasticsearch.ElasticsearchDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.elasticsearch.ElasticsearchRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.elasticsearch.ReactiveElasticsearchRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.elasticsearch.ReactiveElasticsearchRestClientAutoConfiguration,\
org.springframework.boot.autoconfigure.data.jdbc.JdbcRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.jpa.JpaRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.ldap.LdapRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.mongo.MongoDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.mongo.MongoReactiveDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.mongo.MongoReactiveRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.mongo.MongoRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.neo4j.Neo4jDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.neo4j.Neo4jReactiveDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.neo4j.Neo4jReactiveRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.neo4j.Neo4jRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.solr.SolrRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.r2dbc.R2dbcDataAutoConfiguration,\
org.springframework.boot.autoconfigure.data.r2dbc.R2dbcRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.redis.RedisAutoConfiguration,\
org.springframework.boot.autoconfigure.data.redis.RedisReactiveAutoConfiguration,\
org.springframework.boot.autoconfigure.data.redis.RedisRepositoriesAutoConfiguration,\
org.springframework.boot.autoconfigure.data.rest.RepositoryRestMvcAutoConfiguration,\
org.springframework.boot.autoconfigure.data.web.SpringDataWebAutoConfiguration,\
org.springframework.boot.autoconfigure.elasticsearch.ElasticsearchRestClientAutoConfiguration,\
org.springframework.boot.autoconfigure.flyway.FlywayAutoConfiguration,\
org.springframework.boot.autoconfigure.freemarker.FreeMarkerAutoConfiguration,\
org.springframework.boot.autoconfigure.groovy.template.GroovyTemplateAutoConfiguration,\
org.springframework.boot.autoconfigure.gson.GsonAutoConfiguration,\
org.springframework.boot.autoconfigure.h2.H2ConsoleAutoConfiguration,\
org.springframework.boot.autoconfigure.hateoas.HypermediaAutoConfiguration,\
org.springframework.boot.autoconfigure.hazelcast.HazelcastAutoConfiguration,\
org.springframework.boot.autoconfigure.hazelcast.HazelcastJpaDependencyAutoConfiguration,\
org.springframework.boot.autoconfigure.http.HttpMessageConvertersAutoConfiguration,\
org.springframework.boot.autoconfigure.http.codec.CodecsAutoConfiguration,\
org.springframework.boot.autoconfigure.influx.InfluxDbAutoConfiguration,\
org.springframework.boot.autoconfigure.info.ProjectInfoAutoConfiguration,\
org.springframework.boot.autoconfigure.integration.IntegrationAutoConfiguration,\
org.springframework.boot.autoconfigure.jackson.JacksonAutoConfiguration,\
org.springframework.boot.autoconfigure.jdbc.DataSourceAutoConfiguration,\
org.springframework.boot.autoconfigure.jdbc.JdbcTemplateAutoConfiguration,\
org.springframework.boot.autoconfigure.jdbc.JndiDataSourceAutoConfiguration,\
org.springframework.boot.autoconfigure.jdbc.XADataSourceAutoConfiguration,\
org.springframework.boot.autoconfigure.jdbc.DataSourceTransactionManagerAutoConfiguration,\
org.springframework.boot.autoconfigure.jms.JmsAutoConfiguration,\
org.springframework.boot.autoconfigure.jmx.JmxAutoConfiguration,\
org.springframework.boot.autoconfigure.jms.JndiConnectionFactoryAutoConfiguration,\
org.springframework.boot.autoconfigure.jms.activemq.ActiveMQAutoConfiguration,\
org.springframework.boot.autoconfigure.jms.artemis.ArtemisAutoConfiguration,\
org.springframework.boot.autoconfigure.jersey.JerseyAutoConfiguration,\
org.springframework.boot.autoconfigure.jooq.JooqAutoConfiguration,\
org.springframework.boot.autoconfigure.jsonb.JsonbAutoConfiguration,\
org.springframework.boot.autoconfigure.kafka.KafkaAutoConfiguration,\
org.springframework.boot.autoconfigure.availability.ApplicationAvailabilityAutoConfiguration,\
org.springframework.boot.autoconfigure.ldap.embedded.EmbeddedLdapAutoConfiguration,\
org.springframework.boot.autoconfigure.ldap.LdapAutoConfiguration,\
org.springframework.boot.autoconfigure.liquibase.LiquibaseAutoConfiguration,\
org.springframework.boot.autoconfigure.mail.MailSenderAutoConfiguration,\
org.springframework.boot.autoconfigure.mail.MailSenderValidatorAutoConfiguration,\
org.springframework.boot.autoconfigure.mongo.embedded.EmbeddedMongoAutoConfiguration,\
org.springframework.boot.autoconfigure.mongo.MongoAutoConfiguration,\
org.springframework.boot.autoconfigure.mongo.MongoReactiveAutoConfiguration,\
org.springframework.boot.autoconfigure.mustache.MustacheAutoConfiguration,\
org.springframework.boot.autoconfigure.neo4j.Neo4jAutoConfiguration,\
org.springframework.boot.autoconfigure.orm.jpa.HibernateJpaAutoConfiguration,\
org.springframework.boot.autoconfigure.quartz.QuartzAutoConfiguration,\
org.springframework.boot.autoconfigure.r2dbc.R2dbcAutoConfiguration,\
org.springframework.boot.autoconfigure.r2dbc.R2dbcTransactionManagerAutoConfiguration,\
org.springframework.boot.autoconfigure.rsocket.RSocketMessagingAutoConfiguration,\
org.springframework.boot.autoconfigure.rsocket.RSocketRequesterAutoConfiguration,\
org.springframework.boot.autoconfigure.rsocket.RSocketServerAutoConfiguration,\
org.springframework.boot.autoconfigure.rsocket.RSocketStrategiesAutoConfiguration,\
org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration,\
org.springframework.boot.autoconfigure.security.servlet.UserDetailsServiceAutoConfiguration,\
org.springframework.boot.autoconfigure.security.servlet.SecurityFilterAutoConfiguration,\
org.springframework.boot.autoconfigure.security.reactive.ReactiveSecurityAutoConfiguration,\
org.springframework.boot.autoconfigure.security.reactive.ReactiveUserDetailsServiceAutoConfiguration,\
org.springframework.boot.autoconfigure.security.rsocket.RSocketSecurityAutoConfiguration,\
org.springframework.boot.autoconfigure.security.saml2.Saml2RelyingPartyAutoConfiguration,\
org.springframework.boot.autoconfigure.sendgrid.SendGridAutoConfiguration,\
org.springframework.boot.autoconfigure.session.SessionAutoConfiguration,\
org.springframework.boot.autoconfigure.security.oauth2.client.servlet.OAuth2ClientAutoConfiguration,\
org.springframework.boot.autoconfigure.security.oauth2.client.reactive.ReactiveOAuth2ClientAutoConfiguration,\
org.springframework.boot.autoconfigure.security.oauth2.resource.servlet.OAuth2ResourceServerAutoConfiguration,\
org.springframework.boot.autoconfigure.security.oauth2.resource.reactive.ReactiveOAuth2ResourceServerAutoConfiguration,\
org.springframework.boot.autoconfigure.solr.SolrAutoConfiguration,\
org.springframework.boot.autoconfigure.task.TaskExecutionAutoConfiguration,\
org.springframework.boot.autoconfigure.task.TaskSchedulingAutoConfiguration,\
org.springframework.boot.autoconfigure.thymeleaf.ThymeleafAutoConfiguration,\
org.springframework.boot.autoconfigure.transaction.TransactionAutoConfiguration,\
org.springframework.boot.autoconfigure.transaction.jta.JtaAutoConfiguration,\
org.springframework.boot.autoconfigure.validation.ValidationAutoConfiguration,\
org.springframework.boot.autoconfigure.web.client.RestTemplateAutoConfiguration,\
org.springframework.boot.autoconfigure.web.embedded.EmbeddedWebServerFactoryCustomizerAutoConfiguration,\
org.springframework.boot.autoconfigure.web.reactive.HttpHandlerAutoConfiguration,\
org.springframework.boot.autoconfigure.web.reactive.ReactiveWebServerFactoryAutoConfiguration,\
org.springframework.boot.autoconfigure.web.reactive.WebFluxAutoConfiguration,\
org.springframework.boot.autoconfigure.web.reactive.error.ErrorWebFluxAutoConfiguration,\
org.springframework.boot.autoconfigure.web.reactive.function.client.ClientHttpConnectorAutoConfiguration,\
org.springframework.boot.autoconfigure.web.reactive.function.client.WebClientAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.DispatcherServletAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.ServletWebServerFactoryAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.error.ErrorMvcAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.HttpEncodingAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.MultipartAutoConfiguration,\
org.springframework.boot.autoconfigure.web.servlet.WebMvcAutoConfiguration,\
org.springframework.boot.autoconfigure.websocket.reactive.WebSocketReactiveAutoConfiguration,\
org.springframework.boot.autoconfigure.websocket.servlet.WebSocketServletAutoConfiguration,\
org.springframework.boot.autoconfigure.websocket.servlet.WebSocketMessagingAutoConfiguration,\
org.springframework.boot.autoconfigure.webservices.WebServicesAutoConfiguration,\
org.springframework.boot.autoconfigure.webservices.client.WebServiceTemplateAutoConfiguration
```

## 嵌入web容器

- Web Servlet：Tomcat、Jetty和Undertow
- Web Reactive：Netty Web Server

## 生产准备特性

- 指标：/actuator/metrics
- 健康检查：/actuator/health
- 外部化配置：/actuator/configprops 修改应用行为

# Web应用

## 传统Servlet应用概述

- Servlet组件：Servlet、Filter、Listener（传统的三大组件）
- Servlet注册：Servlet注解、Spring Bean、RegistrationBean（如何把Servlet注册进来）
  - 我们允许把Servlet注册成一个Spring Bean加载运行。
  - RegistrationBean
- 异步非阻塞：异步Servlet、非阻塞Servlet

`依赖`

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
```

## Servlet组件

- Servlet

  - 实现

    ```java
    @WebServlet(urlPatterns = {"/my/servlet"})
    public class MyServlet extends HttpServlet {
        @Override
        protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
            resp.getWriter().write("Hello");
        }
    }
    ```

  - URL映射

    ```java
    @WebServlet(urlPatterns = {"/my/servlet"})
    ```

  - 注册

    ```java
    @SpringBootApplication
    @ServletComponentScan(basePackages = {"com.example.demo.web.servlet"})
    public class SpringbootApplication {
    
        public static void main(String[] args) {
            SpringApplication.run(SpringbootApplication.class, args);
        }
    
    }
    ```

- Filter

- Listener

## Servlet注册

### Servlet注释

> Servlet注解方式配置

- @ServleComponentScan+
  - @WebServlet
  - @WebFilter
  - @WebListener

### Spring Bean

> Spring Bean方式配置

- @Bean+
  - Servlet
  - Filter
  - Listener

### RegistrationBean

>RegistrationBean方式配置

- ServletRegistrationBean
- FilterRegistrationBean
- ServletListenerRegistrationBean

## 异步非阻塞

### 异步Servlet

- javax.servlet.ServletRequest#startAsync()
- javax.servlet.AsyncContext

### 非阻塞Servlet

- javax.servlet.ServletInputStream#setReadListener
  - javax.servlet.ReadListener
- javax.servlet.ServletOutputStream#setWriteListener
  - javax.servlet.WriteListener

> 异步调用代码示例

```java
//  asyncSupported = true 设置为支持异步。默认是false！
@WebServlet(urlPatterns = {"/my/async"}, asyncSupported = true)
public class AsyncServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) {
        resp.setContentType("text/html");
        AsyncContext asyncContext = req.startAsync();
        asyncContext.start(() -> {
            try {
                PrintWriter writer = resp.getWriter();
                for (int i = 0; i < 100; i++) {
                    TimeUnit.SECONDS.sleep(1);
                    writer.write("Hello I am asyncContext");
                    writer.flush();
                }
                // 执行完毕后在告知 异步调用完成奥~
                asyncContext.complete();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }
}
```

## Spring Web MVC应用

- Web MVC视图：模板引擎、内容协商、异常处理
- Web MVC REST：资源服务、资源跨域、服务发现
- Web MVC核心：核心架构、处理流程、核心组件

### Web MVC视图

- ViewResolver

- View

> ViewResolver

```java
public interface ViewResolver {

	/**
	 * Resolve the given view by name.
	 * <p>Note: To allow for ViewResolver chaining, a ViewResolver should
	 * return {@code null} if a view with the given name is not defined in it.
	 * However, this is not required: Some ViewResolvers will always attempt
	 * to build View objects with the given name, unable to return {@code null}
	 * (rather throwing an exception when View creation failed).
	 * @param viewName name of the view to resolve
	 * @param locale the Locale in which to resolve the view.
	 * ViewResolvers that support internationalization should respect this.
	 * @return the View object, or {@code null} if not found
	 * (optional, to allow for ViewResolver chaining)
	 * @throws Exception if the view cannot be resolved
	 * (typically in case of problems creating an actual View object)
	 */
	@Nullable
	View resolveViewName(String viewName, Locale locale) throws Exception;

}
```



#### 模板引擎

- Thymeleaf
- Freemarker
- JSP

每种模板引擎对应不同的Resolver实现。如果有多种模板引擎怎么办？使用内容协商。

#### 内容协商

- ContentNegotiationConfigurer
- ContentNegotiationStrategy
- ContentNegotiationViewResolver

内容协商帮助你选择最合适的进行匹配。

#### 异常处理

- @ExceptionHandler
- HandlerExceptionResolver
  - ExceptionHandlerExceptionResolver
- BasicErrorController（SpringBoot）

### Web MVC REST

#### 资源服务

- @RequestMapping
  - @GetMapping
- @ResponseBody
- @RequestBody

#### 资源跨域

- CrossOrigin：[这个是注解驱动方式]
- WebMvcConfigurer#addCrosMappings [这个是接口编程方式]
- 传统解决方案
  - IFrame
  - JSONP

####  服务发现

- HATEOS

### Web MVC核心

#### 核心架构

#### 处理流程

#### 核心组件

- DispatcherServlet [前端控制器，本质也是一个Servlet] -- 把请求转发到不同的Controller中去
- HandlerMapping [处理器映射器]
- HandlerAdapter [把方法转化为内部的实现]
- ViewResolver []
- ...

## Spring Web Flux应用

Spring 5 开始支持的新特性。对Servlet的补充。

- Reactor基础：Java Lambda（实现的）、Mono（核心接口）、Flux（核心接口）
- Web Flux核心：Web MVC注解[兼容]、函数式声明、异步非阻塞
- 使用场景：Web Flux优势和限制

提升系统吞吐量，吞吐量≠快

### Reactor基础

#### Java Lambda

#### Mono

#### Flux

### Web Flux核心

#### Web MVC核心

- `@Controller`
- `@RequestMapping`
- `@ResponseBody`
- `@RequestBody`
- ...

#### 函数式声明

- `RouterFunction` [通过路由的方式表达函数~]

#### 异步非阻塞

- Servlet 3.1+
- Netty Reactor

### 使用场景

#### 页面渲染

#### REST应用

#### 性能测试

[Results from Spring 5 Webflux Performance Tests - Ippon Technologies](https://blog.ippon.tech/spring-5-webflux-performance-tests/)

## Web Server应用

不喜欢用tomcat，或不得不用Jetty，这时候需要做一些切换。

- 切换 Web Server
- 自定义 Servlet Web Server
- 自定义 Reactive Web Server

### 切换Web Server

#### 切换其他Servlet容器

- tomcat->jetty

  ```xml
  <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
      <exclusions>
          <exclusion>
              <groupId>org.springframework.boot</groupId>
              <artifactId>spring-boot-starter-tomcat</artifactId>
          </exclusion>
      </exclusions>
  </dependency>
  <!-- 切换为jetty -->
  <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-jetty</artifactId>
  </dependency>
  ```

#### 替换Servlet容器

- WebFlux

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-webflux</artifactId>
</dependency>
```

WebFlux的优先级是低于传统Servlet容器的。所以我们需要注释掉传统的web容器！

```xml
<!-- 把这个注释掉 -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-web</artifactId>
</dependency>
```

这样我们写的Servlet的内容就失效了，需要注释掉。不然项目无法正常启动。

项目启动后可以看到

```shell
2021-05-02 18:19:29.823  INFO 13668 --- [  restartedMain] com.example.demo.SpringbootApplication   : No active profile set, falling back to default profiles: default
2021-05-02 18:19:29.870  INFO 13668 --- [  restartedMain] .e.DevToolsPropertyDefaultsPostProcessor : Devtools property defaults active! Set 'spring.devtools.add-properties' to 'false' to disable
2021-05-02 18:19:29.871  INFO 13668 --- [  restartedMain] .e.DevToolsPropertyDefaultsPostProcessor : For additional web related logging consider setting the 'logging.level.web' property to 'DEBUG'
2021-05-02 18:19:30.767  INFO 13668 --- [  restartedMain] o.s.b.d.a.OptionalLiveReloadServer       : LiveReload server is running on port 35729
2021-05-02 18:19:31.860  INFO 13668 --- [  restartedMain] o.s.b.web.embedded.netty.NettyWebServer  : Netty started on port 8080
2021-05-02 18:19:31.873  INFO 13668 --- [  restartedMain] com.example.demo.SpringbootApplication   : Started SpringbootApplication in 2.325 seconds (JVM running for 3.442)
```

### 自定义Servlet Web Server

- WebServerFactoryCustomizer [SpringBoot 2.0新增]

  ```java
  
  /**
   * Strategy interface for customizing {@link WebServerFactory web server factories}. Any
   * beans of this type will get a callback with the server factory before the server itself
   * is started, so you can set the port, address, error pages etc.
   * <p>
   * Beware: calls to this interface are usually made from a
   * {@link WebServerFactoryCustomizerBeanPostProcessor} which is a
   * {@link BeanPostProcessor} (so called very early in the ApplicationContext lifecycle).
   * It might be safer to lookup dependencies lazily in the enclosing BeanFactory rather
   * than injecting them with {@code @Autowired}.
   *
   * @param <T> the configurable web server factory
   * @author Phillip Webb
   * @author Dave Syer
   * @author Brian Clozel
   * @since 2.0.0
   * @see WebServerFactoryCustomizerBeanPostProcessor
   */
  @FunctionalInterface
  public interface WebServerFactoryCustomizer<T extends WebServerFactory> {
  
  	/**
  	 * Customize the specified {@link WebServerFactory}.
  	 * @param factory the web server factory to customize
  	 */
  	void customize(T factory);
  
  }
  ```

他有很多实现类。包括Reactive的、Tomcat的、Netty的

### 自定义 Reactive Web Server

- ReactiveWebServerFactoryCustomizer

  ```java
  public class ReactiveWebServerFactoryCustomizer
  		implements WebServerFactoryCustomizer<ConfigurableReactiveWebServerFactory>, Ordered {
  
  	private final ServerProperties serverProperties;
  
  	public ReactiveWebServerFactoryCustomizer(ServerProperties serverProperties) {
  		this.serverProperties = serverProperties;
  	}
  
  	@Override
  	public int getOrder() {
  		return 0;
  	}
  
  	@Override
  	public void customize(ConfigurableReactiveWebServerFactory factory) {
  		PropertyMapper map = PropertyMapper.get().alwaysApplyingWhenNonNull();
  		map.from(this.serverProperties::getPort).to(factory::setPort);
  		map.from(this.serverProperties::getAddress).to(factory::setAddress);
  		map.from(this.serverProperties::getSsl).to(factory::setSsl);
  		map.from(this.serverProperties::getCompression).to(factory::setCompression);
  		map.from(this.serverProperties::getHttp2).to(factory::setHttp2);
  		map.from(this.serverProperties.getShutdown()).to(factory::setShutdown);
  	}
  
  }
  ```

# 数据相关

## 关系型数据库

- JDBC：数据源，JdbcTemplate、自动装配 [关注这三个方面]
- JPA：实体映射关系、实体操作、自动装配
- 事务：Spring事务抽象、JDBC事务处理、自动装配

### JDBC

> 依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-jdbc</artifactId>
    <version>2.4.2</version>
</dependency>
```

> 数据源

- javax.sql.DataSource

JdbcTemplate

> 自动装配

- DataSourceAutoConfiguration

### JPA

> 依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
    <version>2.4.2</version>
</dependency>
```

> 实体映射关系

- @javax.persistence.OneToOne
- @javax.persistence.OneToMany

- @javax.persistence.ManyToOne
- @javax.persistence.ManyToMany

> 实体操作

- javax.persistence.EntityManager

> 自动装配

- HibernateJpaAutoConfiguration

  ```java
  @Configuration(proxyBeanMethods = false)
  @ConditionalOnClass({ LocalContainerEntityManagerFactoryBean.class, EntityManager.class, SessionImplementor.class })
  @EnableConfigurationProperties(JpaProperties.class)
  @AutoConfigureAfter({ DataSourceAutoConfiguration.class }) // 数据源配置完成后再执行
  @Import(HibernateJpaConfiguration.class)
  public class HibernateJpaAutoConfiguration {
  
  }
  ```

### 事务（Transaction）

> 依赖

```xml
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-tx</artifactId>
</dependency>
```

>Spring事务抽象

- PlatformTransactionManager

  ```java
  public interface PlatformTransactionManager extends TransactionManager {
      TransactionStatus getTransaction(@Nullable TransactionDefinition var1) throws TransactionException;
  
      void commit(TransactionStatus var1) throws TransactionException;
  
      void rollback(TransactionStatus var1) throws TransactionException;
  }
  ```

  

>JDBC事务处理

- DataSourceTransactionManager

>自动装配

- TransactionAutoConfiguration

# 功能扩展

## Spring Boot应用

- SpringApplication：失败分析、应用特性、事件监听等。
- Spring Boot配置：外部化配置、Profile、配置属性
- Spring Boot Starter：Starter开发、最佳实践

### SpringApplication

```java
@SpringBootApplication
public class SpringbootApplication {

    public static void main(String[] args) {
        // 为什么要传两个参数，不传会出现什么问题。
        SpringApplication.run(SpringbootApplication.class, args);
    }

}
```



#### 失败分析

- FailureAnalysisReporter

  ```java
  @FunctionalInterface
  public interface FailureAnalysisReporter {
  
  	/**
  	 * Reports the given {@code failureAnalysis} to the user.
  	 * @param analysis the analysis
  	 */
  	void report(FailureAnalysis analysis);
  
  }
  ```

#### 应用特性

- `SpringApplication` Fluent API

  ```java
  @SpringBootApplication
  public class SpringbootApplication {
  
      public static void main(String[] args) {
  //        SpringApplication.run(SpringbootApplication.class, args);
          new SpringApplicationBuilder(SpringbootApplication.class)
  //                .web(WebApplicationType.NONE) 设置了这个 就不会以web应用启动了
                  .properties("a=b")
                  .run(args);
      }
  
  }
  ```

### Spring Boot配置

- 外部化配置

  - `ConfigurationProperty` [Since:2.0.0]

    ```java
    public final class ConfigurationProperty implements OriginProvider, Comparable<ConfigurationProperty> {
    
    	private final ConfigurationPropertyName name;
    
    	private final Object value;
    
        // 跟踪配置从哪里来
    	private final Origin origin;
    	// ...
    }
    ```

    

- @Profile [能力很弱，后续会调整成Conditional]

- 配置属性

  - PropertySource [Spring的]

## Spring Boot Starter

# 运维管理

Spring Boot Actuator

- 通过端点管理各类Web和JMX Endpoints
- 健康检查：Health、HealthIndicator
- 指标：内建Metrics、自定义Metrics

## Spring Boot Actuator

> 依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-actuator</artifactId>
</dependency>
```

> 端点

- Web Endpoint
- JMX Endpoint

> 健康检查

- Health
- HealthIndicator

> 指标

---

# 走向自动装配

## Spring Framework手动配置

- 定义：一种用于声明在`应用`中扮演“组件”角色的注解
- 举例：@Component、@Service、@Configuration[标注这是一个配置]
- 装配：`<context:component=scan>` 或 @ComponentScan

### Spring @Enable模块装配

Spring Framework 3.1开始支持“@Enable模块驱动”。所谓“模块”是具备相同领域的功能组件集合。组合所形成一个独立的单元。比如Web MVC模块、AspectJ模块、Caching（缓存）模块、JMX（Java管理扩展）模块、Async（异步处理）模块等。

- 定义：具备相同领域的功能组件集合，组合所形成一个独立的单元
- 举例：@EnableWevMvc、@EnableAutoConfiguration等
- 实现：注解方式、编程方式

> @Enable注解模块举例

| 框架实现         | @Enable注解模块                | 激活模块           |
| ---------------- | ------------------------------ | ------------------ |
| Spring Framework | @EnableMvc                     | Web MVC模块        |
|                  | @EnableTransactionManagement   | 事务管理模块       |
|                  | @EnableCaching                 | Caching模块        |
|                  | @EnableMBeanExport             | JMX模块            |
|                  | @EnableAsnyc                   | 异步处理模块       |
|                  | EnableWebFlux                  | Web Flux模块       |
|                  | @EnableAspectJAutoProxy        | AspectJ 代理模块   |
|                  |                                |                    |
| Spring Boot      | @EnableAutoConfiguration       | 自动装配模块       |
|                  | @EnableManagementContext       | Actuator管理模块   |
|                  | @EnableConfigurationProperties | 配置属性绑定模块   |
|                  | @EnableOAuth2Sso               | OAuth2单点登录模块 |
|                  |                                |                    |
| Spring Cloud     | @EnableEurekaServer            | Eureka服务模块     |
|                  | @EnableConfigServer            | 配置服务模块       |
|                  | @EnableFeignClients            | Feign客户端模块    |
|                  | @EnableZuulProxy               | 服务网关Zuul模块   |
|                  | @EnableCircuitBreaker          | 服务熔断模块       |

### Spring 条件装配

从Spring Framework 3.1开始，允许在Bean装配时增加前置条件判断。

- 定义：Bean装配的前置判断
- 举例：@Profile、@Conditional
- 实现：注解方式、编程方式

> 条件注解举例

| Spring注解   | 场景说明       | 起始版本 |
| ------------ | -------------- | -------- |
| @Profile     | 配置化条件装配 | 3.1      |
| @Conditional | 编程条件装配   | 4.0      |

## Spring 模式注解装配

### 模式注解

模式注解是一种用于声明在应用中扮演“组件”角色的注解。如Spring Framework中的`@Repository`标注在任何类上，用于扮演仓储角色的模式注解。

`@Component`作为一种由Spring容器托管的通用模式组件，任何被`@Component`标注的组件均为组件扫描的候选对象。类似地，凡是被`@Component`元标注（meta-annotated）的注解，如`@Service`，当任何组件标注它时，也被视作组件扫描的候选对象。

### 模式注解举例

| Spring Framework注解 | 场景说明          | 起始版本 |
| -------------------- | ----------------- | -------- |
| @Repository          | 数据仓储模式注解  | 2.0      |
| @Component           | 通用组件模式注解  | 2.5      |
| @Service             | 服务模式注解      | 2.5      |
| @Controller          | Web控制器模式注解 | 2.5      |
| @Configuration       | 配置类模式注解    | 3.0      |

### 装配方式

> `<context:component-scan>`方式

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd http://www.springframework.org/schema/context https://www.springframework.org/schema/context/spring-context.xsd">

    <!-- 激活注解驱动特性 -->
    <context:annotation-config/>
    <!-- 找寻被 @Component或者其他派生 Annotation标记的类，将它们注册为 Spring Bean -->
    <context:component-scan base-package="com.example.demo"/>
    
</beans>
```

> @ComponentScan方式

```java
@ComponentScan(basePackages = "com.example.demo")
public class SpringConfig {
}

```

### 自定义模式注解

@Component“派生性”

```java
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Repository(value = "firstLevelRepository")
public @interface FirstLevelRepository {
    String value() default "";
}

```

- @Component
  - @Repository
    - FirstLevelRepository

@Component“层次性”

```java
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@FirstLevelRepository(value = "secondLevelRepository")
public @interface SecondLevelRepository {
}
```

- @Component
  - @Repository
    - FirstLevelRepository
      - SecondLevelRepository

## Spring@Enable模块装配

### 模块举例@Enable注解

### 实现方式

#### 注解驱动方式

```java
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
@Documented
@Import(DelegatingWebMvcConfiguration.class)
public @interface EnableWebMvc {
}
```

----

```java
public class WebMvcConfigurationSupport implements ApplicationContextAware, ServletContextAware {
	//...
}
```

#### 接口编程方式

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
// 注意@Import注解，及CachingConfigurationSelector.class
@Import(CachingConfigurationSelector.class)
public @interface EnableCaching {

	/**
	 * Indicate whether subclass-based (CGLIB) proxies are to be created as opposed
	 * to standard Java interface-based proxies. The default is {@code false}. <strong>
	 * Applicable only if {@link #mode()} is set to {@link AdviceMode#PROXY}</strong>.
	 * <p>Note that setting this attribute to {@code true} will affect <em>all</em>
	 * Spring-managed beans requiring proxying, not just those marked with {@code @Cacheable}.
	 * For example, other beans marked with Spring's {@code @Transactional} annotation will
	 * be upgraded to subclass proxying at the same time. This approach has no negative
	 * impact in practice unless one is explicitly expecting one type of proxy vs another,
	 * e.g. in tests.
	 */
	boolean proxyTargetClass() default false;

	/**
	 * Indicate how caching advice should be applied.
	 * <p><b>The default is {@link AdviceMode#PROXY}.</b>
	 * Please note that proxy mode allows for interception of calls through the proxy
	 * only. Local calls within the same class cannot get intercepted that way;
	 * a caching annotation on such a method within a local call will be ignored
	 * since Spring's interceptor does not even kick in for such a runtime scenario.
	 * For a more advanced mode of interception, consider switching this to
	 * {@link AdviceMode#ASPECTJ}.
	 */
	AdviceMode mode() default AdviceMode.PROXY;

	/**
	 * Indicate the ordering of the execution of the caching advisor
	 * when multiple advices are applied at a specific joinpoint.
	 * <p>The default is {@link Ordered#LOWEST_PRECEDENCE}.
	 */
	int order() default Ordered.LOWEST_PRECEDENCE;

}
```

----

```java
public class CachingConfigurationSelector extends AdviceModeImportSelector<EnableCaching> {

	private static final String PROXY_JCACHE_CONFIGURATION_CLASS =
			"org.springframework.cache.jcache.config.ProxyJCacheConfiguration";

	private static final String CACHE_ASPECT_CONFIGURATION_CLASS_NAME =
			"org.springframework.cache.aspectj.AspectJCachingConfiguration";

	private static final String JCACHE_ASPECT_CONFIGURATION_CLASS_NAME =
			"org.springframework.cache.aspectj.AspectJJCacheConfiguration";


	private static final boolean jsr107Present;

	private static final boolean jcacheImplPresent;

	static {
		ClassLoader classLoader = CachingConfigurationSelector.class.getClassLoader();
		jsr107Present = ClassUtils.isPresent("javax.cache.Cache", classLoader);
		jcacheImplPresent = ClassUtils.isPresent(PROXY_JCACHE_CONFIGURATION_CLASS, classLoader);
	}


	/**
	 * Returns {@link ProxyCachingConfiguration} or {@code AspectJCachingConfiguration}
	 * for {@code PROXY} and {@code ASPECTJ} values of {@link EnableCaching#mode()},
	 * respectively. Potentially includes corresponding JCache configuration as well.
	 */
	@Override
	public String[] selectImports(AdviceMode adviceMode) {
		switch (adviceMode) {
			case PROXY:
				return getProxyImports();
			case ASPECTJ:
				return getAspectJImports();
			default:
				return null;
		}
	}

	/**
	 * Return the imports to use if the {@link AdviceMode} is set to {@link AdviceMode#PROXY}.
	 * <p>Take care of adding the necessary JSR-107 import if it is available.
	 */
	private String[] getProxyImports() {
		List<String> result = new ArrayList<>(3);
		result.add(AutoProxyRegistrar.class.getName());
		result.add(ProxyCachingConfiguration.class.getName());
		if (jsr107Present && jcacheImplPresent) {
			result.add(PROXY_JCACHE_CONFIGURATION_CLASS);
		}
		return StringUtils.toStringArray(result);
	}

	/**
	 * Return the imports to use if the {@link AdviceMode} is set to {@link AdviceMode#ASPECTJ}.
	 * <p>Take care of adding the necessary JSR-107 import if it is available.
	 */
	private String[] getAspectJImports() {
		List<String> result = new ArrayList<>(2);
		result.add(CACHE_ASPECT_CONFIGURATION_CLASS_NAME);
		if (jsr107Present && jcacheImplPresent) {
			result.add(JCACHE_ASPECT_CONFIGURATION_CLASS_NAME);
		}
		return StringUtils.toStringArray(result);
	}

}
```

---

```java
public interface ImportSelector {

	/**
	 * Select and return the names of which class(es) should be imported based on
	 * the {@link AnnotationMetadata} of the importing @{@link Configuration} class.
	 * @return the class names, or an empty array if none
	 */
	String[] selectImports(AnnotationMetadata importingClassMetadata);

	/**
	 * Return a predicate for excluding classes from the import candidates, to be
	 * transitively applied to all classes found through this selector's imports.
	 * <p>If this predicate returns {@code true} for a given fully-qualified
	 * class name, said class will not be considered as an imported configuration
	 * class, bypassing class file loading as well as metadata introspection.
	 * @return the filter predicate for fully-qualified candidate class names
	 * of transitively imported configuration classes, or {@code null} if none
	 * @since 5.2.4
	 */
	@Nullable
	default Predicate<String> getExclusionFilter() {
		return null;
	}

}
```

### 自定义@Enable模块

#### 基于注解驱动实现

@EnableHelloWorld

#### 基于接口驱动实现

@EnableServer

HelloWorldImportSelector->HelloWorldConfiguration->HelloWorld

- HelloWorldImportSelector自定义一个类，实现ImportSelector接口
- HelloWorldConfiguration是一个配置类，用于获取Bean-HelloWorld
- HelloWorld是一个字符串类
- 定义注解EnableHelloWorld

>HelloWorldImportSelector类

实现ImportSelector接口。自定义ImportSelector可以在里面书写一些配置逻辑，满足则配置，不满足就不配置，写法灵活~

```java
/**
 * HelloWorld {@link org.springframework.context.annotation.ImportSelector} 实现
 */
public class HelloWorldImportSelector implements ImportSelector {
    @Override
    public String[] selectImports(AnnotationMetadata importingClassMetadata) {
//        return new String[]{"com.example.demo.config.HelloWorldConfig"};
        //getName 类全名
        return new String[]{HelloWorldConfig.class.getName()};

    }
}
```

>HelloWorldConfiguration配置类

```java
@Configuration
public class HelloWorldConfiguration {
    @Bean
    public String helloWorld() {
        return "Hello World";
    }
}
```

> EnableHelloWorld注解

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Import(HelloWorldImportSelector.class)
public @interface EnableHelloWorld {
}
```

> 启动运行

```java
@EnableHelloWorld // 开启Enable注解，HelloWorldConfiguration中定义的Bean都可获取到~
@ComponentScan(basePackages = "com.example.demo.bootstrap")
public class EnableHelloWorldApplication {
    public static void main(String[] args) {
        ConfigurableApplicationContext context = new SpringApplicationBuilder(EnableHelloWorldApplication.class)
                .web(WebApplicationType.NONE)
                .run(args);
        String str = context.getBean("helloWorld", String.class);
        System.out.println(str);
        context.close();
    }
}
```

## Spring 条件配置

### 条件注解举例

| Spring注解   | 场景说明       | 起始版本 |
| ------------ | -------------- | -------- |
| @Profile     | 配置化条件装配 | 3.1      |
| @Conditional | 编程条件装配   | 4.0      |

### 实现方式

#### 配置方式

`@Profile`

Spring Framework 4.0后Profile采用的Condition进行实现的。

#### 编程方式

`@Conditional`

```java
@Target({ ElementType.TYPE, ElementType.METHOD })
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Conditional(OnClassCondition.class)
public @interface ConditionalOnClass {

	/**
	 * The classes that must be present. Since this annotation is parsed by loading class
	 * bytecode, it is safe to specify classes here that may ultimately not be on the
	 * classpath, only if this annotation is directly on the affected component and
	 * <b>not</b> if this annotation is used as a composed, meta-annotation. In order to
	 * use this annotation as a meta-annotation, only use the {@link #name} attribute.
	 * @return the classes that must be present
	 */
	Class<?>[] value() default {};

	/**
	 * The classes names that must be present.
	 * @return the class names that must be present.
	 */
	String[] name() default {};

}
```

---

```java
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
```

### 自定义条件装配

#### 基于配置方式

计算服务，多整数求和

`@Prifile("Java7")：for循环实现`

`@Prifile("Java8")：lambda实现`

`启动类设置好Prifiles属性`

```java
public interface Calculate {
    Integer sum(Integer... val);
}

@Profile("Java7")
@Service
public class Java7Calculate implements Calculate {
    @Override
    public Integer sum(Integer... val) {
        System.out.println("==================Java7==================");
        int sum = 0;
        for (Integer integer : val) {
            sum += integer;
        }
        return sum;
    }
}

@Profile("Java8")
@Service
public class Java8Calculate implements Calculate {
    @Override
    public Integer sum(Integer... val) {
        System.out.println("==================Java8==================");
        return Stream.of(val).reduce(0, Integer::sum);
    }
}
```

---

```java
@ComponentScan(basePackages = "com.example.demo")
public class ProfileApplication {
    public static void main(String[] args) {
        ConfigurableApplicationContext context = new SpringApplicationBuilder(ProfileApplication.class)
                .web(WebApplicationType.NONE)
                .profiles("Java8")
                .run();
        Calculate bean = context.getBean(Calculate.class);
        System.out.println(bean.sum(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));

    }
}
```

#### 基于编程方式

自定义注解`@ConditionalOnProperty`判断Spring应用上下文xx配置是否存在/匹配

Condition实现类`OnSystemPropertyCondition`定义条件，符合条件则触发，不符合则不触发

```java
/**
 * Java系统属性判断
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.TYPE, ElementType.METHOD})
@Documented
@Conditional(OnSystemPropertyCondition.class)
public @interface ConditionalOnSystemProperty {
    // java系统属性名
    String name();

    // java系统属性值
    String value();
}
```

----

```java
public class OnSystemPropertyCondition implements Condition {

    @Override
    public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
        // 获取元信息.获得ConditionalOnSystemProperty的原信息（name和value的值）
        Map<String, Object> annotationAttributes = metadata.getAnnotationAttributes(ConditionalOnSystemProperty.class.getName());
        assert annotationAttributes != null;
        String name = String.valueOf(annotationAttributes.get("name"));
        String value = String.valueOf(annotationAttributes.get("value"));
        return name.equals(value);
    }
}
```

---

```java
@SpringBootApplication(scanBasePackages = "com.example.demo")
public class ConditionalOnSystemPropertyApplication {

    // 条件满足产生类"hello",不满足则不产生。
    @Bean
    @ConditionalOnSystemProperty(name = "java", value = "java")
    public String hello() {
        return "hello";
    }

    public static void main(String[] args) {
        ConfigurableApplicationContext context = new SpringApplicationBuilder(EnableHelloWorldApplication.class)
                .web(WebApplicationType.NONE)
                .run(args);
        String str = context.getBean("hello", String.class);
        System.out.println(str);
        context.close();
    }
}
```

## Spring Boot自动装配

WebMvcAutoConfiguration

```java
@Configuration(proxyBeanMethods = false) // 模式注解。是Component的“派生”注解
@ConditionalOnWebApplication(type = Type.SERVLET) // 实际上是Spring 4的Condition注解实现的
@ConditionalOnClass({ Servlet.class, DispatcherServlet.class, WebMvcConfigurer.class })
@ConditionalOnMissingBean(WebMvcConfigurationSupport.class)
@AutoConfigureOrder(Ordered.HIGHEST_PRECEDENCE + 10)
@AutoConfigureAfter({ DispatcherServletAutoConfiguration.class, TaskExecutionAutoConfiguration.class,
		ValidationAutoConfiguration.class })
public class WebMvcAutoConfiguration {
    
}
```

- 定义：基于约定大于配置的原则，实现Spring组件自动装配的目的。
- 装配：模式注解、@Enable模块、条件装配、工厂加载机制
- 实现：激活自动装配、实现自动装配、配置自动装配的实现

### 底层装配技术

- Spring 模式注解装配
- Spring `@Enable`模块装配
- Spring 条件装配
- Spring 工厂加载机制
  - 实现类：`SpringFactoriesLoader`
  - 配置资源：`META-INF/spring.factories`

### 自动装配举例

参考`META-INF/spring.factories`

### 实现方法

1.激活自动装配：`@EnableAutoConfiguration`

2.实现自动装配：`XXXAutoConfiguration`

3.配置自动装配实现：`META-INF/spring.factories`

### 自定义自动装配

- 我们自定义的HelloWorldAutoConfig，有条件判断注解，条件判断为True
  - 条件判断：name == key
  - 模式注解：`@Configuration`
  - `@Enable`模块：`@EnableHelloWorld` 会加载-->`HelloWorldImportSelector`会加载-->`HelloWorldConfig`-->最终生成一个bean

---

> 自定义自动装配流程如下：

- 自定义一个`HelloWorldAutoConfiguration`配置类。

- 在`resources`下新建`META-INF`目录，在`META-INF`目录下创建文件`spring.factories`,在该目录下配置自动装配信息

  ```properties
  # Auto Configure
  org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
  com.example.demo.config.HelloWorldAutoConfiguration
  ```

- 这样一个最简单的自动装配就完成啦！
- 如果想要更复杂的配置，可以加Condition判断，自定义的@EnableXX等等~

----

> 一个自定义自动装配的Demo

- 配置类

  ```java
  @Configuration
  public class HelloWorldAutoConfiguration {
      @Bean
      public Object getObj() {
          return new Object();
      }
  }
  ```

- resource/META-INF/spring.factories

  ```properties
  # Auto Configure
  org.springframework.boot.autoconfigure.EnableAutoConfiguration=\
  com.example.demo.config.HelloWorldAutoConfiguration
  ```

- 运行

  ```java
  /**
   * {@link EnableAutoConfiguration}
   * 自动装配
   */
  @EnableAutoConfiguration
  public class EnableAutoConfigurationApplication {
      public static void main(String[] args) {
          ConfigurableApplicationContext context = new SpringApplicationBuilder(EnableAutoConfigurationApplication.class)
                  .web(WebApplicationType.NONE)
                  .run(args);
          System.out.println(context.getBean("getObj", Object.class));
          context.close();
      }
  }
  ```

# Web MVC核心

- 理解 Spring Web MVC架构
- 认识 Spring Web MVC
- 简化 Spring Web MVC

## 理解Web MVC架构

### 基础架构：Servlet

Web Browser 发送HTTP请求给Web Server  --> Web Server 服务把请求转发到Servlet容器中。--> Servlet容器进行一系列操作。

### Servlet特点

- 请求/响应式（Request/Response）
- 屏蔽网络通讯的细节
- 完整的生命周期

### Servlet职责

- 处理请求
- 资源管理
- 视图渲染

### Web MVC架构

> 核心架构：前端控制器（Front Controller）

<img src="img/boot/FCMainClass.gif">

- 资源：[Core J2EE Patterns](http://www.corej2eepatterns.com/FrontController.htm)
- 实现：Spring Web MVC DispatcherServlet
  - [DispatcherServlet (Spring Framework)](https://docs.spring.io/spring-framework/docs/1.0.0/javadoc-api/org/springframework/web/servlet/DispatcherServlet.html)
  - [DispatcherServlet (Spring Framework 5.0.0.RELEASE API)](https://docs.spring.io/spring-framework/docs/5.0.0.RELEASE/javadoc-api/org/springframework/web/servlet/DispatcherServlet.html)

## 认识Web MVC

### 一般认识

Spring Framework时代的一般认识 

- 实现Controller

```java
@Controller
public class Hello{
    @RequestMapping("")
    public String index(){
        return "Hello";
    }
}
```

- 配置 Web MVC组件

```xml
<context:component-scan base-package="xxx.xx.web"/>

<bean class="org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerMapping"/>

<bea class="org.springframework.web.servlet.mvc.method.annotation.RequestMappingHandlerAdapter"/>

<bean id="viewResolver" class="org.springframework.web.servlet.view.InternalResourceViewResolver">
	<property name="viewClass" value="org.springframework.web.servlet.view.JstlView"></property>
    <property name="prefix" value="/WEB-INF/views/"></property>
    <property name="suffix" value=".jsp"></property>
</bean>
```

- 部署DispatcherServlet [在web.xml中配置]

```xml
<servlet>
    <servlet-name>app</servlet-name>
    <servlet-class>org.springframework.web.servlet.DispatcherServlet</servlet-class>
    <init-param>
        <param-name>contextConfigLocation</param-name>
        <param-value>/WEB-INF/app-context.xml</param-value>
    </init-param>
   <!--  --> 
    <load-on-startup>1</load-on-startup>
</servlet>
<servlet-mapping>
    <servlet-name>app</servlet-name>
    <url-pattern>/</url-pattern>
</servlet-mapping>
```

- 使用可执行Tomcat Maven插件 [这个好像可以打包成jar包, 我们可以直接运行这个jar包！]

```xml
<plugin>
    <groupId>org.apache.tomcat.maven</groupId>
    <artifactId>tomcat7-maven-plugin</artifactId>
    <version>2.2</version>
    <executions>
        <execution>
            <id>tomcat-run</id>
            <goals>
                <goal>exec-war-only</goal>
            </goals>
            <phase>package</phase>
            <configuration>
                <!-- ServletContext path-->
                <path>/</path>
            </configuration>
        </execution>
    </executions>
</plugin>
```

**小提示**：SpringBoot spring-boot-starter-parent 中的 spring-boot-dependencies 有定义各个包需要的版本！

`maven命令 [打包]：mvn -Dmaven.test.skpi -u clean package`

### 重新认识

Spring Framework时代的重新认识。

- Web MVC 核心组件
- Web MVC 注解驱动
- Web MVC 自动装配

#### Web MVC核心组件

- 处理器管理
  - 映射：handlerMapping [ RequestMappingHandlerMapping ]，用于处理映射
  - 适配：HandlerAdapter
  - 执行：HandlerExecutionChain [ 处理器的执行链 ]
- 页面渲染
  - 视图解析器：ViewResolver
  - 国际化：LocaleResolver、LocaleContextResolver
  - 个性化：ThemeResolver
- 异常处理
  - 异常解析：HandlerExceptionResolver

| Bean type                                                    | Explanation                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| `HandlerMapping`                                             | 映射请求（Request）到处理器（Handler）加上器关联的拦截器（HandlerInterceptor）列表，其映射关系基于不同的`HandlerMapping`实现的一些标注细节。其中两种主要`HandlerMapping`实现，RequestMappingHandlerMapping支持标注`@RequestMapping`的方法，`SimpleUrlHandlerMapping`维护精确的URI路径与处理器的映射。 |
| `HandlerAdapter`                                             | ‎帮助`DispatcherServlet`调用请求处理器（Handler），无需关注其中实际的调用细节。比如，调用注解实现的`Controller`需要解析其关联的注解。`HandlerAdapter`的主要目的是为了屏蔽与`DispatcherServlet`之间的诸多细节。 |
| [`HandlerExceptionResolver`](https://docs.spring.io/spring-framework/docs/current/reference/html/web.html#mvc-exceptionhandlers) | 解析异常，可能策略是将异常处理映射到其他处理器（Handlers）、或到某个HTML错误页面，或者其他。 |
| [`ViewResolver`](https://docs.spring.io/spring-framework/docs/current/reference/html/web.html#mvc-viewresolver) | 从处理器（Handler）返回字符类型的逻辑视图名称解析出实际的View对象，该对象将渲染后的内容输出到HTTP响应中。 |
| [`LocaleResolver`](https://docs.spring.io/spring-framework/docs/current/reference/html/web.html#mvc-localeresolver), [LocaleContextResolver](https://docs.spring.io/spring-framework/docs/current/reference/html/web.html#mvc-timezone) | 从客户端解析出Locale，为其实现国际化视图。                   |
| [`MultipartResolver`](https://docs.spring.io/spring-framework/docs/current/reference/html/web.html#mvc-multipart) | 解析多部分请求（如Web浏览器文件上传）的抽象实现。            |

Spring Web MVC的运行流程参看MVC相关笔记。

#### Web MVC注解驱动

> 基本配置步骤：具体看我的MVC笔记奥

- 注解配置：@Configuration（Spring范式注解）
- 组件激活：@EnableWebMvc（Spring模块装配）
- 自定义组件：WebMvcConfigurer（Spring Bean）

> 常用注解

- 模型属性：@ModelAttribute
- 请求头：@RequestHeader
- Cookie：@CookieValue
- 校验参数：@Valid、@Validated
- 注解处理：@ExceptionHandler
- 切面通知：@ControllerAdvice
  - @Component的专门化，用于声明@ExceptionHandler、@InitBinder或@ModelAttribute方法的类，以便在多个@Controller类之间共享。
  - 结合@ModelAttribute注解使用。具体看MVC相关笔记。
    - 印象中，@ModelAttribute修饰的方法数据会放在ModelAndView里。

#### Web MVC自动装配

- Servlet 依赖 3.0+
- Servlet SPI：ServletContainerInitializer
- Spring适配：SpringServletContainerInitializer
- Spring SPI：WebApplicationInitializer
- 编程驱动：AbstractDispatcherServletInitializer
- 注解驱动：AbstractAnnotationConfigDispatcherServletInitializer

Servlet 3.1 规范：Servlet启动的时候，onStartup方法会被回调。什么意思呢？看看SPI接口。

```java
public interface ServletContainerInitializer {
    // ServletContext：动态添加一些功能，如addServlet addJspFile
    void onStartup(Set<Class<?>> var1, ServletContext var2) throws ServletException;
}
```

----

SpringServletcontainerInitializer实现了这个接口

```java
// 筛选器 实现了WebApplicationInitializer的类才会被调用onStartup方法
@HandlesTypes(WebApplicationInitializer.class) 
public class SpringServletContainerInitializer implements ServletContainerInitializer {

	/**
	 * Delegate the {@code ServletContext} to any {@link WebApplicationInitializer}
	 * implementations present on the application classpath.
	 * <p>Because this class declares @{@code HandlesTypes(WebApplicationInitializer.class)},
	 * Servlet 3.0+ containers will automatically scan the classpath for implementations
	 * of Spring's {@code WebApplicationInitializer} interface and provide the set of all
	 * such types to the {@code webAppInitializerClasses} parameter of this method.
	 * <p>If no {@code WebApplicationInitializer} implementations are found on the classpath,
	 * this method is effectively a no-op. An INFO-level log message will be issued notifying
	 * the user that the {@code ServletContainerInitializer} has indeed been invoked but that
	 * no {@code WebApplicationInitializer} implementations were found.
	 * <p>Assuming that one or more {@code WebApplicationInitializer} types are detected,
	 * they will be instantiated (and <em>sorted</em> if the @{@link
	 * org.springframework.core.annotation.Order @Order} annotation is present or
	 * the {@link org.springframework.core.Ordered Ordered} interface has been
	 * implemented). Then the {@link WebApplicationInitializer#onStartup(ServletContext)}
	 * method will be invoked on each instance, delegating the {@code ServletContext} such
	 * that each instance may register and configure servlets such as Spring's
	 * {@code DispatcherServlet}, listeners such as Spring's {@code ContextLoaderListener},
	 * or any other Servlet API componentry such as filters.
	 * @param webAppInitializerClasses all implementations of
	 * {@link WebApplicationInitializer} found on the application classpath
	 * @param servletContext the servlet context to be initialized
	 * @see WebApplicationInitializer#onStartup(ServletContext)
	 * @see AnnotationAwareOrderComparator
	 */
	@Override
	public void onStartup(@Nullable Set<Class<?>> webAppInitializerClasses, ServletContext servletContext)
			throws ServletException {

		List<WebApplicationInitializer> initializers = Collections.emptyList();

		if (webAppInitializerClasses != null) {
			initializers = new ArrayList<>(webAppInitializerClasses.size());
			for (Class<?> waiClass : webAppInitializerClasses) {
				// Be defensive: Some servlet containers provide us with invalid classes,
				// no matter what @HandlesTypes says...
				if (!waiClass.isInterface() && !Modifier.isAbstract(waiClass.getModifiers()) &&
						WebApplicationInitializer.class.isAssignableFrom(waiClass)) {
					try {
						initializers.add((WebApplicationInitializer)
								ReflectionUtils.accessibleConstructor(waiClass).newInstance());
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

@HandlesTypes的解释<a href="https://www.cnblogs.com/hello-shf/p/10926271.html">相关博客</a>

　　简单来说，当实现了Servlet3.0规范的容器（比如tomcat7及以上版本）启动时，通过SPI扩展机制自动扫描所有已添加的jar包下的META-INF/services/javax.servlet.ServletContainerInitializer中指定的全路径的类，并实例化该类，然后回调META-INF/services/javax.servlet.ServletContainerInitializer文件中指定的ServletContainerInitializer的实现类的onStartup方法。 如果该类存在@HandlesTypes注解，并且在@HandlesTypes注解中指定了我们感兴趣的类，所有实现了这个类的onStartup方法将会被调用。

　　再直白一点来说，存在web.xml的时候，Servlet容器会根据web.xml中的配置初始化我们的jar包（也可以说web.xml是我们的jar包和Servlet联系的中介）。而在Servlet3.0容器初始化时会调用jar包META-INF/services/javax.servlet.ServletContainerInitializer中指定的类的实现（javax.servlet.ServletContainerInitializer中的实现替代了web.xml的作用，而所谓的在@HandlesTypes注解中指定的感兴趣的类，可以理解为具体实现了web.xml的功能，当然也可以有其他的用途）

##### Servlet SPI

配合@HandlesType

##### Spring适配

SpringServletContainerInitializer

##### Spring SPI

基础接口：WebApplicationinitializer [直接实现裸接口]

编程驱动：AbstractDispatcherServletInitializer

注解驱动：AbstractAnnotationConfigDispatcherServletInitializer

##### 示例重构

```java
/**
 * DispatcherServlet配置类，配置扫描web（Controller类）包
 */
@ComponentScan(basePackages = "com.example.demo.controller")
public class DispatcherServletConfiguration {
}
```

----

```java
public class DefaultAnnotationConfigDispatcherServletInitializer extends AbstractAnnotationConfigDispatcherServletInitializer {

    @Override
    protected Class<?>[] getRootConfigClasses() {
        return new Class[0];
    }

    @Override
    protected Class<?>[] getServletConfigClasses() {
        return new Class[]{DispatcherServletConfiguration.class};
    }

    @Override
    protected String[] getServletMappings() {
        return new String[0];
    }

}
```

## 简化Web MVC

SpringBoot时代的简化

- 完全自动装配
- 装配条件
- 外部化配置

### 完全自动装配

- DispatcherServlet：DispatcherServletAutoConfiguration
- 替换@EnableWebMvc：WebMvcAutoConfiguration
- Servlet容器：ServletWebServerFactoryAutoConfiguration（通过Spring Bean的方式运行）

把这几个类的源码看一下。

#### 理解自动装配顺序性

绝对顺序：@AtuoConfigureOrder

相对顺序：@AutoConfigureAfter

### 装配条件

- Web类型：Servlet
- API依赖：Servlet、Spring Web MVC
- Bean依赖：WebMvcConfigurationSupport

### 外部化配置

- Web MVC配置：WebMvcProperties
- 资源配置：ResourceProperties





