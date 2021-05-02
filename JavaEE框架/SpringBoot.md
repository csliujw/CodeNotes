# SpringBoot

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

# 核心特性

## Spring Boot三大特性

- 组件自动装配：Web MVC、Web Flux、JDBC等
- 嵌入式Web容器：Tomcat、Jetty以及Undertow
- 生产准备特性：指标、健康检查、外部化配置等

## 组件自动装配

- 激活：@EnableAutoConfiguration
- 配置：/META-INF/spring.factories（这个目录是相对于Classpath而言，META元，INF信息，META-INF元信息；spring.factories工厂模式，key-value键值对的配置信息）
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





## Spring Framework手动配置

- 定义：一种用于声明在`应用`中扮演“组件”角色的注解
- 举例：@Component、@Service、@Configuration[标注这是一个配置]
- 装配：`<context:component=scan>` 或 @ComponentScan

### 

