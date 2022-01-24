# 快速入门

## SpringBoot介绍

能快速创建出生产级别的Spring应用

> 优点

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

> 缺点

- 迭代快，需要时刻关注变化

- 封装太深，内部原理复杂，不易精通

> 时代背景

[James Lewis and Martin Fowler (2014)](https://martinfowler.com/articles/microservices.html)  提出微服务完整概念。https://martinfowler.com/microservices/

In short, the **microservice architectural style** is an approach to developing a single application as a **suite of small services**, each **running in its own process** and communicating with **lightweight** mechanisms, often an **HTTP** resource API. These services are **built around business capabilities** and **independently deployable** by fully **automated deployment** machinery. There is a **bare minimum of centralized management** of these services, which may be **written in different programming languages** and use different data storage technologies.-- [James Lewis and Martin Fowler (2014)](https://martinfowler.com/articles/microservices.html)

- 微服务是一种架构风格
- 一个应用拆分为一组小型服务
- 每个服务运行在自己的进程内，也就是可独立部署和升级
- 服务之间使用轻量级HTTP交互
- 服务围绕业务功能拆分
- 可以由全自动部署机制独立部署
- 去中心化，服务自治。服务可以使用不同的语言、不同的存储技术

> 分布式

![image](https://cdn.nlark.com/yuque/0/2020/png/1613913/1599562347965-a617a866-4270-44e9-9c5b-ced552683eda.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_14%2Ctext_YXRndWlndS5jb20g5bCa56GF6LC3%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10)

分布式的困难

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

分布式的解决

- SpringBoot + SpringCloud

![image](https://cdn.nlark.com/yuque/0/2020/png/1613913/1599799119457-841ef47a-6585-4ca4-8e3d-8298e796012c.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_10%2Ctext_YXRndWlndS5jb20g5bCa56GF6LC3%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10)

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

### 阿里云镜像创建

创建工程时，切换选择starter服务路径，然后手工输入阿里云地址即可，地址：http://start.aliyun.com或https://start.aliyun.com

![image-20211122163605950](img\boot\image-20211122163605950.png)

阿里为了便于自己公司开发使用，特此在依赖坐标中添加了一些阿里自主的技术，也是为了推广自己的技术吧，所以在依赖选择列表中，你有了更多的选择。此外，阿里提供的地址更符合国内开发者的使用习惯，里面有一些SpringBoot官网上没有给出的坐标。

![image-20211122163937408](img\boot\image-20211122163937408.png)

阿里云地址默认创建的SpringBoot工程版本是<font color="#ff0000"><b>2.4.1</b></font>，所以如果你想更换其他的版本，创建项目后在pom文件中手工修改即可，别忘了刷新一下，加载新版本信息。

<font color="#ff0000"><b>注意</b></font>：阿里云提供的工程创建地址初始化完毕后和使用SpringBoot官网创建出来的工程略有区别，主要是在配置文件的形式上有区别,这个信息在后面讲解SpringBoot程序的执行流程时给大家揭晓。

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

## 基础配置

### 属性配置

SpringBoot通过配置文件application.properties就可以修改默认的配置

![image-20211123165428245](img\boot\image-20211123165428245.png)

将默认端口改为80端口。properties格式的文件书写规范是key=value

```properties
name=itheima
```

<img src="img\image-20211123165719091.png" alt="image-20211123165719091" style="zoom:80%;" />

```properties
server.port=80
```

以前修改端口在 tomcat 服务器的配置文件中改，现在在 SpringBoot专 用的配置文件中改，简化开发者配置的书写位置，集中管理。

1. SpringBoot程序可以在application.properties文件中进行属性配置
2. application.properties文件中只要输入要配置的属性关键字就可以根据提示进行设置
3. SpringBoot将配置信息集中在一个文件中写，不管你是服务器的配置，还是数据库的配置，总之都写在一起，逃离一个项目十几种配置文件格式的尴尬局面

**总结**

**SpringBoot默认配置文件是application.properties**

做完了端口的配置，趁热打铁，再做几个配置，目前项目启动时会显示一些日志信息，就来改一改这里面的一些设置。

**关闭运行日志图表（banner)**

```properties
spring.main.banner-mode=off
```

**设置运行日志的显示级别**

```properties
logging.level.root=debug
```

统一了配置文件的格式

​		我们现在配置了3个信息，但是又有新的问题了。这个配置是随便写的吗？什么都能配？有没有一个东西显示所有能配置的项呢？此外这个配置和什么东西有关呢？会不会因为我写了什么东西以后才可以写什么配置呢？比如我现在没有写数据库相关的东西，能否配置数据呢？一个一个来，先说第一个问题，都能配置什么。

​		打开SpringBoot的官网，找到SpringBoot官方文档，打开查看附录中的Application Properties就可以获取到对应的配置项了，网址奉上：https://docs.spring.io/spring-boot/docs/current/reference/html/application-properties.html#application-properties

​		能写什么的问题解决了，再来说第二个问题，这个配置项和什么有关。在pom中注释掉导入的spring-boot-starter-web，然后刷新工程，你会发现配置的提示消失了。闹了半天是设定使用了什么技术才能做什么配置。也合理，不然没有使用对应技术，配了也是白配。

<font color="#f0f"><b>温馨提示</b></font>

​		所有的starter中都会依赖下面这个starter，叫做spring-boot-starter。这个starter是所有的SpringBoot的starter的基础依赖，里面定义了SpringBoot相关的基础配置，关于这个starter我们到开发应用篇和原理篇中再深入讲解。

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter</artifactId>
    <version>2.5.4</version>
    <scope>compile</scope>
</dependency>
```

**总结**

1. SpringBoot中导入对应starter后，提供对应配置属性
2. 书写SpringBoot配置采用关键字+提示形式书写

### 配置文件分类

​		现在已经能够进行SpringBoot相关的配置了，但是properties格式的配置写起来总是觉得看着不舒服，所以就期望存在一种书写起来更简便的配置格式提供给开发者使用。有吗？还真有，SpringBoot除了支持properties格式的配置文件，还支持另外两种格式的配置文件。三种配置文件格式分别如下:

- properties格式
- yml格式
- yaml格式


​		一看到全新的文件格式，各位小伙伴肯定想，这下又要学习新的语法格式了。怎么说呢？从知识角度来说，要学，从开发角度来说，不用学。为什么呢？因为SpringBoot的配置在Idea工具下有提示啊，跟着提示走就行了。下面列举三种不同文件格式配置相同的属性范例，先了解一下。

- application.properties（properties格式）

```properties
server.port=80
```

- application.yml（yml格式）

```YML
server:
  port: 81
```

- application.yaml（yaml格式）

```yaml
server:
  port: 82
```

​		仔细看会发现yml格式和yaml格式除了文件名后缀不一样，格式完全一样，是这样的，yml和yaml文件格式就是一模一样的，只是文件后缀不同，所以可以合并成一种格式来看。那对于这三种格式来说，以后用哪一种比较多呢？记清楚，以后基本上都是用yml格式的，本课程后面的所有知识都是基于yml格式来制作的，以后在企业开发过程中用这个格式的机会也最多，一定要重点掌握。

**总结**

1. SpringBoot提供了3种配置文件的格式
    - properties（传统格式/默认格式）
    - **yml**（主流格式）
    - yaml

**思考**

​		现在我们已经知道使用三种格式都可以做配置了，万一三个都写了，按什么优先级来？

#### 配置文件优先级

​		其实三个文件如果共存的话，谁生效说的就是配置文件加载的优先级别。先说一点，虽然以后这种情况很少出现，但是这个知识还是可以学习一下的。我们就让三个配置文件书写同样的信息，比如都配置端口，然后我们让每个文件配置的端口号都不一样，最后启动程序后看启动端口是多少就知道谁的加载优先级比较高了。

- application.properties（properties格式）

```properties
server.port=80
```

- application.yml（yml格式）

```YML
server:
  port: 81
```

- application.yaml（yaml格式）

```yaml
server:
  port: 82
```

​		启动后发现目前的启动端口为80，把80对应的文件删除掉，然后再启动，现在端口又改成了81。现在我们就已经知道了3个文件的加载优先顺序是什么。

```XML
application.properties  >  application.yml  >  application.yaml
```

​		虽然得到了一个知识结论，但是我们实际开发的时候还是要看最终的效果为准。也就是你要的最终效果是什么自己是明确的，上述结论只能帮助你分析结论产生的原因。这个知识了解一下就行了，因为以后同时写多种配置文件格式的情况实在是较少。

​		最后我们把配置文件内容给修改一下

- application.properties（properties格式）

```properties
server.port=80
spring.main.banner-mode=off
```

- application.yml（yml格式）

```YML
server:
  port: 81
logging: 
  level: 
    root: debug
```

- application.yaml（yaml格式）

```yaml
server:
  port: 82
```

​		我们发现不仅端口生效了，最终显示80，同时其他两条配置也生效了，看来每个配置文件中的项都会生效，只不过如果多个配置文件中有相同类型的配置会优先级高的文件覆盖优先级的文件中的配置。如果配置项不同的话，所有的配置项都会生效。

**总结**

1. 配置文件间的加载优先级	properties（最高）>  yml  >  yaml（最低）
2. 不同配置文件中相同配置按照加载优先级相互覆盖，不同配置文件中不同配置全部保留 

#### **自动提示功能消失解决方案**

​		在做程序的过程中，可能有些小伙伴会基于各种各样的原因导致配置文件中没有提示，这个确实很让人头疼，所以下面给大家说一下如果自动提示功能消失了怎么解决。

​		先要明确一个核心，就是自动提示功能不是SpringBoot技术给我们提供的，是我们在Idea工具下编程，这个编程工具给我们提供的。明白了这一点后，再来说为什么会出现这种现象。其实这个自动提示功能消失的原因还是蛮多的，如果想解决这个问题，就要知道为什么会消失，大体原因有如下2种：

1. Idea认为你现在写配置的文件不是个配置文件，所以拒绝给你提供提示功能

2. Idea认定你是合理的配置文件，但是Idea加载不到对应的提示信息

    这里我们主要解决第一个现象，第二种现象到原理篇再讲解。第一种现象的解决方式如下：

**步骤①**：打开设置，【Files】→【Project Structure...】

![image-20211126160548690](img\boot\image-20211126160548690.png)

**步骤②**：在弹出窗口中左侧选择【Facets】，右侧选中Spring路径下对应的模块名称，也就是你自动提示功能消失的那个模块

<img src="C:/development/Code/note/CodeNotes/JavaEE Framework/img/image-20211126160726589.png" alt="image-20211126160726589" style="zoom:67%;" />![image-20211126160844372](img\boot\image-20211126160844372.png)

**步骤③**：点击Customize Spring Boot按钮，此时可以看到当前模块对应的配置文件是哪些了。如果没有你想要称为配置文件的文件格式，就有可能无法弹出提示

![image-20211126160946448](img\boot\image-20211126160946448.png)<img src="C:/development/Code/note/CodeNotes/JavaEE Framework/img/image-20211126160954338.png" alt="image-20211126160954338" style="zoom:80%;" />

**步骤④**：选择添加配置文件，然后选中要作为配置文件的具体文件就OK了

<img src="C:/development/Code/note/CodeNotes/JavaEE Framework/img/image-20211126161145082.png" alt="image-20211126161145082" style="zoom:80%;" /><img src="img\boot\image-20211126161156324.png" alt="image-20211126161156324" style="zoom: 67%;" />

​		到这里就做完了，其实就是Idea的一个小功能

![image-20211126161301699](img\boot\image-20211126161301699.png)



**总结**

1. 指定SpringBoot配置文件

    - Setting → Project Structure → Facets
    - 选中对应项目/工程
    - Customize Spring Boot
    - 选择配置文件


### yaml文件

SpringBoot的配置以后主要使用yml结尾的这种文件格式，并且在书写时可以通过提示的形式加载正确的格式。但是这种文件还是有严格的书写格式要求的。下面就来说一下具体的语法格式。

YAML（YAML Ain't Markup Language），一种数据序列化格式。具有容易阅读、容易与脚本语言交互、以数据为核心，重数据轻格式的特点。常见的文件扩展名有两种：

- .yml格式（主流）

- .yaml格式

    具体的语法格式要求如下：

1. 大小写敏感
2. 属性层级关系使用多行描述，**每行结尾使用冒号结束**
3. 使用缩进表示层级关系，同层级左侧对齐，只允许使用空格（不允许使用Tab键）
4. 属性值前面添加空格（属性名与属性值之间使用冒号+空格作为分隔）
5. #号 表示注释

上述规则不要死记硬背，按照书写习惯慢慢适应，并且在Idea下由于具有提示功能，慢慢适应着写格式就行了。核心的一条规则要记住，<font color="#ff0000"><b>数据前面要加空格与冒号隔开</b></font>。

下面列出常见的数据书写格式，熟悉一下

```YAML
boolean: TRUE  						#TRUE,true,True,FALSE,false，False均可
float: 3.14    						#6.8523015e+5  #支持科学计数法
int: 123       						#0b1010_0111_0100_1010_1110    #支持二进制、八进制、十六进制
null: ~        						#使用~表示null
string: HelloWorld      			#字符串可以直接书写
string2: "Hello World"  			#可以使用双引号包裹特殊字符
date: 2018-02-17        			#日期必须使用yyyy-MM-dd格式
datetime: 2018-02-17T15:02:31+08:00  #时间和日期之间使用T连接，最后使用+代表时区
```

此外，yaml格式中也可以表示数组，在属性名书写位置的下方使用减号作为数据开始符号，每行书写一个数据，减号与数据间空格分隔。

```YAML
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

**总结**

1. yaml语法规则
    - 大小写敏感
    - 属性层级关系使用多行描述，每行结尾使用冒号结束
    - 使用缩进表示层级关系，同层级左侧对齐，只允许使用空格（不允许使用Tab键）
    - 属性值前面添加空格（属性名与属性值之间使用冒号+空格作为分隔）
    - #号 表示注释
2. 注意属性名冒号后面与数据之间有一个**空格**
3. 字面值、对象数据格式、数组数据格式

**思考**

​		现在我们已经知道了yaml具有严格的数据格式要求，并且已经可以正确的书写yaml文件了，那这些文件书写后其实是在定义一些数据。这些数据是给谁用的呢？大部分是SpringBoot框架内部使用，但是如果我们想配置一些数据自己使用，能不能用呢？答案是可以的，那如何读取yaml文件中的数据呢？咱们下一节再说。

### yaml数据读取

​		对于yaml文件中的数据，其实你就可以想象成这就是一个小型的数据库，里面保存有若干数据，每个数据都有一个独立的名字，如果你想读取里面的数据，肯定是支持的，下面就介绍3种读取数据的方式。

#### 读取单一数据

​		yaml中保存的单个数据，可以使用Spring中的注解@Value读取单个数据，属性名引用方式：<font color="#ff0000"><b>${一级属性名.二级属性名……}</b></font>

<img src="img\boot\image-20211126180433356.png" alt="image-20211126180433356" style="zoom:80%;" />

​		记得使用@Value注解时，要将该注解写在某一个指定的Spring管控的bean的属性名上方，这样当bean进行初始化时候就可以读取到对应的单一数据了。

**总结**

1. 使用@Value配合SpEL读取单个数据
2. 如果数据存在多层级，依次书写层级名称即可

#### 读取全部数据

​		读取单一数据可以解决读取数据的问题，但是如果定义的数据量过大，这么一个一个书写肯定会累死人的，SpringBoot提供了一个对象，能够把所有的数据都封装到这一个对象中，这个对象叫做Environment，使用自动装配注解可以将所有的yaml数据封装到这个对象中

<img src="img\boot\image-20211126180738569.png" alt="image-20211126180738569" style="zoom:80%;" />

​		数据封装到了Environment对象中，获取属性时，通过Environment的接口操作进行，具体方法是getProperties（String），参数填写属性名即可

**总结**

1. 使用Environment对象封装全部配置信息
2. 使用@Autowired自动装配数据到Environment对象中

#### 读取对象数据

​		单一数据读取书写比较繁琐，全数据读取封装的太厉害了，每次拿数据还要一个一个的getProperties（）,总之用起来都不是很舒服。由于Java是一个面向对象的语言，很多情况下，我们会将一组数据封装成一个对象。SpringBoot也提供了可以将一组yaml对象数据封装一个Java对象的操作

​		首先定义一个对象，并将该对象纳入Spring管控的范围，也就是定义成一个bean，然后使用注解@ConfigurationProperties指定该对象加载哪一组yaml中配置的信息。

<img src="img\boot\image-20211126181126382.png" alt="image-20211126181126382" style="zoom:80%;" />

​		这个@ConfigurationProperties必须告诉他加载的数据前缀是什么，这样指定前缀下的所有属性就封装到这个对象中。记得数据属性名要与对象的变量名一一对应啊，不然没法封装。其实以后如果你要定义一组数据自己使用，就可以先写一个对象，然后定义好属性，下面到配置中根据这个格式书写即可。

​	<img src="img\boot\image-20211126181423432.png" alt="image-20211126181423432" style="zoom:80%;" />

<font color="#f0f"><b>温馨提示</b></font>

​		细心的小伙伴会发现一个问题，自定义的这种数据在yaml文件中书写时没有弹出提示，咱们到原理篇再揭秘如何弹出提示。

**总结**

1. 使用@ConfigurationProperties注解绑定配置信息到封装类中
2. 封装类需要定义为Spring管理的bean，否则无法进行属性注入



#### yaml文件中的数据引用

​		如果你在书写yaml数据时，经常出现如下现象，比如很多个文件都具有相同的目录前缀

```YAML
center:
	dataDir: /usr/local/fire/data
    tmpDir: /usr/local/fire/tmp
    logDir: /usr/local/fire/log
    msgDir: /usr/local/fire/msgDir
```

​		或者

```YAML
center:
	dataDir: D:/usr/local/fire/data
    tmpDir: D:/usr/local/fire/tmp
    logDir: D:/usr/local/fire/log
    msgDir: D:/usr/local/fire/msgDir
```

​		这个时候你可以使用引用格式来定义数据，其实就是搞了个变量名，然后引用变量了，格式如下：

```YAML
baseDir: /usr/local/fire
center:
    dataDir: ${baseDir}/data
    tmpDir: ${baseDir}/tmp
    logDir: ${baseDir}/log
    msgDir: ${baseDir}/msgDir
```

​	还有一个注意事项，在书写字符串时，如果需要使用转义字符，需要将数据字符串使用双引号包裹起来

```YAML
lesson: "Spring\tboot\nlesson"
```

**总结**

1. 在配置文件中可以使用${属性名}方式引用属性值
2. 如果属性中出现特殊字符，可以使用双引号包裹起来作为字符解析

​		到这里有关yaml文件的基础使用就先告一段落，实用篇中再继续研究更深入的内容。

```yaml

```

## 配置信息的加载顺序

- 先加载 yml
- 再加载 yaml
- 再加载 properties
- <span style="color:red">后加载的覆盖先加载的哦~</span>

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

### 自动配置原理

#### 引导加载自动配置类

- @SpringBootConfiguration 表示是一个配置类
- @ComponentScan 指定扫描哪些，Spring注解；
- @EnableAutoConfiguration，开启自动配置 

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

![image-20211003173551017](img\boot\AutoConfigurationImportSelector.png)

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

```xml
<dependency>
    <groupId>org.projectlombok</groupId>
    <artifactId>lombok</artifactId>
</dependency>
```

简单使用

```java
//idea中搜索安装lombok插件
//===============================简化JavaBean开发===================================
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
//================================简化日志开发===================================
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

### 配置问题提示

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-configuration-processor</artifactId>
    <optional>true</optional>
</dependency>
```

# Web 开发基础

![yuque_diagram](img\boot\yuque_diagram.jpg)



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

![image-20211023111242760](img\boot\image-20211023111242760.png)

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

![image-20211023114049369](img\boot\image-20211023114049369.png)

![image-20211023114945974](img\boot\image-20211023114945974.png)

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

![image-20211024200517771](img\boot\image-20211024200517771-16350771187701.png)

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

![image-20211024201053219](img\boot\image-20211024201053219.png)

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

![image-20211024201539502](img\boot\image-20211024201539502.png)

### 确定目标方法每一个参数值

#### 解析器

#### 解析参数

#### 自定义类型参数

### 目标方法执行完成

### 处理派发结果

# 基本框架整合

https://github.com/spring-projects/spring-boot/tree/master/spring-boot-samples

## 整合JDBC

SpringBoot整合JDBC只要引入对应数据库的连接驱动包（如mysql的驱动），然后再properties/yaml配置文件中书写配置即可。

> 依赖

- springboot的jdbc api
- mysql驱动

> properties文件

```properties
# jdbc基础配置
spring.datasource.username=root
spring.datasource.password=root
spring.datasource.url=jdbc:mysql://localhost:3306/blog
spring.datasource.driver-class-name=com.mysql.jdbc.Driver
```

> pom文件

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

## 整合 Druid

> 概述

SpringBoot整合Druid，需要导入数据库的连接驱动包（如mysql的驱动），Druid包，然后书写对应的配置文件。注意：由于数据库连接相关的配置文件是在Druid中进行设置的，所以前缀名要一致。

> 依赖

- mysql驱动
- springboot jdbc api
- druid依赖
- springboot-web依赖，用来注册servlet，filter 启用druid的控制台
- log4j，我们使用这个日志框架进行记录

> pom文件

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

> properties文件

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

> 代码

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

## 整合 MyBatis

> 概述

数据库采用MySQL，连接池采用Druid。整合方式有SQL用纯注解和SQL采用xml两个版本。

> 依赖

- mysql驱动
- springboot jdbc api
- druid 依赖
- springboot-web 依赖，用来注册 servlet，filter 启用druid的控制台
- log4j，我们使用这个日志框架进行记录
- mybatis 和 Spring 的整合包
- 其他的会自动帮我们导入的，不必担心

> pom文件

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

> properties件

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

> SQL纯注解

每个dao接口都加上注解 mapper 或 在启动引导类位置处用@MapperScan("扫描得包全名")

> SQL书写

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

> SQL xml版本

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

## 整合 JPA

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

## 整合MVC

> 概述

SpringBoot 有默认的配置，它替我们设置了有关 MVC 的一些默认配置。我们可以不使用这些默认配置，全面接管相关配置（全部由我们自行定义），也可以只修改必要的部分，其他的仍采用 Spring Boot 为我们提供的默认配置。一般是不采用全面接管。

> 依赖

springboot-web模块

> pom文件

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

> 配置代码

拦截器

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

注册组件

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

测试代码

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

## 整合 MVC&MyBatis

> 概述

SpringBoot的ssm整合配置

> 依赖

- mysql驱动
- springboot jdbc api
- druid依赖
- springboot-web依赖，用来注册servlet，filter 启用druid的控制台
- log4j，我们使用这个日志框架进行记录
- mybatis和spring的整合包
- 其他的会自动帮我们导入的，不必担心

> pom文件

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

> properties文件

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

> Config代码

druid

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

web

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

mybatis的sql文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.bbxx.boot02.mapper.ArticleMapper">

</mapper>
```

## 整合 Redis

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

## 集成 JUnit

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

# 基础练习

基于SpringBoot实现SSMP整合，学习如下技术的整合方式

- 整合JUnit
- 整合MyBatis
- 整合MyBatis-Plus
- 整合Druid


## 整合JUnit

先来看一下不使用 SpringBoot 技术时，Spring整合 JUnit 的制作方式

```JAVA
//加载spring整合junit专用的类运行器
@RunWith(SpringJUnit4ClassRunner.class)
//指定对应的配置信息
@ContextConfiguration(classes = SpringConfig.class)
public class AccountServiceTestCase {
    //注入你要测试的对象
    @Autowired
    private AccountService accountService;
    @Test
    public void testGetById(){
        //执行要测试的对象对应的方法
        System.out.println(accountService.findById(2));
    }
}

```

其中核心代码是前两个注解，第一个注解`@RunWith`是设置Spring专用的测试类运行器，简单说就是Spring程序执行程序有自己的一套独立的运行程序的方式，不能使用JUnit提供的类运行方式了，必须指定一下，但是格式是固定的，琢磨一下，<font color="#ff0000"><b>每次都指定一样的东西，这个东西写起来没有技术含量啊</b></font>，第二个注解@ContextConfiguration是用来设置Spring核心配置文件或配置类的，简单说就是加载Spring的环境你要告诉Spring具体的环境配置是在哪里写的，虽然每次加载的文件都有可能不同，但是仔细想想，如果文件名是固定的，这个貌似也是一个固定格式。既然<font color="#ff0000"><b>有可能是固定格式，那就有可能每次都写一样的东西，也是一个没有技术含量的内容书写</b></font>

SpringBoot就抓住上述两条没有技术含量的内容书写进行开发简化，能走默认值的走默认值，能不写的就不写，具体格式如下

```JAVA
@SpringBootTest
class Springboot04JunitApplicationTests {
    //注入你要测试的对象
    @Autowired
    private BookDao bookDao;
    @Test
    void contextLoads() {
        //执行要测试的对象对应的方法
        bookDao.save();
        System.out.println("two...");
    }
}
```

看看这次简化成什么样了，一个注解就搞定了，而且还没有参数，再体会SpringBoot整合其他技术的优势在哪里，就两个字——<font color="#ff0000"><b>简化</b></font>。使用一个注解@SpringBootTest替换了前面两个注解。至于内部是怎么回事？和之前一样，只不过都走默认值。

这个时候有人就问了，你加载的配置类或者配置文件是哪一个？就是我们前面启动程序使用的引导类。如果想手工指定引导类有两种方式，第一种方式使用属性的形式进行，在注解@SpringBootTest中添加classes属性指定配置类

```JAVA
@SpringBootTest(classes = Springboot04JunitApplication.class)
class Springboot04JunitApplicationTests {
    //注入你要测试的对象
    @Autowired
    private BookDao bookDao;
    @Test
    void contextLoads() {
        //执行要测试的对象对应的方法
        bookDao.save();
        System.out.println("two...");
    }
}
```

第二种方式回归原始配置方式，仍然使用@ContextConfiguration注解进行，效果是一样的

```JAVA
@SpringBootTest
@ContextConfiguration(classes = Springboot04JunitApplication.class)
class Springboot04JunitApplicationTests {
    //注入你要测试的对象
    @Autowired
    private BookDao bookDao;
    @Test
    void contextLoads() {
        //执行要测试的对象对应的方法
        bookDao.save();
        System.out.println("two...");
    }
}
```

<font color="#f0f"><b>温馨提示</b></font>

使用SpringBoot整合JUnit需要保障导入test对应的starter，由于初始化项目时此项是默认导入的，所以此处没有提及，其实和之前学习的内容一样，用什么技术导入对应的starter即可。

**总结**

1. 导入测试对应的starter
2. 测试类使用@SpringBootTest修饰
3. 使用自动装配的形式添加要测试的对象
4. 测试类如果存在于引导类所在包或子包中无需指定引导类
5. 测试类如果不存在于引导类所在的包或子包中需要通过classes属性指定引导类

## 整合MyBatis

整合完JUnit下面再来说一下整合MyBatis，这个技术是大部分公司都要使用的技术，务必掌握。如果对Spring整合MyBatis不熟悉的小伙伴好好复习一下，下面列举出原始整合的全部内容，以配置类的形式为例进行

导入坐标，MyBatis坐标不能少，Spring整合MyBatis还有自己专用的坐标，此外Spring进行数据库操作的jdbc坐标是必须的，剩下还有mysql驱动坐标，本例中使用了Druid数据源，这个倒是可以不要

```XML
<dependencies>
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid</artifactId>
        <version>1.1.16</version>
    </dependency>
    <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis</artifactId>
        <version>3.5.6</version>
    </dependency>
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <version>5.1.47</version>
    </dependency>
    <!--1.导入mybatis与spring整合的jar包-->
    <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis-spring</artifactId>
        <version>1.3.0</version>
    </dependency>
    <!--导入spring操作数据库必选的包-->
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-jdbc</artifactId>
        <version>5.2.10.RELEASE</version>
    </dependency>
</dependencies>
```

Spring核心配置

```JAVA
@Configuration
@ComponentScan("com.itheima")
@PropertySource("jdbc.properties")
public class SpringConfig {
}
```

MyBatis要交给Spring接管的bean

```JAVA
//定义mybatis专用的配置类
@Configuration
public class MyBatisConfig {
//    定义创建SqlSessionFactory对应的bean
    @Bean
    public SqlSessionFactoryBean sqlSessionFactory(DataSource dataSource){
        //SqlSessionFactoryBean是由mybatis-spring包提供的，专用于整合用的对象
        SqlSessionFactoryBean sfb = new SqlSessionFactoryBean();
        //设置数据源替代原始配置中的environments的配置
        sfb.setDataSource(dataSource);
        //设置类型别名替代原始配置中的typeAliases的配置
        sfb.setTypeAliasesPackage("com.itheima.domain");
        return sfb;
    }
//    定义加载所有的映射配置
    @Bean
    public MapperScannerConfigurer mapperScannerConfigurer(){
        MapperScannerConfigurer msc = new MapperScannerConfigurer();
        msc.setBasePackage("com.itheima.dao");
        return msc;
    }

}
```

数据源对应的bean，此处使用Druid数据源

```JAVA
@Configuration
public class JdbcConfig {
    @Value("${jdbc.driver}")
    private String driver;
    @Value("${jdbc.url}")
    private String url;
    @Value("${jdbc.username}")
    private String userName;
    @Value("${jdbc.password}")
    private String password;

    @Bean("dataSource")
    public DataSource dataSource(){
        DruidDataSource ds = new DruidDataSource();
        ds.setDriverClassName(driver);
        ds.setUrl(url);
        ds.setUsername(userName);
        ds.setPassword(password);
        return ds;
    }
}
```

数据库连接信息（properties格式）

```properties
jdbc.driver=com.mysql.jdbc.Driver
jdbc.url=jdbc:mysql://localhost:3306/spring_db?useSSL=false
jdbc.username=root
jdbc.password=root
```

上述格式基本上是最简格式了，要写的东西还真不少。下面看看SpringBoot整合MyBaits格式

**步骤①**：创建模块

![image-20211129092156020](img\boot\image-20211129092156020.png)

**步骤②**：勾选要使用的技术，MyBatis，由于要操作数据库，还要勾选对应数据库

![image-20211129092210993](img\boot\image-20211129092210993.png)

或者手工导入对应技术的starter，和对应数据库的坐标

```XML
<dependencies>
    <!--1.导入对应的starter-->
    <dependency>
        <groupId>org.mybatis.spring.boot</groupId>
        <artifactId>mybatis-spring-boot-starter</artifactId>
        <version>2.2.0</version>
    </dependency>

    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

**步骤③**：配置数据源相关信息，没有这个信息你连接哪个数据库都不知道

```yaml
#2.配置相关信息
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db
    username: root
    password: root
```

​		结束了，就这么多，没了。有人就很纳闷，这就结束了？对，这就结束了，SpringBoot把配置中所有可能出现的通用配置都简化了。下面写一个MyBatis程序运行需要的Dao（或者Mapper）就可以运行了

**实体类**

```JAVA
public class Book {
    private Integer id;
    private String type;
    private String name;
    private String description;
}
```

**映射接口（Dao）**

```JAVA
@Mapper
public interface BookDao {
    @Select("select * from tbl_book where id = #{id}")
    public Book getById(Integer id);
}
```

**测试类**

```JAVA
@SpringBootTest
class Springboot05MybatisApplicationTests {
    @Autowired
    private BookDao bookDao;
    @Test
    void contextLoads() {
        System.out.println(bookDao.getById(1));
    }
}
```

​		完美，开发从此变的就这么简单。再体会一下SpringBoot如何进行第三方技术整合的，是不是很优秀？具体内部的原理到原理篇再展开讲解

​		<font color="#ff0000"><b>注意</b></font>：当前使用的SpringBoot版本是2.5.4，对应的坐标设置中Mysql驱动使用的是8x版本。使用SpringBoot2.4.3（不含）之前版本会出现一个小BUG，就是MySQL驱动升级到8以后要求强制配置时区，如果不设置会出问题。解决方案很简单，驱动url上面添加上对应设置就行了

```YAML
#2.配置相关信息
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
    username: root
    password: root
```

​		这里设置的UTC是全球标准时间，你也可以理解为是英国时间，中国处在东八区，需要在这个基础上加上8小时，这样才能和中国地区的时间对应的，也可以修改配置为Asia/Shanghai，同样可以解决这个问题。

```YAML
#2.配置相关信息
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=Asia/Shanghai
    username: root
    password: root
```

​		如果不想每次都设置这个东西，也可以去修改mysql中的配置文件mysql.ini，在mysqld项中添加default-time-zone=+8:00也可以解决这个问题。其实方式方法很多，这里就说这么多吧。

​		此外在运行程序时还会给出一个提示，说数据库驱动过时的警告，根据提示修改配置即可，弃用**com.mysql.jdbc.Driver**，换用<font color="#ff0000"><b>com.mysql.cj.jdbc.Driver</b></font>。前面的例子中已经更换了驱动了，在此说明一下。

```tex
Loading class `com.mysql.jdbc.Driver'. This is deprecated. The new driver class is `com.mysql.cj.jdbc.Driver'. The driver is automatically registered via the SPI and manual loading of the driver class is generally unnecessary.
```

**总结**

1. 整合操作需要勾选MyBatis技术，也就是导入MyBatis对应的starter

2. 数据库连接相关信息转换成配置

3. 数据库SQL映射需要添加@Mapper被容器识别到

4. MySQL 8.X驱动强制要求设置时区

    - 修改url，添加serverTimezone设定
    - 修改MySQL数据库配置

5. 驱动类过时，提醒更换为com.mysql.cj.jdbc.Driver


## 整合MyBatis-Plus

做完了两种技术的整合了，各位小伙伴要学会总结，我们做这个整合究竟哪些是核心？总结下来就两句话

- 导入对应技术的starter坐标

- 根据对应技术的要求做配置

    虽然看起来有点虚，但是确实是这个理儿，下面趁热打铁，再换一个技术，看看是不是上面这两步。

​		接下来在MyBatis的基础上再升级一下，整合MyBaitsPlus（简称MP），国人开发的技术，符合中国人开发习惯，谁用谁知道。来吧，一起做整合

**步骤①**：导入对应的starter

```XML
<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus-boot-starter</artifactId>
    <version>3.4.3</version>
</dependency>
```

​		关于这个坐标，此处要说明一点，之前我们看的starter都是spring-boot-starter-？？？，也就是说都是下面的格式

```tex
Spring-boot-start-***
```

​	而MyBatis与MyBatisPlus这两个坐标的名字书写比较特殊，是第三方技术名称在前，boot和starter在后。此处简单提一下命名规范，后期原理篇会再详细讲解

| starter所属 | 命名规则                                                    | 示例                                                      |
| ----------- | ----------------------------------------------------------- | --------------------------------------------------------- |
| 官方提供    | spring-boot-starter-技术名称                                | spring-boot-starter-web <br/>spring-boot-starter-test     |
| 第三方提供  | 第三方技术名称-spring-boot-starter                          | mybatis-spring-boot-starter<br/>druid-spring-boot-starter |
| 第三方提供  | 第三方技术名称-boot-starter（第三方技术名称过长，简化命名） | mybatis-plus-boot-starter                                 |

<font color="#f0f"><b>温馨提示</b></font>

​		有些小伙伴在创建项目时想通过勾选的形式找到这个名字，别翻了，没有。截止目前，SpringBoot官网还未收录此坐标，而我们Idea创建模块时读取的是SpringBoot官网的Spring Initializr，所以也没有。如果换用阿里云的url创建项目可以找到对应的坐标。

**步骤②**：配置数据源相关信息

```yaml
#2.配置相关信息
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db
    username: root
    password: root
```

​	没了，就这么多，剩下的就是写MyBaitsPlus的程序了

**映射接口（Dao）**

```JAVA
@Mapper
public interface BookDao extends BaseMapper<Book> {
}
```

​	核心在于Dao接口继承了一个BaseMapper的接口，这个接口中帮助开发者预定了若干个常用的API接口，简化了通用API接口的开发工作。

<img src="img\boot\image-20211129100313919.png" alt="image-20211129100313919" style="zoom:80%;" />

​	下面就可以写一个测试类进行测试了，此处省略。

<font color="#f0f"><b>温馨提示</b></font>

​		目前数据库的表名定义规则是tbl_模块名称，为了能和实体类相对应，需要做一个配置，相关知识各位小伙伴可以到MyBatisPlus课程中去学习，此处仅给出解决方案。配置application.yml文件，添加如下配置即可，设置所有表名的通用前缀名

```yaml
mybatis-plus:
  global-config:
    db-config:
      table-prefix: tbl_		#设置所有表的通用前缀名称为tbl_
```

**总结**

1. 手工添加MyBatis-Plus对应的starter
2. 数据层接口使用BaseMapper简化开发
3. 需要使用的第三方技术无法通过勾选确定时，需要手工添加坐标

## 整合Druid

使用SpringBoot整合了3个技术了，发现套路基本相同，导入对应的starter，然后做配置，各位小伙伴需要一直强化这套思想。下面再整合一个技术，继续深入强化此思想。

前面整合MyBatis和MyBatisPlus的时候，使用的数据源对象都是SpringBoot默认的数据源对象，下面我们手工控制一下，自己指定了一个数据源对象，Druid。

在没有指定数据源时，我们的配置如下：

```YAML
#2.配置相关信息
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=Asia/Shanghai
    username: root
    password: root
```

此时虽然没有指定数据源，但是根据SpringBoot的德行，肯定帮我们选了一个它认为最好的数据源对象，这就是HiKari。通过启动日志可以查看到对应的身影。

```tex
2021-11-29 09:39:15.202  INFO 12260 --- [           main] com.zaxxer.hikari.HikariDataSource       : HikariPool-1 - Starting...
2021-11-29 09:39:15.208  WARN 12260 --- [           main] com.zaxxer.hikari.util.DriverDataSource  : Registered driver with driverClassName=com.mysql.jdbc.Driver was not found, trying direct instantiation.
2021-11-29 09:39:15.551  INFO 12260 --- [           main] com.zaxxer.hikari.HikariDataSource       : HikariPool-1 - Start completed.
```

上述信息中每一行都有HiKari的身影，如果需要更换数据源，其实只需要两步即可。

1. 导入对应的技术坐标

2. 配置使用指定的数据源类型

    下面就切换一下数据源对象

**步骤①**：导入对应的坐标（注意，是坐标，此处不是starter）

```XML
<dependencies>
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid</artifactId>
        <version>1.1.16</version>
    </dependency>
</dependencies>
```

**步骤②**：修改配置，在数据源配置中有一个type属性，专用于指定数据源类型

```YAML
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
    username: root
    password: root
    type: com.alibaba.druid.pool.DruidDataSource
```

这里其实要提出一个问题的，目前的数据源配置格式是一个通用格式，不管你换什么数据源都可以用这种形式进行配置。但是新的问题又来了，如果对数据源进行个性化的配置，例如配置数据源对应的连接数量，这个时候就有新的问题了。每个数据源技术对应的配置名称都一样吗？肯定不是啊，各个厂商不可能提前商量好都写一样的名字啊，怎么办？就要使用专用的配置格式了。这个时候上面这种通用格式就不能使用了，怎么办？还能怎么办？按照SpringBoot整合其他技术的通用规则来套啊，导入对应的starter，进行相应的配置即可。

**步骤①**：导入对应的starter

```XML
<dependencies>
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid-spring-boot-starter</artifactId>
        <version>1.2.6</version>
    </dependency>
</dependencies>
```

**步骤②**：修改配置

```YAML
spring:
  datasource:
    druid:
      driver-class-name: com.mysql.cj.jdbc.Driver
      url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
      username: root
      password: root
```

注意观察，配置项中，在datasource下面并不是直接配置url这些属性的，而是先配置了一个druid节点，然后再配置的url这些东西。言外之意，url这个属性是druid下面的属性，那你能想到什么？除了这4个常规配置外，还有druid专用的其他配置。通过提示功能可以打开druid相关的配置查阅

<img src="img\boot\image-20211129112610729.png" alt="image-20211129112610729" style="zoom:80%;" />

与druid相关的配置超过200条以上，这就告诉你，如果想做druid相关的配置，使用这种格式就可以了，这里就不展开描述了，太多了。

这是我们做的第4个技术的整合方案，还是那两句话：<font color="#ff0000"><b>导入对应starter，使用对应配置</b></font>。没了，SpringBoot整合其他技术就这么简单粗暴。

**总结**

1. 整合Druid需要导入Druid对应的starter
2. 根据Druid提供的配置方式进行配置
3. 整合第三方技术通用方式
    - 导入对应的starter
    - 根据提供的配置格式，配置非默认值对应的配置项

## SSMP整合综合案例

将所有知识贯穿起来，同时做一个小功能，体会一下。案例的最终效果

**主页面**

![image-20211129113447844](img\boot\image-20211129113447844.png)

**添加**

![image-20211129113522459](img\boot\image-20211129113522459.png)

**删除**

![image-20211129113550829](img\boot\image-20211129113550829.png)

**修改**

**分页**

![image-20211129113628969](img\boot\image-20211129113628969.png)

**条件查询**

![image-20211129113650369](img\boot\image-20211129113650369.png)

​	整体案例中需要采用的技术如下，先了解一下，做到哪一个说哪一个

1. 实体类开发————使用Lombok快速制作实体类
2. Dao开发————整合MyBatisPlus，制作数据层测试
3. Service开发————基于MyBatisPlus进行增量开发，制作业务层测试类
4. Controller开发————基于Restful开发，使用PostMan测试接口功能
5. Controller开发————前后端开发协议制作
6. 页面开发————基于VUE+ElementUI制作，前后端联调，页面数据处理，页面消息处理
    - 列表
    - 新增
    - 修改
    - 删除
    - 分页
    - 查询
7. 项目异常处理
8. 按条件查询————页面功能调整、Controller修正功能、Service修正功能

​		可以看的出来，东西还是很多的，希望通过这个案例，各位小伙伴能够完成基础开发的技能训练。整体开发过程采用做一层测一层的形式进行，过程完整，战线较长，希望各位能跟紧进度，完成这个小案例的制作。

#### 0.模块创建

​		对于这个案例如果按照企业开发的形式进行应该制作后台微服务，前后端分离的开发。

<img src="img\boot\image-20211129114306062.png" alt="image-20211129114306062" style="zoom:80%;" />

​		我知道这个对初学的小伙伴要求太高了，咱们简化一下。后台做单体服务器，前端不使用前后端分离的制作了。

<img src="img\boot\image-20211129114328967.png" alt="image-20211129114328967" style="zoom:80%;" />

​		一个服务器即充当后台服务调用，又负责前端页面展示，降低学习的门槛。

​		下面我们创建一个新的模块，加载要使用的技术对应的starter，修改配置文件格式为yml格式，并把web访问端口先设置成80。

**pom.xml**

```XML
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>
```

**application.yml**

```yaml
server:
  port: 80
```

#### 1.实体类开发

​		本案例对应的模块表结构如下：

```tex
-- ----------------------------
-- Table structure for tbl_book
-- ----------------------------
DROP TABLE IF EXISTS `tbl_book`;
CREATE TABLE `tbl_book`  (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `type` varchar(20) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `name` varchar(50) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  `description` varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 51 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of tbl_book
-- ----------------------------
INSERT INTO `tbl_book` VALUES (1, '计算机理论', 'Spring实战 第5版', 'Spring入门经典教程，深入理解Spring原理技术内幕');
INSERT INTO `tbl_book` VALUES (2, '计算机理论', 'Spring 5核心原理与30个类手写实战', '十年沉淀之作，手写Spring精华思想');
INSERT INTO `tbl_book` VALUES (3, '计算机理论', 'Spring 5 设计模式', '深入Spring源码剖析Spring源码中蕴含的10大设计模式');
INSERT INTO `tbl_book` VALUES (4, '计算机理论', 'Spring MVC+MyBatis开发从入门到项目实战', '全方位解析面向Web应用的轻量级框架，带你成为Spring MVC开发高手');
INSERT INTO `tbl_book` VALUES (5, '计算机理论', '轻量级Java Web企业应用实战', '源码级剖析Spring框架，适合已掌握Java基础的读者');
INSERT INTO `tbl_book` VALUES (6, '计算机理论', 'Java核心技术 卷I 基础知识（原书第11版）', 'Core Java 第11版，Jolt大奖获奖作品，针对Java SE9、10、11全面更新');
INSERT INTO `tbl_book` VALUES (7, '计算机理论', '深入理解Java虚拟机', '5个维度全面剖析JVM，大厂面试知识点全覆盖');
INSERT INTO `tbl_book` VALUES (8, '计算机理论', 'Java编程思想（第4版）', 'Java学习必读经典,殿堂级著作！赢得了全球程序员的广泛赞誉');
INSERT INTO `tbl_book` VALUES (9, '计算机理论', '零基础学Java（全彩版）', '零基础自学编程的入门图书，由浅入深，详解Java语言的编程思想和核心技术');
INSERT INTO `tbl_book` VALUES (10, '市场营销', '直播就该这么做：主播高效沟通实战指南', '李子柒、李佳琦、薇娅成长为网红的秘密都在书中');
INSERT INTO `tbl_book` VALUES (11, '市场营销', '直播销讲实战一本通', '和秋叶一起学系列网络营销书籍');
INSERT INTO `tbl_book` VALUES (12, '市场营销', '直播带货：淘宝、天猫直播从新手到高手', '一本教你如何玩转直播的书，10堂课轻松实现带货月入3W+');
```

​		根据上述表结构，制作对应的实体类

**实体类**

```JAVA
public class Book {
    private Integer id;
    private String type;
    private String name;
    private String description;
}
```

​		实体类的开发可以自动通过工具手工生成get/set方法，然后覆盖toString()方法，方便调试，等等。不过这一套操作书写很繁琐，有对应的工具可以帮助我们简化开发，介绍一个小工具，lombok。

​		Lombok，一个Java类库，提供了一组注解，简化POJO实体类开发，SpringBoot目前默认集成了lombok技术，并提供了对应的版本控制，所以只需要提供对应的坐标即可，在pom.xml中添加lombok的坐标。

```XML
<dependencies>
    <!--lombok-->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
    </dependency>
</dependencies>
```

​		使用lombok可以通过一个注解@Data完成一个实体类对应的getter，setter，toString，equals，hashCode等操作的快速添加

```JAVA
import lombok.Data;
@Data
public class Book {
    private Integer id;
    private String type;
    private String name;
    private String description;
}
```

​		到这里实体类就做好了，是不是比不使用lombok简化好多，这种工具在Java开发中还有N多，后面遇到了能用的实用开发技术时，在不增加各位小伙伴大量的学习时间的情况下，尽量多给大家介绍一些。

**总结**

1. 实体类制作
2. 使用lombok简化开发
    - 导入lombok无需指定版本，由SpringBoot提供版本
    - @Data注解



#### 2.数据层开发——基础CRUD

​		数据层开发本次使用MyBatisPlus技术，数据源使用前面学习的Druid，学都学了都用上。

**步骤①**：导入MyBatisPlus与Druid对应的starter，当然mysql的驱动不能少

```xml
<dependencies>
    <dependency>
        <groupId>com.baomidou</groupId>
        <artifactId>mybatis-plus-boot-starter</artifactId>
        <version>3.4.3</version>
    </dependency>
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid-spring-boot-starter</artifactId>
        <version>1.2.6</version>
    </dependency>
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

**步骤②**：配置数据库连接相关的数据源配置

```YAML
server:
  port: 80

spring:
  datasource:
    druid:
      driver-class-name: com.mysql.cj.jdbc.Driver
      url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
      username: root
      password: root
```

**步骤③**：使用MyBatisPlus的标准通用接口BaseMapper加速开发，别忘了@Mapper和泛型的指定

```JAVA
@Mapper
public interface BookDao extends BaseMapper<Book> {
}
```

**步骤④**：制作测试类测试结果，这个测试类制作是个好习惯，不过在企业开发中往往都为加速开发跳过此步，且行且珍惜吧

```JAVA
package com.itheima.dao;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.itheima.domain.Book;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
public class BookDaoTestCase {

    @Autowired
    private BookDao bookDao;

    @Test
    void testGetById(){
        System.out.println(bookDao.selectById(1));
    }

    @Test
    void testSave(){
        Book book = new Book();
        book.setType("测试数据123");
        book.setName("测试数据123");
        book.setDescription("测试数据123");
        bookDao.insert(book);
    }

    @Test
    void testUpdate(){
        Book book = new Book();
        book.setId(17);
        book.setType("测试数据abcdefg");
        book.setName("测试数据123");
        book.setDescription("测试数据123");
        bookDao.updateById(book);
    }

    @Test
    void testDelete(){
        bookDao.deleteById(16);
    }

    @Test
    void testGetAll(){
        bookDao.selectList(null);
    }
}
```

<font color="#f0f"><b>温馨提示</b></font>

​		MyBatisPlus技术默认的主键生成策略为雪花算法，生成的主键ID长度较大，和目前的数据库设定规则不相符，需要配置一下使MyBatisPlus使用数据库的主键生成策略，方式嘛还是老一套，做配置。在application.yml中添加对应配置即可，具体如下

```yaml
server:
  port: 80

spring:
  datasource:
    druid:
      driver-class-name: com.mysql.cj.jdbc.Driver
      url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
      username: root
      password: root

mybatis-plus:
  global-config:
    db-config:
      table-prefix: tbl_		#设置表名通用前缀
      id-type: auto				#设置主键id字段的生成策略为参照数据库设定的策略，当前数据库设置id生成策略为自增
```

##### 查看MyBatisPlus运行日志

​		在进行数据层测试的时候，因为基础的CRUD操作均由MyBatisPlus给我们提供了，所以就出现了一个局面，开发者不需要书写SQL语句了，这样程序运行的时候总有一种感觉，一切的一切都是黑盒的，作为开发者我们啥也不知道就完了。如果程序正常运行还好，如果报错了，这个时候就很崩溃，你甚至都不知道从何下手，因为传递参数、封装SQL语句这些操作完全不是你开发出来的，所以查看执行期运行的SQL语句就成为当务之急。

​		SpringBoot整合MyBatisPlus的时候充分考虑到了这点，通过配置的形式就可以查阅执行期SQL语句，配置如下

```YAML
mybatis-plus:
  global-config:
    db-config:
      table-prefix: tbl_
      id-type: auto
  configuration:
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
```

​		再来看运行结果，此时就显示了运行期执行SQL的情况。

```tex
Creating a new SqlSession
SqlSession [org.apache.ibatis.session.defaults.DefaultSqlSession@2c9a6717] was not registered for synchronization because synchronization is not active
JDBC Connection [com.mysql.cj.jdbc.ConnectionImpl@6ca30b8a] will not be managed by Spring
==>  Preparing: SELECT id,type,name,description FROM tbl_book
==> Parameters: 
<==    Columns: id, type, name, description
<==        Row: 1, 计算机理论, Spring实战 第5版, Spring入门经典教程，深入理解Spring原理技术内幕
<==        Row: 2, 计算机理论, Spring 5核心原理与30个类手写实战, 十年沉淀之作，手写Spring精华思想
<==        Row: 3, 计算机理论, Spring 5 设计模式, 深入Spring源码剖析Spring源码中蕴含的10大设计模式
<==        Row: 4, 计算机理论, Spring MVC+MyBatis开发从入门到项目实战, 全方位解析面向Web应用的轻量级框架，带你成为Spring MVC开发高手
<==        Row: 5, 计算机理论, 轻量级Java Web企业应用实战, 源码级剖析Spring框架，适合已掌握Java基础的读者
<==        Row: 6, 计算机理论, Java核心技术 卷I 基础知识（原书第11版）, Core Java 第11版，Jolt大奖获奖作品，针对Java SE9、10、11全面更新
<==        Row: 7, 计算机理论, 深入理解Java虚拟机, 5个维度全面剖析JVM，大厂面试知识点全覆盖
<==        Row: 8, 计算机理论, Java编程思想（第4版）, Java学习必读经典,殿堂级著作！赢得了全球程序员的广泛赞誉
<==        Row: 9, 计算机理论, 零基础学Java（全彩版）, 零基础自学编程的入门图书，由浅入深，详解Java语言的编程思想和核心技术
<==        Row: 10, 市场营销, 直播就该这么做：主播高效沟通实战指南, 李子柒、李佳琦、薇娅成长为网红的秘密都在书中
<==        Row: 11, 市场营销, 直播销讲实战一本通, 和秋叶一起学系列网络营销书籍
<==        Row: 12, 市场营销, 直播带货：淘宝、天猫直播从新手到高手, 一本教你如何玩转直播的书，10堂课轻松实现带货月入3W+
<==        Row: 13, 测试类型, 测试数据, 测试描述数据
<==        Row: 14, 测试数据update, 测试数据update, 测试数据update
<==        Row: 15, -----------------, 测试数据123, 测试数据123
<==      Total: 15
```

​		其中清晰的标注了当前执行的SQL语句是什么，携带了什么参数，对应的执行结果是什么，所有信息应有尽有。

​		此处设置的是日志的显示形式，当前配置的是控制台输出，当然还可以由更多的选择，根据需求切换即可

<img src="img\boot\image-20211129143207295.png" alt="image-20211129143207295" style="zoom:80%;" />



**总结**

1. 手工导入starter坐标（2个），mysql驱动（1个）

2. 配置数据源与MyBatisPlus对应的配置

3. 开发Dao接口（继承BaseMapper）

4. 制作测试类测试Dao功能是否有效

5. 使用配置方式开启日志，设置日志输出方式为标准输出即可查阅SQL执行日志

    

#### 3.数据层开发——分页功能制作

​		前面仅仅是使用了MyBatisPlus提供的基础CRUD功能，实际上MyBatisPlus给我们提供了几乎所有的基础操作，这一节说一下如何实现数据库端的分页操作。

​		MyBatisPlus提供的分页操作API如下：

```JAVA
@Test
void testGetPage(){
    IPage page = new Page(2,5);
    bookDao.selectPage(page, null);
    System.out.println(page.getCurrent());
    System.out.println(page.getSize());
    System.out.println(page.getTotal());
    System.out.println(page.getPages());
    System.out.println(page.getRecords());
}
```

​		其中selectPage方法需要传入一个封装分页数据的对象，可以通过new的形式创建这个对象，当然这个对象也是MyBatisPlus提供的，别选错包了。创建此对象时需要指定两个分页的基本数据

- 当前显示第几页
- 每页显示几条数据


​		可以通过创建Page对象时利用构造方法初始化这两个数据。

```JAVA
IPage page = new Page(2,5);
```

​		将该对象传入到查询方法selectPage后，可以得到查询结果，但是我们会发现当前操作查询结果返回值仍然是一个IPage对象，这又是怎么回事？

```JAVA
IPage page = bookDao.selectPage(page, null);
```

​		原来这个IPage对象中封装了若干个数据，而查询的结果作为IPage对象封装的一个数据存在的，可以理解为查询结果得到后，又塞到了这个IPage对象中，其实还是为了高度的封装，一个IPage描述了分页所有的信息。下面5个操作就是IPage对象中封装的所有信息了。

```JAVA
@Test
void testGetPage(){
    IPage page = new Page(2,5);
    bookDao.selectPage(page, null);
    System.out.println(page.getCurrent());		//当前页码值
    System.out.println(page.getSize());			//每页显示数
    System.out.println(page.getTotal());		//数据总量
    System.out.println(page.getPages());		//总页数
    System.out.println(page.getRecords());		//详细数据
}
```

​		到这里就知道这些数据如何获取了，但是当你去执行这个操作时，你会发现并不像我们分析的这样，实际上这个分页功能当前是无效的。为什么这样呢？这个要源于MyBatisPlus的内部机制。

​		对于MySQL的分页操作使用limit关键字进行，而并不是所有的数据库都使用limit关键字实现的，这个时候MyBatisPlus为了制作的兼容性强，将分页操作设置为基础查询操作的升级版，你可以理解为IPhone6与IPhone6S-PLUS的关系。

​		基础操作中有查询全部的功能，而在这个基础上只需要升级一下（PLUS）就可以得到分页操作。所以MyBatisPlus将分页操作做成了一个开关，你用分页功能就把开关开启，不用就不需要开启这个开关。而我们现在没有开启这个开关，所以分页操作是没有的。这个开关是通过MyBatisPlus的拦截器的形式存在的，其中的原理这里不分析了，有兴趣的小伙伴可以学习MyBatisPlus这门课程进行详细解读。具体设置方式如下：

**定义MyBatisPlus拦截器并将其设置为Spring管控的bean**

```JAVA
@Configuration
public class MPConfig {
    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor(){
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
        interceptor.addInnerInterceptor(new PaginationInnerInterceptor());
        return interceptor;
    }
}
```

​		上述代码第一行是创建MyBatisPlus的拦截器栈，这个时候拦截器栈中没有具体的拦截器，第二行是初始化了分页拦截器，并添加到拦截器栈中。如果后期开发其他功能，需要添加全新的拦截器，按照第二行的格式继续add进去新的拦截器就可以了。

**总结**

1. 使用IPage封装分页数据
2. 分页操作依赖MyBatisPlus分页拦截器实现功能
3. 借助MyBatisPlus日志查阅执行SQL语句



#### 4.数据层开发——条件查询功能制作

​		除了分页功能，MyBatisPlus还提供有强大的条件查询功能。以往我们写条件查询要自己动态拼写复杂的SQL语句，现在简单了，MyBatisPlus将这些操作都制作成API接口，调用一个又一个的方法就可以实现各种条件的拼装。这里给大家普及一下基本格式，详细的操作还是到MyBatisPlus的课程中查阅吧。

​		下面的操作就是执行一个模糊匹配对应的操作，由like条件书写变为了like方法的调用。

```JAVA
@Test
void testGetBy(){
    QueryWrapper<Book> qw = new QueryWrapper<>();
    qw.like("name","Spring");
    bookDao.selectList(qw);
}
```

​		其中第一句QueryWrapper对象是一个用于封装查询条件的对象，该对象可以动态使用API调用的方法添加条件，最终转化成对应的SQL语句。第二句就是一个条件了，需要什么条件，使用QueryWapper对象直接调用对应操作即可。比如做大于小于关系，就可以使用lt或gt方法，等于使用eq方法，等等，此处不做更多的解释了。

​		这组API使用还是比较简单的，但是关于属性字段名的书写存在着安全隐患，比如查询字段name，当前是以字符串的形态书写的，万一写错，编译器还没有办法发现，只能将问题抛到运行器通过异常堆栈告诉开发者，不太友好。

​		MyBatisPlus针对字段检查进行了功能升级，全面支持Lambda表达式，就有了下面这组API。由QueryWrapper对象升级为LambdaQueryWrapper对象，这下就避免了上述问题的出现。

```JAVA
@Test
void testGetBy2(){
    String name = "1";
    LambdaQueryWrapper<Book> lqw = new LambdaQueryWrapper<Book>();
    lqw.like(Book::getName,name);
    bookDao.selectList(lqw);
}
```

​		为了便于开发者动态拼写SQL，防止将null数据作为条件使用，MyBatisPlus还提供了动态拼装SQL的快捷书写方式。

```JAVA
@Test
void testGetBy2(){
    String name = "1";
    LambdaQueryWrapper<Book> lqw = new LambdaQueryWrapper<Book>();
    //if(name != null) lqw.like(Book::getName,name);		//方式一：JAVA代码控制
    lqw.like(name != null,Book::getName,name);				//方式二：API接口提供控制开关
    bookDao.selectList(lqw);
}
```

​		其实就是个格式，没有区别。关于MyBatisPlus的基础操作就说到这里吧，如果这一块知识不太熟悉的小伙伴建议还是完整的学习一下MyBatisPlus的知识吧，这里只是蜻蜓点水的用了几个操作而已。

**总结**

1. 使用QueryWrapper对象封装查询条件

2. 推荐使用LambdaQueryWrapper对象

3. 所有查询操作封装成方法调用

4. 查询条件支持动态条件拼装

    

#### 5.业务层开发

​		数据层开发告一段落，下面进行业务层开发，其实标准业务层开发很多初学者认为就是调用数据层，怎么说呢？这个理解是没有大问题的，更精准的说法应该是<font color="#ff0000"><b>组织业务逻辑功能，并根据业务需求，对数据持久层发起调用</b></font>。有什么差别呢？目标是为了组织出符合需求的业务逻辑功能，至于调不调用数据层还真不好说，有需求就调用，没有需求就不调用。

​		一个常识性的知识普及一下，业务层的方法名定义一定要与业务有关，例如登录操作

```JAVA
login(String username,String password);
```

​		而数据层的方法名定义一定与业务无关，是一定，不是可能，也不是有可能，例如根据用户名密码查询

```JAVA
selectByUserNameAndPassword(String username,String password);
```

​		我们在开发的时候是可以根据完成的工作不同划分成不同职能的开发团队的。比如一个哥们制作数据层，他就可以不知道业务是什么样子，拿到的需求文档要求可能是这样的

```tex
接口：传入用户名与密码字段，查询出对应结果，结果是单条数据
接口：传入ID字段，查询出对应结果，结果是单条数据
接口：传入离职字段，查询出对应结果，结果是多条数据
```

​		但是进行业务功能开发的哥们，拿到的需求文档要求差别就很大

```tex
接口：传入用户名与密码字段，对用户名字段做长度校验，4-15位，对密码字段做长度校验，8到24位，对密码字段做特殊字符校验，不允许存在空格，查询结果为对象。如果为null，返回BusinessException，封装消息码INFO_LOGON_USERNAME_PASSWORD_ERROR
```

​		你比较一下，能是一回事吗？差别太大了，所以说业务层方法定义与数据层方法定义差异化很大，只不过有些入门级的开发者手懒或者没有使用过公司相关的ISO标准化文档而已。

​		多余的话不说了，咱们做案例就简单制作了，业务层接口定义如下：

```JAVA
public interface BookService {
    Boolean save(Book book);
    Boolean update(Book book);
    Boolean delete(Integer id);
    Book getById(Integer id);
    List<Book> getAll();
    IPage<Book> getPage(int currentPage,int pageSize);
}
```

​		业务层实现类如下，转调数据层即可：

```JAVA
@Service
public class BookServiceImpl implements BookService {

    @Autowired
    private BookDao bookDao;

    @Override
    public Boolean save(Book book) {
        return bookDao.insert(book) > 0;
    }

    @Override
    public Boolean update(Book book) {
        return bookDao.updateById(book) > 0;
    }

    @Override
    public Boolean delete(Integer id) {
        return bookDao.deleteById(id) > 0;
    }

    @Override
    public Book getById(Integer id) {
        return bookDao.selectById(id);
    }

    @Override
    public List<Book> getAll() {
        return bookDao.selectList(null);
    }

    @Override
    public IPage<Book> getPage(int currentPage, int pageSize) {
        IPage page = new Page(currentPage,pageSize);
        bookDao.selectPage(page,null);
        return page;
    }
}
```

​		别忘了对业务层接口进行测试，测试类如下：

```JAVA
@SpringBootTest
public class BookServiceTest {
    @Autowired
    private IBookService bookService;

    @Test
    void testGetById(){
        System.out.println(bookService.getById(4));
    }
    @Test
    void testSave(){
        Book book = new Book();
        book.setType("测试数据123");
        book.setName("测试数据123");
        book.setDescription("测试数据123");
        bookService.save(book);
    }
    @Test
    void testUpdate(){
        Book book = new Book();
        book.setId(17);
        book.setType("-----------------");
        book.setName("测试数据123");
        book.setDescription("测试数据123");
        bookService.updateById(book);
    }
    @Test
    void testDelete(){
        bookService.removeById(18);
    }

    @Test
    void testGetAll(){
        bookService.list();
    }

    @Test
    void testGetPage(){
        IPage<Book> page = new Page<Book>(2,5);
        bookService.page(page);
        System.out.println(page.getCurrent());
        System.out.println(page.getSize());
        System.out.println(page.getTotal());
        System.out.println(page.getPages());
        System.out.println(page.getRecords());
    }

}
```

**总结**

1. Service接口名称定义成业务名称，并与Dao接口名称进行区分
2. 制作测试类测试Service功能是否有效



##### 业务层快速开发

​		其实MyBatisPlus技术不仅提供了数据层快速开发方案，业务层MyBatisPlus也给了一个通用接口，个人观点不推荐使用，凑合能用吧，其实就是一个封装+继承的思想，代码给出，实际开发慎用。

​		业务层接口快速开发

```JAVA
public interface IBookService extends IService<Book> {
    //添加非通用操作API接口
}
```

​		业务层接口实现类快速开发，关注继承的类需要传入两个泛型，一个是数据层接口，另一个是实体类。

```JAVA
@Service
public class BookServiceImpl extends ServiceImpl<BookDao, Book> implements IBookService {
    @Autowired
    private BookDao bookDao;
	//添加非通用操作API
}
```

​		如果感觉MyBatisPlus提供的功能不足以支撑你的使用需要（其实是一定不能支撑的，因为需求不可能是通用的），在原始接口基础上接着定义新的API接口就行了，此处不再说太多了，就是自定义自己的操作了，但是不要和已有的API接口名冲突即可。

**总结**

1. 使用通用接口（ISerivce<T>）快速开发Service
2. 使用通用实现类（ServiceImpl<M,T>）快速开发ServiceImpl
3. 可以在通用接口基础上做功能重载或功能追加
4. 注意重载时不要覆盖原始操作，避免原始提供的功能丢失



#### 6.表现层开发

​		终于做到表现层了，做了这么多都是基础工作。其实你现在回头看看，哪里还有什么SpringBoot的影子？前面1,2步就搞完了。继续完成表现层制作吧，咱们表现层的开发使用基于Restful的表现层接口开发，功能测试通过Postman工具进行。

​		表现层接口如下:

```JAVA
@RestController
@RequestMapping("/books")
public class BookController2 {

    @Autowired
    private IBookService bookService;

    @GetMapping
    public List<Book> getAll(){
        return bookService.list();
    }

    @PostMapping
    public Boolean save(@RequestBody Book book){
        return bookService.save(book);
    }

    @PutMapping
    public Boolean update(@RequestBody Book book){
        return bookService.modify(book);
    }

    @DeleteMapping("{id}")
    public Boolean delete(@PathVariable Integer id){
        return bookService.delete(id);
    }

    @GetMapping("{id}")
    public Book getById(@PathVariable Integer id){
        return bookService.getById(id);
    }

    @GetMapping("{currentPage}/{pageSize}")
    public IPage<Book> getPage(@PathVariable int currentPage,@PathVariable int pageSize){
        return bookService.getPage(currentPage,pageSize, null);
    }
}
```

​		在使用Postman测试时关注提交类型，对应上即可，不然就会报405的错误码了。

**普通GET请求**

![image-20211129152237371](img\boot\image-20211129152237371.png)

**PUT请求传递json数据，后台实用@RequestBody接收数据**

![image-20211129152300177](img\boot\image-20211129152300177.png)

**GET请求传递路径变量，后台实用@PathVariable接收数据**

![image-20211129152320073](C:\Users\itcast\AppData\Roaming\Typora\typora-user-images\image-20211129152320073.png)

**总结**

1. 基于Restful制作表现层接口
    - 新增：POST
    - 删除：DELETE
    - 修改：PUT
    - 查询：GET
2. 接收参数
    - 实体数据：@RequestBody
    - 路径变量：@PathVariable



#### 7.表现层消息一致性处理

​		目前我们通过Postman测试后业务层接口功能是通的，但是这样的结果给到前端开发者会出现一个小问题。不同的操作结果所展示的数据格式差异化严重。

​	**增删改操作结果**

```tex
true
```

​	**查询单个数据操作结果**

```json
{
    "id": 1,
    "type": "计算机理论",
    "name": "Spring实战 第5版",
    "description": "Spring入门经典教程"
}
```

​	**查询全部数据操作结果**

```json
[
    {
        "id": 1,
        "type": "计算机理论",
        "name": "Spring实战 第5版",
        "description": "Spring入门经典教程"
    },
    {
        "id": 2,
        "type": "计算机理论",
        "name": "Spring 5核心原理与30个类手写实战",
        "description": "十年沉淀之作"
    }
]
```

​		每种不同操作返回的数据格式都不一样，而且还不知道以后还会有什么格式，这样的结果让前端人员看了是很容易让人崩溃的，必须将所有操作的操作结果数据格式统一起来，需要设计表现层返回结果的模型类，用于后端与前端进行数据格式统一，也称为**前后端数据协议**

```JAVA
@Data
public class R {
    private Boolean flag;
    private Object data;
}
```

​		其中flag用于标识操作是否成功，data用于封装操作数据，现在的数据格式就变了

```JSON
{
    "flag": true,
    "data":{
        "id": 1,
        "type": "计算机理论",
        "name": "Spring实战 第5版",
        "description": "Spring入门经典教程"
    }
}
```

​		表现层开发格式也需要转换一下

<img src="img\boot\image-20211129153301262.png" alt="image-20211129153301262" style="zoom:80%;" />

<img src="img\boot\image-20211129153319052.png" alt="image-20211129153319052" style="zoom:80%;" />

<img src="img\boot\image-20211129153332499.png" alt="image-20211129153332499" style="zoom:80%;" />

​		结果这么一折腾，全格式统一，现在后端发送给前端的数据格式就统一了，免去了不少前端解析数据的烦恼。

**总结**

1. 设计统一的返回值结果类型便于前端开发读取数据

2. 返回值结果类型可以根据需求自行设定，没有固定格式

3. 返回值结果模型类用于后端与前端进行数据格式统一，也称为前后端数据协议

    

#### 8.前后端联通性测试

​		后端的表现层接口开发完毕，就可以进行前端的开发了。

​		将前端人员开发的页面保存到lresources目录下的static目录中，建议执行maven的clean生命周期，避免缓存的问题出现。

​	<img src="img\boot\image-20211129153831990.png" alt="image-20211129153831990"/>

​		在进行具体的功能开发之前，先做联通性的测试，通过页面发送异步提交（axios），这一步调试通过后再进行进一步的功能开发。

```js
//列表
getAll() {
	axios.get("/books").then((res)=>{
		console.log(res.data);
	});
},
```

​		只要后台代码能够正常工作，前端能够在日志中接收到数据，就证明前后端是通的，也就可以进行下一步的功能开发了。

**总结**

1. 单体项目中页面放置在resources/static目录下
2. created钩子函数用于初始化页面时发起调用
3. 页面使用axios发送异步请求获取数据后确认前后端是否联通

#### 9.页面基础功能开发

##### 	F-1.列表功能（非分页版）

​		列表功能主要操作就是加载完数据，将数据展示到页面上，此处要利用VUE的数据模型绑定，发送请求得到数据，然后页面上读取指定数据即可。

​		**页面数据模型定义**

```js
data:{
	dataList: [],		//当前页要展示的列表数据
	...
},
```

​		异步请求获取数据

```JS
//列表
getAll() {
    axios.get("/books").then((res)=>{
        this.dataList = res.data.data;
    });
},
```

​		这样在页面加载时就可以获取到数据，并且由VUE将数据展示到页面上了。

**总结**：将查询数据返回到页面，利用前端数据绑定进行数据展示

##### 	F-2.添加功能

​		添加功能用于收集数据的表单是通过一个弹窗展示的，因此在添加操作前首先要进行弹窗的展示，添加后隐藏弹窗即可。因为这个弹窗一直存在，因此当页面加载时首先设置这个弹窗为不可显示状态，需要展示，切换状态即可。

​		**默认状态**

```js
data:{
	dialogFormVisible: false,	//添加表单是否可见
	...
},
```

​		**切换为显示状态**

```JS
//弹出添加窗口
handleCreate() {
	this.dialogFormVisible = true;
},
```

​		由于每次添加数据都是使用同一个弹窗录入数据，所以每次操作的痕迹将在下一次操作时展示出来，需要在每次操作之前清理掉上次操作的痕迹。

​		**定义清理数据操作**

```js
//重置表单
resetForm() {
    this.formData = {};
},
```

​		**切换弹窗状态时清理数据**

```js
//弹出添加窗口
handleCreate() {
    this.dialogFormVisible = true;
    this.resetForm();
},
```

​		至此准备工作完成，下面就要调用后台完成添加操作了。

​		**添加操作**

```js
//添加
handleAdd () {
    //发送异步请求
    axios.post("/books",this.formData).then((res)=>{
        //如果操作成功，关闭弹层，显示数据
        if(res.data.flag){
            this.dialogFormVisible = false;
            this.$message.success("添加成功");
        }else {
            this.$message.error("添加失败");
        }
    }).finally(()=>{
        this.getAll();
    });
},
```

1. 将要保存的数据传递到后台，通过post请求的第二个参数传递json数据到后台
2. 根据返回的操作结果决定下一步操作
    - 如何是true就关闭添加窗口，显示添加成功的消息
    - 如果是false保留添加窗口，显示添加失败的消息
3. 无论添加是否成功，页面均进行刷新，动态加载数据（对getAll操作发起调用）

​		**取消添加操作**

```JS
//取消
cancel(){
    this.dialogFormVisible = false;
    this.$message.info("操作取消");
},
```

**总结**

1. 请求方式使用POST调用后台对应操作
2. 添加操作结束后动态刷新页面加载数据
3. 根据操作结果不同，显示对应的提示信息
4. 弹出添加Div时清除表单数据

##### 	F-3.删除功能

​		模仿添加操作制作删除功能，差别之处在于删除操作仅传递一个待删除的数据id到后台即可。

​		**删除操作**

```JS
// 删除
handleDelete(row) {
    axios.delete("/books/"+row.id).then((res)=>{
        if(res.data.flag){
            this.$message.success("删除成功");
        }else{
            this.$message.error("删除失败");
        }
    }).finally(()=>{
        this.getAll();
    });
},
```

​		**删除操作提示信息**

```JS
// 删除
handleDelete(row) {
    //1.弹出提示框
    this.$confirm("此操作永久删除当前数据，是否继续？","提示",{
        type:'info'
    }).then(()=>{
        //2.做删除业务
        axios.delete("/books/"+row.id).then((res)=>{
       		if(res.data.flag){
            	this.$message.success("删除成功");
        	}else{
            	this.$message.error("删除失败");
        	}
        }).finally(()=>{
            this.getAll();
        });
    }).catch(()=>{
        //3.取消删除
        this.$message.info("取消删除操作");
    });
}，	
```

**总结**

1. 请求方式使用Delete调用后台对应操作
2. 删除操作需要传递当前行数据对应的id值到后台
3. 删除操作结束后动态刷新页面加载数据
4. 根据操作结果不同，显示对应的提示信息
5. 删除操作前弹出提示框避免误操作

##### 	F-4.修改功能

​		修改功能可以说是列表功能、删除功能与添加功能的合体。几个相似点如下：

1. 页面也需要有一个弹窗用来加载修改的数据，这一点与添加相同，都是要弹窗

2. 弹出窗口中要加载待修改的数据，而数据需要通过查询得到，这一点与查询全部相同，都是要查数据

3. 查询操作需要将要修改的数据id发送到后台，这一点与删除相同，都是传递id到后台

4. 查询得到数据后需要展示到弹窗中，这一点与查询全部相同，都是要通过数据模型绑定展示数据

5. 修改数据时需要将被修改的数据传递到后台，这一点与添加相同，都是要传递数据

    所以整体上来看，修改功能就是前面几个功能的大合体

    **查询并展示数据**

```JS
//弹出编辑窗口
handleUpdate(row) {
    axios.get("/books/"+row.id).then((res)=>{
        if(res.data.flag){
            //展示弹层，加载数据
            this.formData = res.data.data;
            this.dialogFormVisible4Edit = true;
        }else{
            this.$message.error("数据同步失败，自动刷新");
        }
    });
},
```

​		**修改操作**

```JS
//修改
handleEdit() {
    axios.put("/books",this.formData).then((res)=>{
        //如果操作成功，关闭弹层并刷新页面
        if(res.data.flag){
            this.dialogFormVisible4Edit = false;
            this.$message.success("修改成功");
        }else {
            this.$message.error("修改失败，请重试");
        }
    }).finally(()=>{
        this.getAll();
    });
},
```

**总结**

1. 加载要修改数据通过传递当前行数据对应的id值到后台查询数据（同删除与查询全部）
2. 利用前端双向数据绑定将查询到的数据进行回显（同查询全部）
3. 请求方式使用PUT调用后台对应操作（同新增传递数据）
4. 修改操作结束后动态刷新页面加载数据（同新增）
5. 根据操作结果不同，显示对应的提示信息（同新增）

#### 10.业务消息一致性处理

​		目前的功能制作基本上达成了正常使用的情况，什么叫正常使用呢？也就是这个程序不出BUG，如果我们搞一个BUG出来，你会发现程序马上崩溃掉。比如后台手工抛出一个异常，看看前端接收到的数据什么样子。

```json
{
    "timestamp": "2021-09-15T03:27:31.038+00:00",
    "status": 500,
    "error": "Internal Server Error",
    "path": "/books"
}
```

​		面对这种情况，前端的同学又不会了，这又是什么格式？怎么和之前的格式不一样？

```json
{
    "flag": true,
    "data":{
        "id": 1,
        "type": "计算机理论",
        "name": "Spring实战 第5版",
        "description": "Spring入门经典教程"
    }
}
```

​		看来不仅要对正确的操作数据格式做处理，还要对错误的操作数据格式做同样的格式处理。

​		首先在当前的数据结果中添加消息字段，用来兼容后台出现的操作消息。

```JAVA
@Data
public class R{
    private Boolean flag;
    private Object data;
    private String msg;		//用于封装消息
}
```

​		后台代码也要根据情况做处理，当前是模拟的错误。

```JAVA
@PostMapping
public R save(@RequestBody Book book) throws IOException {
    Boolean flag = bookService.insert(book);
    return new R(flag , flag ? "添加成功^_^" : "添加失败-_-!");
}
```

​		然后在表现层做统一的异常处理，使用SpringMVC提供的异常处理器做统一的异常处理。

```JAVA
@RestControllerAdvice
public class ProjectExceptionAdvice {
    @ExceptionHandler(Exception.class)
    public R doOtherException(Exception ex){
        //记录日志
        //发送消息给运维
        //发送邮件给开发人员,ex对象发送给开发人员
        ex.printStackTrace();
        return new R(false,null,"系统错误，请稍后再试！");
    }
}
```

​		页面上得到数据后，先判定是否有后台传递过来的消息，标志就是当前操作是否成功，如果返回操作结果false，就读取后台传递的消息。

```JS
//添加
handleAdd () {
	//发送ajax请求
    axios.post("/books",this.formData).then((res)=>{
        //如果操作成功，关闭弹层，显示数据
        if(res.data.flag){
            this.dialogFormVisible = false;
            this.$message.success("添加成功");
        }else {
            this.$message.error(res.data.msg);			//消息来自于后台传递过来，而非固定内容
        }
    }).finally(()=>{
        this.getAll();
    });
},
```

**总结**

1. 使用注解@RestControllerAdvice定义SpringMVC异常处理器用来处理异常的
2. 异常处理器必须被扫描加载，否则无法生效
3. 表现层返回结果的模型类中添加消息属性用来传递消息到页面

#### 11.页面功能开发

##### 	F-5.分页功能

​		分页功能的制作用于替换前面的查询全部，其中要使用到elementUI提供的分页组件。

```js
<!--分页组件-->
<div class="pagination-container">
    <el-pagination
		class="pagiantion"
		@current-change="handleCurrentChange"
		:current-page="pagination.currentPage"
		:page-size="pagination.pageSize"
		layout="total, prev, pager, next, jumper"
		:total="pagination.total">
    </el-pagination>
</div>
```

​		为了配合分页组件，封装分页对应的数据模型。

```js
data:{
	pagination: {	
		//分页相关模型数据
		currentPage: 1,	//当前页码
		pageSize:10,	//每页显示的记录数
		total:0,		//总记录数
	}
},
```

​		修改查询全部功能为分页查询，通过路径变量传递页码信息参数。

```JS
getAll() {
    axios.get("/books/"+this.pagination.currentPage+"/"+this.pagination.pageSize).then((res) => {
    });
},
```

​		后台提供对应的分页功能。

```JAVA
@GetMapping("/{currentPage}/{pageSize}")
public R getAll(@PathVariable Integer currentPage,@PathVariable Integer pageSize){
    IPage<Book> pageBook = bookService.getPage(currentPage, pageSize);
    return new R(null != pageBook ,pageBook);
}
```

​		页面根据分页操作结果读取对应数据，并进行数据模型绑定。

```JS
getAll() {
    axios.get("/books/"+this.pagination.currentPage+"/"+this.pagination.pageSize).then((res) => {
        this.pagination.total = res.data.data.total;
        this.pagination.currentPage = res.data.data.current;
        this.pagination.pagesize = res.data.data.size;
        this.dataList = res.data.data.records;
    });
},
```

​		对切换页码操作设置调用当前分页操作。

```JS
//切换页码
handleCurrentChange(currentPage) {
    this.pagination.currentPage = currentPage;
    this.getAll();
},
```

**总结**

1. 使用el分页组件
2. 定义分页组件绑定的数据模型
3. 异步调用获取分页数据
4. 分页数据页面回显

##### 	F-6.删除功能维护

​		由于使用了分页功能，当最后一页只有一条数据时，删除操作就会出现BUG，最后一页无数据但是独立展示，对分页查询功能进行后台功能维护，如果当前页码值大于最大页码值，重新执行查询。其实这个问题解决方案很多，这里给出比较简单的一种处理方案。

```JAVA
@GetMapping("{currentPage}/{pageSize}")
public R getPage(@PathVariable int currentPage,@PathVariable int pageSize){
    IPage<Book> page = bookService.getPage(currentPage, pageSize);
    //如果当前页码值大于了总页码值，那么重新执行查询操作，使用最大页码值作为当前页码值
    if( currentPage > page.getPages()){
        page = bookService.getPage((int)page.getPages(), pageSize);
    }
    return new R(true, page);
}
```



##### 	F-7.条件查询功能

​		最后一个功能来做条件查询，其实条件查询可以理解为分页查询的时候除了携带分页数据再多带几个数据的查询。这些多带的数据就是查询条件。比较一下不带条件的分页查询与带条件的分页查询差别之处，这个功能就好做了

- 页面封装的数据：带不带条件影响的仅仅是一次性传递到后台的数据总量，由传递2个分页相关数据转换成2个分页数据加若干个条件

- 后台查询功能：查询时由不带条件，转换成带条件，反正不带条件的时候查询条件对象使用的是null，现在换成具体条件，差别不大

- 查询结果：不管带不带条件，出来的数据只是有数量上的差别，其他都差别，这个可以忽略

    经过上述分析，看来需要在页面发送请求的格式方面做一定的修改，后台的调用数据层操作时发送修改，其他没有区别。

    页面发送请求时，两个分页数据仍然使用路径变量，其他条件采用动态拼装url参数的形式传递。

    **页面封装查询条件字段**

    ```vue
    pagination: {		
    //分页相关模型数据
    	currentPage: 1,		//当前页码
    	pageSize:10,		//每页显示的记录数
    	total:0,			//总记录数
    	name: "",
    	type: "",
    	description: ""
    },
    ```

    页面添加查询条件字段对应的数据模型绑定名称

    ```HTML
    <div class="filter-container">
        <el-input placeholder="图书类别" v-model="pagination.type" class="filter-item"/>
        <el-input placeholder="图书名称" v-model="pagination.name" class="filter-item"/>
        <el-input placeholder="图书描述" v-model="pagination.description" class="filter-item"/>
        <el-button @click="getAll()" class="dalfBut">查询</el-button>
        <el-button type="primary" class="butT" @click="handleCreate()">新建</el-button>
    </div>
    ```

    将查询条件组织成url参数，添加到请求url地址中，这里可以借助其他类库快速开发，当前使用手工形式拼接，降低学习要求

    ```JS
    getAll() {
        //1.获取查询条件,拼接查询条件
        param = "?name="+this.pagination.name;
        param += "&type="+this.pagination.type;
        param += "&description="+this.pagination.description;
        console.log("-----------------"+ param);
        axios.get("/books/"+this.pagination.currentPage+"/"+this.pagination.pageSize+param).then((res) => {
            this.dataList = res.data.data.records;
        });
    },
    ```

    后台代码中定义实体类封查询条件

    ```JAVA
    @GetMapping("{currentPage}/{pageSize}")
    public R getAll(@PathVariable int currentPage,@PathVariable int pageSize,Book book) {
        System.out.println("参数=====>"+book);
        IPage<Book> pageBook = bookService.getPage(currentPage,pageSize);
        return new R(null != pageBook ,pageBook);
    }
    ```

    对应业务层接口与实现类进行修正

    ```JAVA
    public interface IBookService extends IService<Book> {
        IPage<Book> getPage(Integer currentPage,Integer pageSize,Book queryBook);
    }
    ```

    ```JAVA
    @Service
    public class BookServiceImpl2 extends ServiceImpl<BookDao,Book> implements IBookService {
        public IPage<Book> getPage(Integer currentPage,Integer pageSize,Book queryBook){
            IPage page = new Page(currentPage,pageSize);
            LambdaQueryWrapper<Book> lqw = new LambdaQueryWrapper<Book>();
            lqw.like(Strings.isNotEmpty(queryBook.getName()),Book::getName,queryBook.getName());
            lqw.like(Strings.isNotEmpty(queryBook.getType()),Book::getType,queryBook.getType());
            lqw.like(Strings.isNotEmpty(queryBook.getDescription()),Book::getDescription,queryBook.getDescription());
            return bookDao.selectPage(page,lqw);
        }
    }
    ```

    页面回显数据

    ```js
    getAll() {
        //1.获取查询条件,拼接查询条件
        param = "?name="+this.pagination.name;
        param += "&type="+this.pagination.type;
        param += "&description="+this.pagination.description;
        console.log("-----------------"+ param);
        axios.get("/books/"+this.pagination.currentPage+"/"+this.pagination.pageSize+param).then((res) => {
            this.pagination.total = res.data.data.total;
            this.pagination.currentPage = res.data.data.current;
            this.pagination.pagesize = res.data.data.size;
            this.dataList = res.data.data.records;
        });
    },
    ```

**总结**

1. 定义查询条件数据模型（当前封装到分页数据模型中）
2. 异步调用分页功能并通过请求参数传递数据到后台

# 运维实用篇

​		小伙伴解决了一些遇到的问题，并且已经发现了部分问题具有典型性，预计将有些问题在后面篇章的合适位置添加到本套课程中，作为解决方案提供给大家。

​		实用篇共分为两块内容，分别是运维实用篇和开发实用篇。其实划分的标准是我自己制定的，因为这里面的知识有一些还是比较散的，做两个阶段的划分是为了更好的将同类知识点进行归类，帮助学习者找到知识之间的关联性，这样有助于知识的记忆存储转换，经过一系列的知识反复出现与强化练习，将临时记忆转换成永久性记忆。做课程嘛，不能仅以讲完为目标，要以学习者的学习收获为目标，这也是我这么多年教学秉承的基本理念。

运维实用篇的定位是玩转配置，为开发实用篇中做各种技术的整合做好准备工作。与开发实用篇相比，运维实用篇的内容显得略微单薄，并且有部分知识模块在运维实用篇和开发实用篇中都要讲一部分，这些内容都后置到开发实用篇中了。运维实用篇中都包含的内容：

- SpringBoot程序的打包与运行
- 配置高级
- 多环境开发
- 日志

## 打包与运行

1. 开发部门使用Git、SVN等版本控制工具上传工程到版本服务器
2. 服务器使用版本控制工具下载工程
3. 服务器上使用Maven工具在当前真机环境下重新构建项目
4. 启动服务

打包是指将程序转换成一个可执行的文件，所谓运行指不依赖开发环境执行打包产生的文件。上述两个操作都有对应的命令可以快速执行。

### **程序打包**

SpringBoot程序是基于Maven创建的，在Maven中提供有打包的指令，叫做package。本操作可以在Idea环境下执行。

```JAVA
mvn package
```

打包后会产生一个与工程名类似的jar文件，其名称是由模块名+版本号+.jar组成的。

### **程序运行**

程序包打好以后，就可以直接执行了。在程序包所在路径下，执行指令。

```JAVA
java -jar 工程包名.jar
```

执行程序打包指令后，程序正常运行，与在Idea下执行程序没有区别。

<font color="#ff0000"><b>特别关注</b></font>：如果你的计算机中没有安装java的jdk环境，是无法正确执行上述操作的，因为程序执行使用的是java指令。

<font color="#ff0000"><b>特别关注</b></font>：在使用向导创建SpringBoot工程时，pom.xml文件中会有如下配置，这一段配置千万不能删除，否则打包后无法正常执行程序。

```XML
<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

**总结**

1. SpringBoot工程可以基于java环境下独立运行jar文件启动服务
2. SpringBoot工程执行mvn命令package进行打包
3. 执行jar命令：java –jar 工程名.jar

### 打包失败处理

打包以后执行会出现一些问题，导致程序无法正常执行，例如下面的现象

<img src="img\boot\image-20211201094223991.png" alt="image-20211201094223991" style="zoom:67%;" />

要想搞清楚这个问题就要说说.jar文件的工作机制了，知道了这个东西就知道如何避免此类问题的发生了。

搞java开发平时会接触很多jar包，比如mysql的驱动jar包，而上面我们打包程序后得到的也是一个jar文件。这个时候如果你使用上面的java -jar指令去执行mysql的驱动jar包就会出现上述不可执行的现象，而我们的SpringBoot项目为什么能执行呢？其实是因为打包方式不一样。

在SpringBoot工程的pom.xml中有下面这组配置，这组配置决定了打包出来的程序包是否可以执行。

```XML
<build>
    <plugins>
        <plugin>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

我们分别开启这段配置和注释掉这段配置分别执行两次打包，然后观察两次打包后的程序包的差别，共有3处比较明显的特征

- 打包后文件的大小不同
- 打包后所包含的内容不同
- 打包程序中个别文件内容不同

先看第一个现象，文件大小不同。带有配置时打包生成的程序包大小如下：

<img src="img\boot\image-20211201095610270.png" alt="image-20211201095610270" style="zoom:150%;" />

不难看出，带有配置的程序包体积比不带配置的大了30倍。

大的程序包中除了包含小的程序包中的内容，还有别的东西。都有什么呢？回到BOOT-INF目录下，打开lib目录，里面显示了很多个jar文件。

​		<img src="img\boot\image-20211201102025791.png" alt="image-20211201102025791" style="zoom:50%;" />

​		仔细翻阅不难发现，这些jar文件都是我们制作这个工程时导入的坐标对应的文件。大概可以想明白了，SpringBoot程序为了让自己打包生成的程序可以独立运行，不仅将项目中自己开发的内容进行了打包，还把当前工程运行需要使用的jar包全部打包进来了。为什么这样做呢？就是为了可以独立运行。不依赖程序包外部的任何资源可以独立运行当前程序。这也是为什么大的程序包容量是小的程序包容量的30倍的主要原因。

再看看大程序包还有什么不同之处，在最外层目录包含一个org目录，进入此目录，目录名是org\springframework\boot\loader，在里面可以找到一个<font color="#ff0000"><b>JarLauncher.class</b></font>的文件，先记得这个文件。再看这套目录名，明显是一个Spring的目录名，为什么要把Spring框架的东西打包到这个程序包中呢？不清楚。

回到两个程序包的最外层目录，查看名称相同的文件夹META-INF下都有一个叫做MANIFEST.MF的文件，但是大小不同，打开文件，比较内容区别

小容量文件的MANIFEST.MF

```shell
Manifest-Version: 1.0
Implementation-Title: springboot_08_ssmp
Implementation-Version: 0.0.1-SNAPSHOT
Build-Jdk-Spec: 1.8
Created-By: Maven Jar Plugin 3.2.0
```

大容量文件的MANIFEST.MF

```shell
Manifest-Version: 1.0
Spring-Boot-Classpath-Index: BOOT-INF/classpath.idx
Implementation-Title: springboot_08_ssmp
Implementation-Version: 0.0.1-SNAPSHOT
Spring-Boot-Layers-Index: BOOT-INF/layers.idx
Start-Class: com.itheima.SSMPApplication
Spring-Boot-Classes: BOOT-INF/classes/
Spring-Boot-Lib: BOOT-INF/lib/
Build-Jdk-Spec: 1.8
Spring-Boot-Version: 2.5.4
Created-By: Maven Jar Plugin 3.2.0
Main-Class: org.springframework.boot.loader.JarLauncher
```

大文件中明显比小文件中多了几行信息，其中最后一行信息是Main-Class: org.springframework.boot.loader.<font color="#ff0000"><b>JarLauncher</b></font>。这句话什么意思呢？如果使用java -jar执行此程序包，将执行Main-Class属性配置的类，这个类恰巧就是前面看到的那个文件。原来SpringBoot打包程序中出现Spring框架的东西是为这里服务的。而这个org.springframework.boot.loader.<font color="#ff0000"><b>JarLauncher</b></font>类内部要查找Start-Class属性中配置的类，并执行对应的类。这个属性在当前配置中也存在，对应的就是我们的引导类类名。

现在这组设定的作用就搞清楚了

1. SpringBoot程序添加配置后会打出一个特殊的包，包含Spring框架部分功能，原始工程内容，原始工程依赖的jar包
2. 首先读取MANIFEST.MF文件中的Main-Class属性，用来标记执行java -jar命令后运行的类
3. JarLauncher类执行时会找到Start-Class属性，也就是启动类类名
4. 运行启动类时会运行当前工程的内容
5. 运行当前工程时会使用依赖的jar包，从lib目录中查找

SpringBoot打出来了包为了能够独立运行，将所有需要使用的资源全部都添加到了这个包里。这就是为什么这个jar包能独立运行的原因。

再来看之前的报错信息：

<img src="img\boot\image-20211201094223991.png" alt="image-20211201094223991" style="zoom:67%;" />

由于打包时没有使用那段配置，结果打包后形成了一个普通的jar包，在MANIFEST.MF文件中也就没有了Main-Class对应的属性了，所以运行时提示找不到主清单属性，这就是报错的原因。

上述内容搞清楚对我们编程意义并不大，但是对各位小伙伴理清楚SpringBoot工程独立运行的机制是有帮助的。其实整体过程主要是带着大家分析，如果以后遇到了类似的问题，多给自己提问，多问一个为什么，兴趣自己就可以独立解决问题了。

**总结**：spring-boot-maven-plugin插件用于将当前程序打包成一个可以独立运行的程序包

### 命令行启动常见问题及解决方案

在DOS环境下启动SpringBoot工程时，可能会遇到端口占用的问题。

```JAVA
# 查询端口
netstat -ano
# 查询指定端口
netstat -ano |findstr "端口号"
# 根据进程PID查询进程名称
tasklist |findstr "进程PID号"
# 根据PID杀死任务
taskkill /F /PID "进程PID号"
# 根据进程名称杀死任务
taskkill -f -t -im "进程名称"
```

关于打包与运行程序其实还有一系列的配置和参数，下面的内容中遇到再说，这里先开个头，知道如何打包和运行程序。

### Linux下快速启动

- `nohub java -jar xxx.jar > server.log`   后台启动jar，日志保存在 server.log 里
- `ps -ef | grep "java -jar"` 搜索包含 “java -jar”的命令
- `kill -9 pid编号` 杀死进程

## 配置高级

### 临时属性设置

程序包打好以后，里面的配置都已经是固定的了，比如配置了服务器的端口是8080。如果我要启动项目，发现当前我的服务器上已经有应用启动起来并且占用了8080端口，这时候可以通过临时属性的方式快速修改某些配置。

![image-20211206095113771](img\boot\image-20211206095113771.png)             ![image-20211206095524343](img\boot\image-20211206095524343.png)             ![image-20211206095101581](img\boot\image-20211206095101581.png)

```shell
java –jar springboot.jar –-server.port=80
```

在命令输入完毕后，空一格，然后输入两个-号。下面按照属性名=属性值的形式添加对应参数就可以了。当属性存在多级名称时，中间使用点分隔，和properties文件中的属性格式完全相同。

如果你发现要修改的属性不止一个，可以按照上述格式继续写，属性与属性之间使用空格分隔。

```shell
java –jar springboot.jar –-server.port=80 --logging.level.root=debug
```

#### **属性加载优先级**

现在我们的程序配置受两个地方控制了，第一配置文件，第二临时属性。并且我们发现临时属性的加载优先级要高于配置文件的。那是否还有其他的配置方式呢？其实是有的，而且还不少，打开官方文档中对应的内容，就可以查看配置读取的优先顺序。地址奉上：[https://docs.spring.io/spring-boot/docs/current/reference/html/spring-boot-features.html#boot-features-external-config](https://docs.spring.io/spring-boot/docs/current/reference/html/spring-boot-features.html)

<img src="img\boot\image-20211206100859236.png" alt="image-20211206100859236" style="zoom:67%;" />

我们可以看到，居然有14种配置的位置，而我们现在使用的是这里面的2个。第3条Config data说的就是使用配置文件，第11条Command line arguments说的就是使用命令行临时参数。而这14种配置的顺序就是SpringBoot加载配置的顺序，言外之意，命令行临时属性比配置文件的加载优先级高，所以这个列表**上面的优先级低，下面的优先级高**。其实这个东西不用背的，你就记得一点，你最终要什么效果，你自己是知道的，不管这个顺序是怎么个高低排序，开发时一定要配置成你要的顺序为准。这个顺序只是在你想不明白问题的时候帮助你分析罢了。

比如你现在加载了一个user.name属性。结果你发现出来的结果和你想的不一样，那肯定是别的优先级比你高的属性覆盖你的配置属性了，那你就可以看着这个顺序挨个排查。哪个位置有可能覆盖了你的属性。

在yaml中配置了user.name属性值，然后读取出来的时候居然不是自己的配置值，因为在系统属性中有一个属性叫做user.name，两个相互冲突了。而系统属性的加载优先顺序在上面这个列表中是5号，高于3号，所以SpringBoot最终会加载系统配置属性user.name。

**总结**

1. 使用jar命令启动SpringBoot工程时可以使用临时属性替换配置文件中的属性
2. 临时属性添加方式：java –jar 工程名.jar –-属性名=值
3. 多个临时属性之间使用空格分隔
4. 临时属性必须是当前boot工程支持的属性，否则设置无效

#### 开发环境中使用临时属性

开发环境中如何使用临时属性，其实就是Idea界面下如何操作了。

打开SpringBoot引导类的运行界面，在里面找到配置项。其中Program arguments对应的位置就是添加临时属性的，可以加几个试试效果。

<img src="img\boot\image-20211206101947622.png" alt="image-20211206101947622" style="zoom:80%;" />

我们运行main方法的时候，如果想使用main方法的参数，也就是下面的args参数，就是在上面这个位置添加的参数。

```JAVA
public static void main(String[] args) {
}
```

通过这个args就可以获取到参数。再来看我们的引导类是如何书写的

```JAVA
public static void main(String[] args) {
    SpringApplication.run(SSMPApplication.class,args);
}
```

这个args参数居然传递给了run方法，看来在Idea中配置的临时参数就是通过这个位置传递到我们的程序中的。言外之意，这里如果不用这个args是不是就断开了外部传递临时属性的入口呢？是这样的，我们可以使用下面的调用方式，这样外部临时属性就无法进入到SpringBoot程序中了。

```JAVA
public static void main(String[] args) {
    SpringApplication.run(SSMPApplication.class);
}
```

或者还可以使用如下格式来玩这个操作，就是将配置不写在配置文件中，直接写成一个字符串数组，传递给程序入口。当然，这种做法并没有什么实际开发意义。

```JAVA
public static void main(String[] args) {
    String[] arg = new String[1];
    arg[0] = "--server.port=8082";
    SpringApplication.run(SSMPApplication.class, arg);
}
```

**总结**：启动SpringBoot程序时，可以选择是否使用命令行属性为SpringBoot程序传递启动属性

**思考**：现在使用临时属性可以在启动项目前临时更改配置了，如果需要变更的属性偏多，如何处理？

### 配置文件分类

SpringBoot提供了配置文件和临时属性的方式来对程序进行配置。前面一直说的是临时属性，这一节要说说配置文件了。其实这个配置文件我们一直在使用，只不过我们用的是SpringBoot提供的4级配置文件中的其中一个级别。4个级别分别是：

- 类路径下配置文件（一直使用的是这个，也就是resources目录中的application.yml文件）
- 类路径下config目录下配置文件
- 程序包所在目录中配置文件
- 程序包所在目录中config目录下配置文件

其实上述4种文件是提供给你了4种配置文件书写的位置，功能都是一样的，都是做配置的，就是优先级顺序不一样。上面4个文件的加载优先顺序为

- 1、file ：config/application.yml **【最高】**
- 2、file ：application.yml
- 3、classpath：config/application.yml
- 4、classpath：application.yml  **【最低】**

那为什么设计这种多种呢？说一个最典型的应用吧。

- 场景A：你作为一个开发者，你做程序的时候为了方便自己写代码，配置的数据库肯定是连接你自己本机的，咱们使用4这个级别，也就是之前一直用的application.yml。
- 场景B：现在项目开发到了一个阶段，要联调测试了，连接的数据库是测试服务器的数据库，肯定要换一组配置吧。你可以选择把你之前的文件中的内容都改了，目前还不麻烦。
- 场景C：测试完了，一切OK。你继续写你的代码，你发现你原来写的配置文件被改成测试服务器的内容了，你要再改回来。现在明白了不？场景B中把你的内容都改掉了，你现在要重新改回来，以后呢？改来改去吗？

解决方案很简单，用上面的3这个级别的配置文件就可以快速解决这个问题，再写一个配置就行了。两个配置文件共存，因为config目录中的配置加载优先级比你的高，所以配置项如果和级别4里面的内容相同就覆盖了。级别1和2什么时候使用呢？程序打包以后就要用这个级别了，管你程序里面配置写的是什么？我的级别高，可以轻松覆盖你，就不用考虑这些配置冲突的问题了。

**总结**

配置文件分为4种

- 项目类路径配置文件：服务于开发人员本机开发与测试
- 项目类路径config目录中配置文件：服务于项目经理整体调控
- 工程路径配置文件：服务于运维人员配置涉密线上环境
- 工程路径config目录中配置文件：服务于运维经理整体调控

多层级配置文件间的属性采用叠加并覆盖的形式作用于程序。


### 自定义配置文件

之前咱们做配置使用的配置文件都是 application.yml，其实这个文件也是可以改名字的，这样方便维护。比如我2020年4月1日搞活动，走了一组配置，2020年5月1日活动取消，恢复原始配置，这个时候只需要重新更换一下配置文件就可以了。但是你总不能在原始配置文件上修改吧，不然搞完活动以后，活动的配置就留不下来了，不利于维护。

自定义配置文件方式有如下两种：

**方式一：使用临时属性设置配置文件名，注意仅仅是名称，不要带扩展名**

<img src="img\boot\image-20211206105548238.png" alt="image-20211206105548238"/>

**方式二：使用临时属性设置配置文件路径，这个是全路径名**

<img src="img\boot\image-20211206105716450.png" alt="image-20211206105716450"/>

也可以设置加载多个配置文件

<img src="img\boot\image-20211206105750285.png" alt="image-20211206105750285" />

使用的属性一个是spring.config.name，另一个是spring.config.location，这个一定要区别清楚。

<font color="#f0f"><b>温馨提示</b></font>：我们现在研究的都是SpringBoot单体项目，就是单服务器版本。其实企业开发现在更多的是使用基于SpringCloud技术的多服务器项目。这种配置方式和我们现在学习的完全不一样，所有的服务器将不再设置自己的配置文件，而是通过配置中心获取配置，动态加载配置信息。集中管理更为方便。

**总结**

1. 配置文件可以修改名称，通过启动参数设定
2. 配置文件可以修改路径，通过启动参数设定
3. 微服务开发中配置文件通过配置中心进行设置

## 多环境开发

什么是多环境？其实就是说你的电脑上写的程序最终要放到别人的服务器上去运行。每个计算机环境不一样，这就是多环境。常见的多环境开发主要兼顾3种环境设置，开发环境——自己用的，测试环境——自己公司用的，生产环境——甲方爸爸用的。因为这是绝对不同的三台电脑，所以环境肯定有所不同，比如连接的数据库不一样，设置的访问端口不一样等等。

<img src="img\image-20211206110958819.png" alt="image-20211206110958819"/>

### 多环境开发（yaml单一文件版）	

那什么是多环境开发？就是针对不同的环境设置不同的配置属性即可。比如你自己开发时，配置你的端口如下：

```yaml
server:
  port: 80
```

如何想设计两组环境呢？中间使用三个减号分隔开

```yaml
server:
  port: 80
---
server:
  port: 81
```

如何区分两种环境呢？起名字呗

```yaml
spring:
	profiles: pro
server:
	port: 80
---
spring:
	profiles: dev
server:
	port: 81
```

那用哪一个呢？设置默认启动哪个就可以了

```yaml
spring:
	profiles:
		active: pro		# 启动pro
---
spring:
	profiles: pro
server:
	port: 80
---
spring:
	profiles: dev
server:
	port: 81
```

就这么简单，再多来一组环境也OK

```yaml
spring:
	profiles:
		active: pro		# 启动pro
---
spring:
	profiles: pro
server:
	port: 80
---
spring:
	profiles: dev
server:
	port: 81
---
spring:
	profiles: test
server:
	port: 82
```

其中关于环境名称定义上述格式是过时格式，<font color="#f0f"><b>标准格式如下</b></font>

```yaml
spring:
	config:
    	activate:
        	on-profile: pro
```

**总结**

1. 多环境开发需要设置若干种常用环境，例如开发、生产、测试环境
2. yaml格式中设置多环境使用---区分环境设置边界
3. 每种环境的区别在于加载的配置属性不同
4. 启用某种环境时需要指定启动时使用该环境

### 多环境开发（yaml多文件版）	

将所有的配置都放在一个配置文件中，尤其是每一个配置应用场景都不一样，这显然不合理，于是就有了将一个配置文件拆分成多个配置文件的想法。拆分后，每个配置文件中写自己的配置，主配置文件中写清楚用哪一个配置文件就好了。

**主配置文件**

```yaml
spring:
	profiles:
		active: pro		# 启动pro(active: pro 选择的配置文件的名称是 application-pro.yml)
```

**环境配置文件**

```yaml
server:
	port: 80
```

环境配置文件因为每一个都是配置自己的项，所以连名字都不用写里面了。那问题是如何区分这是哪一组配置呢？使用文件名区分。

**application-pro.yaml**

```yaml
server:
	port: 80
```

**application-dev.yaml**

```yaml
server:
	port: 81
```

文件的命名规则为：application-环境名.yml。

在配置文件中，如果某些配置项所有环境都一样，可以将这些项写入到主配置中，只有哪些有区别的项才写入到环境配置文件中。

- 主配置文件中设置公共配置（全局）
- 环境分类配置文件中常用于设置冲突属性（局部）

**总结**

1. 可以使用独立配置文件定义环境属性

2. 独立配置文件便于线上系统维护更新并保障系统安全性


### 多环境开发（properties多文件版）

SpringBoot最早期提供的配置文件格式是properties格式的，这种格式的多环境配置也了解一下吧。

**主配置文件**

```properties
spring.profiles.active=pro
```

**环境配置文件**

**application-pro.properties**

```properties
server.port=80
```

**application-dev.properties**

```properties
server.port=81
```

文件的命名规则为：application-环境名.properties。

**总结**：properties文件多环境配置仅支持多文件格式

### 多环境开发独立配置文件书写技巧

作为程序员在搞配置的时候往往处于一种分久必合合久必分的局面。开始先写一起，后来为了方便维护就拆分。对于多环境开发也是如此，下面给大家说一下如何基于多环境开发做配置独立管理，务必掌握。

**准备工作**：将所有的配置根据功能对配置文件中的信息进行拆分，并制作成独立的配置文件，命名规则如下

- application-devDB.yml
- application-devRedis.yml
- application-devMVC.yml

**使用**：使用include属性在激活指定环境的情况下，同时对多个环境进行加载使其生效，多个环境间使用逗号分隔

```yaml
spring:
	profiles:
    	active: dev
        include: devDB,devRedis,devMVC
```

比较一下，现在相当于加载dev配置时，再加载对应的3组配置，从结构上就很清晰，用了什么，对应的名称是什么

**注意**：当主环境dev与其他环境有相同属性时，主环境属性生效；其他环境中有相同属性时，最后加载的环境属性生效

**改良**：但是上面的设置也有一个问题，比如我要切换dev环境为pro时，include也要修改。因为include属性只能使用一次，这就比较麻烦了。SpringBoot从2.4版开始使用group属性替代include属性，降低了配置书写量。简单说就是我先写好，你爱用哪个用哪个。

```yaml
spring:
	profiles:
    	active: dev
        group:
        	"dev": devDB,devRedis,devMVC
      		"pro": proDB,proRedis,proMVC
      		"test": testDB,testRedis,testMVC
```

现在再来看，如果切换dev到pro，只需要改一下是不是就结束了？完美！

**总结**：<span style="color:red">**多环境开发使用group属性设置配置文件分组，便于线上维护管理**</span>

### 多环境开发控制

多环境开发到这里基本上说完了，最后说一个冲突问题。就是 maven 和 Spring Boot 同时设置多环境的话怎么搞。

要想处理这个冲突问题，你要先理清一个关系，究竟谁在多环境开发中其主导地位。也就是说如果现在都设置了多环境，谁的应该是保留下来的，另一个应该遵从相同的设置。

maven是做什么的？项目构建管理的，最终生成代码包的，Spring Boot 是干什么的？简化开发的。简化，又不是其主导作用。最终还是要靠 maven 来管理整个工程，所以 Spring Boot 应该听 maven的。整个确认后下面就好做了。大体思想如下：

- 先在 maven 环境中设置用什么具体的环境
- 在 SpringBoot 中读取 maven 设置的环境即可

**maven中设置多环境（使用属性方式区分环境）**

```xml
<profiles>
    <profile>
        <id>env_dev</id>
        <properties>
            <profile.active>dev</profile.active>
        </properties>
        <activation>
            <activeByDefault>true</activeByDefault>		<!--默认启动环境-->
        </activation>
    </profile>
    <profile>
        <id>env_pro</id>
        <properties>
            <profile.active>pro</profile.active>
        </properties>
    </profile>
</profiles>
```

**SpringBoot中读取maven设置值**

```yaml
spring:
	profiles:
    	active: @profile.active@
```

上面的@属性名@就是读取maven中配置的属性值的语法格式。

**总结**

1. 当Maven与SpringBoot同时对多环境进行控制时，以Mavn为主，SpringBoot使用@..@占位符读取Maven对应的配置属性值
2. 基于SpringBoot读取Maven配置属性的前提下，如果在Idea下测试工程时pom.xml每次更新需要手动compile方可生效

## 日志

- 编程期调试代码
- 运营期记录信息
    - 记录日常运营重要信息（峰值流量、平均响应时长……）
    - 记录应用报错信息（错误堆栈）
    - 记录运维过程数据（扩容、宕机、报警……）


### 代码中使用日志工具记录日志

日志的使用格式非常固定，直接上操作步骤：

**步骤①**：添加日志记录操作

```JAVA
@RestController
@RequestMapping("/books")
// Spring Boot 默认是 info 级别的，所以只有 info 级别和 info 级别以上的才会输出。debug 级别的日志不会输出
public class BookController extends BaseClass{
    private static final Logger log = LoggerFactory.getLogger(BookController.class);
    @GetMapping
    public String getById(){
        log.debug("debug...");
        log.info("info...");
        log.warn("warn...");
        log.error("error...");
        return "springboot is running...2";
    }
}
```

上述代码中log对象就是用来记录日志的对象，下面的log.debug，log.info这些操作就是写日志的API了。

**步骤②**：设置日志输出级别

日志设置好以后可以根据设置选择哪些参与记录。这里是根据日志的级别来设置的。日志的级别分为6种，分别是：

- TRACE：运行堆栈信息，使用率低
- DEBUG：程序员调试代码使用
- INFO：记录运维过程数据
- WARN：记录运维过程报警数据
- ERROR：记录错误堆栈信息
- FATAL：灾难信息，合并计入ERROR

一般情况下，开发时候使用DEBUG，上线后使用INFO，运维信息记录使用 WARN 即可。下面就设置一下日志级别：

```yaml
# 开启debug模式，输出调试信息，常用于检查系统运行状况
debug: true
```

这么设置太简单粗暴了，日志系统通常都提供了细粒度的控制

```yaml
# 开启debug模式，输出调试信息，常用于检查系统运行状况
debug: true

# 设置日志级别，root表示根节点，即整体应用日志级别
logging:
	level:
    	root: debug # 整体设置为 debug 级别的
    	com.xx.controller: warn # 这个包下设置为 warn 级别的
```

还可以再设置更细粒度的控制

**步骤③**：<span style="color:red">**设置日志组，控制指定包对应的日志输出级别，也可以直接控制指定包对应的日志输出级别，推荐使用**</span>

```yaml
logging:
	# 设置日志组
    group:
    	# 自定义组名，设置当前组中所包含的包
        ebank: com.itheima.controller,com.itheima.dao
        iservice: com.service
    level:
    	root: warn
        # 为对应组设置日志级别
        ebank: debug
        iservice: warn
    	# 为对包设置日志级别
        com.itheima.controller: debug
```

说白了就是总体设置一下，每个包设置一下，如果感觉设置的麻烦，就先把包分个组，对组设置，没了，就这些。

**总结**

1. 日志用于记录开发调试与运维过程消息
2. 日志的级别共6种，通常使用4种即可，分别是DEBUG，INFO,WARN,ERROR
3. 可以通过日志组或代码包的形式进行日志显示级别的控制

### 优化日志对象创建代码

写代码的时候每个类都要写创建日志记录对象，这个可以优化一下，使用前面用过的lombok技术给我们提供的工具类即可。

```JAVA
@RestController
@RequestMapping("/books")
public class BookController extends BaseClass{
    private static final Logger log = LoggerFactory.getLogger(BookController.class);	//这一句可以不写了
}
```

导入lombok后使用注解搞定，日志对象名为log

```JAVA
@Slf4j		//这个注解替代了下面那一行
@RestController
@RequestMapping("/books")
public class BookController extends BaseClass{
    private static final Logger log = LoggerFactory.getLogger(BookController.class);	//这一句可以不写了
}
```

**总结**：基于lombok提供的 `@Slf4j` 注解为类快速添加日志对象

### 日志输出格式控制

日志已经能够记录了，但是目前记录的格式是SpringBoot给我们提供的，如果想自定义控制就需要自己设置了。先分析一下当前日志的记录格式。

![image-20211206123431222](img\boot\image-20211206123431222.png)

- PID：进程ID，用于表明当前操作所处的进程，当多服务同时记录日志时，该值可用于协助程序员调试程序 
- 所属类/接口名：当前显示信息为SpringBoot重写后的信息，名称过长时，简化包名书写为首字母，甚至直接删除

对于单条日志信息来说，日期，触发位置，记录信息是最核心的信息。级别用于做筛选过滤，PID与线程名用于做精准分析。了解这些信息后就可以 DIY 日志格式了。本课程不做详细的研究，有兴趣的小伙伴可以学习相关的知识。下面给出课程中模拟的官方日志模板的书写格式，便于大家学习。

```yaml
logging:
	pattern:
    	console: "%d %clr(%p) --- [%16t] %clr(%-40.40c){cyan} : %m %n"
```

**总结**：日志输出格式设置规则 `%d:日期`；`%m:消息`；`%n:换行`；`%clr:彩色`；`[%16t]:线程，占16位`；`%c:类名，-40是左对齐，.40是控制内容的截取`；`{cyan}:青色`

### 日志文件

日志信息显示，记录已经控制住了，下面就要说一下日志的转存了。日志不能仅显示在控制台上，要把日志记录到文件中，方便后期维护查阅。
对于日志文件的使用存在各种各样的策略，例如每日记录，分类记录，报警后记录等。这里主要研究日志文件如何记录。
记录日志到文件中格式非常简单，设置日志文件名即可。

```YAML
logging:
	file:
    	name: server.log
```

虽然使用上述格式可以将日志记录下来了，但是面对线上的复杂情况，一个文件记录肯定是不能够满足运维要求的，通常会每天记录日志文件，同时为了便于维护，还要限制每个日志文件的大小。下面给出日志文件的常用配置方式：

```YAML
logging:
	file:
    	name: server.log
	logback:
    	rollingpolicy:
        	max-file-size: 3KB
            file-name-pattern: server.%d{yyyy-MM-dd}.%i.log
```

以上格式是基于logback日志技术设置每日日志文件的设置格式，要求容量到达3KB以后就转存信息到第二个文件中。文件命名规则中的%d标识日期，%i是一个递增变量，用于区分日志文件。

**总结**：日志记录到文件、日志文件格式设置

