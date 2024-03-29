# 快速入门

## 注解补充

> @Inherited 元注解的作用

假定注解 @Demo 被 @Inherited 修饰

```java
@Inherited
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
public @interface Demo {}
```

类 Father 上有 @Demo 注解，则 Father 的子类 Son 可以继承到这个注解

```java
@Demo
class Father{}
class Son extends Father{}
```

在解析 Son 上的注解时，可以获取到 @Demo 注解。

```java
// 测试子类继承注解
public class ExtendAnnotation {
    public static void main(String[] args) {
        Annotation[] annotations =
                Son.class.getAnnotations();
        Arrays.stream(annotations).forEach(System.out::println);
        // @com.ex.anno.Demo() 子类继承到了父类的注解 @Demo
    }
}
```

## 介绍

Spring Boot 是整合 Spring 技术栈的一站式框架，是简化 Spring 技术栈的快速开发脚手架。能快速创建出生产级别的 Spring 应用。

> <b>优点</b>

- Create stand-alone Spring applications 【创建独立 Spring 应用】
- Embed Tomcat, Jetty or Undertow directly (no need to deploy WAR files)【内嵌 web 服务器，支持 tomcat、jetty 和 undertow。】
- Provide opinionated 'starter' dependencies to simplify your build configuration【自动 starter 依赖，简化构建配置】
- Automatically configure Spring and 3rd party libraries whenever possible【自动配置 Spring 以及第三方功能】
- Provide production-ready features such as metrics, health checks, and externalized configuration【提供生产级别的监控、健康检查及外部化配置】
- Absolutely no code generation and no requirement for XML configuration【无代码生成、无需编写 XML】

> <b>缺点</b>

- 迭代快，需要时刻关注变化

- 封装太深，内部原理复杂，不易精通

## 创建项目

### IDEA 创建项目

使用 Spring Initializer 快速创建，直接选择它，不选择 maven。然后给 maven 的 settings.xml 配置文件的 profiles 标签添加

```xml
<parent>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-parent</artifactId>
    <version>1.5.9.RELEASE</version>
</parent>
<dependencies>
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

编写一个主程序；启动 Spring Boot 应用，用 @SpringBootApplication 来标注一个主程序类，说明这是一个Spring Boot应用

```java
@SpringBootApplication
public class HelloWorldMainApplication {

    public static void main(String[] args) {
        // Spring应用启动起来
        SpringApplication.run(HelloWorldMainApplication.class,args);
    }
}
```

编写相关的 Controller、Service

```java
@RestController // @RestController = @Controller + @ResponseBody 
public class HelloController {
    @RequestMapping("/hello")
    public String hello(){
        return "Hello World!";
    }
}
```

运行主程序测试

> <b>简化部署</b>

```xml
<!-- 这个插件，可以将应用打包成一个可执行的jar包 -->
<build>
    <plugins>
        <plugin>
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

将这个应用打成 jar 包，直接使用 java -jar 的命令进行执行；打成 jar 包可以直接用 maven 的 package 命令或者控制台 mvn package。

使用 `mvn Spring-boot:run` 启动

<span style="color:red"><b>注意：</b>如果引入了 JDBC 相关的依赖，需要我们在配置文件中配置连接数据库相关的操作。</span>

### 官方骨架创建

> 官方骨架创建遇到的问题

- 有时候 IDEA 创建项目总是出错，我们可以直接去官网选好依赖，下载过来，导入到 IDEA 中。

- 导入后用 maven 的 Reload All Maven Projects 导入所有的依赖，导入后发现下面这个配置文件报错。

```xml
<plugin>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-maven-plugin</artifactId>
</plugin>  
```

加个对应的版本号就可以了

```xml
<plugin>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-maven-plugin</artifactId>
    <version>2.4.2</version>
</plugin>  
```

### 阿里云镜像创建

创建工程时，切换选择 starter 服务路径，然后手工输入阿里云地址即可，地址：http://start.aliyun.com 或 https://start.aliyun.com

<div align="center"><img src="img/boot/image-20211122163605950.png"></div>

阿里为了便于自己公司开发使用，特此在依赖坐标中添加了一些阿里自主的技术，所以在依赖选择列表中，我们有了更多的选择。此外，阿里提供的地址更符合国内开发者的使用习惯，里面有一些 Spring Boot 官网上没有给出的坐标。

<div align="center"><img src="img/boot/image-20211122163937408.png"></div>

阿里云地址默认创建的 Spring Boot 工程版本是 <font color="#ff0000"><b>2.4.1</b></font>，所以如果想更换其他的版本，创建项目后在 pom 文件中手工修改，然后刷新一下，加载新版本信息。

<b style="color:red">注意：</b>阿里云提供的工程创建地址初始化完毕后和使用 Spring Boot 官网创建出来的工程略有区别，主要是在配置文件的形式上有区别，这个信息在 Spring Boot 程序的执行流程时讲解。

### 目录结构说明

> Spring Boot 目录解析

- resources 文件夹中目录结构
    - static：保存静态资源；js css img
    - templates：保存页面资源；Spring Boot 默认不支持 jsp 需要进行配置。
    - mybatis 的配置文件之类的需要放在 resources 文件夹下面。resources 是资源的根路径。可以把 resources 当成编译后的 classes 文件夹。resources 下的文件最终都会被部署到 classpath 文件下。

### 启动类&热部署设置

> 启动类说明

```java
@SpringBootApplication
// 配置MyBatis的Mapper扫描，如果 Mapper 上标注了注解 @Mapper 可以不进行扫描。
// 使用了 @MapperSacn 就不需要用 @Mapper 了。
// https://blog.csdn.net/xiaojin21cen/article/details/103273172
@MapperScan("cn.baobaoxuxu.community.mapper") 
public class CommunityApplication {
    // 程序的启动入口。
    public static void main(String[] args) {
        SpringApplication.run(CommunityApplication.class, args);
    }
}
```

热部署 ( IDEA 进行 Spring Boot 热部署失败的原因是，IDEA 默认情况下不会自动编译，需要对 IDEA 进行自动编译的设置)
- Settings --> Compiler
- Ctrl + Shift + Alt + / -->选择 Registry--> compiler.automake.allow.when.app.running ✔

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-devtools</artifactId>
    <scope>runtime</scope>
    <optional>true</optional>
</dependency>
```

## 基础配置

### 属性配置

Spring Boot 通过配置文件 application.properties 就可以修改默认的配置

<div align="center"><img src="img/boot/image-20211123165428245.png"></div>

将默认端口改为 80 端口。properties 格式的文件书写规范是 key=value  `如：name=itheima`

<div align="center"><img src="img/boot/image-20211123165719091.png"></div>

```properties
server.port=80
```

以前需要在 tomcat 服务器的配置文件中修改端口号，现在可以在 Spring Boot 专用的配置文件中改，<b>简化开发者配置的书写位置，集中管理。</b>

1. Spring Boot 程序可以在 application.properties 文件中进行属性配置
2. application.properties 文件中只要输入要配置的属性关键字就可以根据提示进行设置
3. Spring Boot 将配置信息集中在一个文件中写，不管你是服务器的配置，还是数据库的配置，总之都写在一起，逃离一个项目十几种配置文件格式的尴尬局面

<b>总结</b>

<b>Spring Boot 默认配置文件是 application.properties</b>

做完了端口的配置，趁热打铁，再做几个配置，目前项目启动时会显示一些日志信息，就来改一改这里面的一些设置。

<b>关闭运行日志图表（banner)</b>

```properties
spring.main.banner-mode=off
```

<b>设置运行日志的显示级别</b>

```properties
logging.level.root=debug
```

统一了配置文件的格式

我们现在配置了 3 个信息，但是又有新的问题了。这个配置是随便写的吗？什么都能配？有没有一个东西显示所有能配置的项呢？此外这个配置和什么东西有关呢？会不会因为我写了什么东西以后才可以写什么配置呢？比如我现在没有写数据库相关的东西，能否配置数据呢？一个一个来，先说第一个问题，都能配置什么。

打开 Spring Boot 的官网，找到 Spring Boot 官方文档，打开查看附录中的 Application Properties 就可以获取到对应的配置项了：https://docs.Spring.io/Spring-boot/docs/current/reference/html/application-properties.html#application-properties

能写什么的问题解决了，再来说第二个问题，这个配置项和什么有关。在 pom 中注释掉导入的 Spring-boot-starter-web，然后刷新工程，你会发现配置的提示消失了。设定使用了什么技术才能做什么配置。

<font color="#f0f"><b>温馨提示</b></font>

所有的 starter 中都会依赖下面这个 starter，叫做 Spring-boot-starter。这个 starter 是所有 Spring Boot starter 的基础依赖，里面定义了 Spring Boot 相关的基础配置，关于这个 starter 深入理解请看原理篇。

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter</artifactId>
    <version>2.5.4</version>
    <scope>compile</scope>
</dependency>
```

<b>总结</b>

1. Spring Boot 中导入对应 starter 后，提供对应配置属性
2. 书写 Spring Boot 配置采用关键字+提示形式书写

### 配置文件分类

Spring Boot 除了支持 properties 格式的配置文件，还支持另外两种格式的配置文件。三种配置文件格式分别如下:

- properties 格式
- yml 格式
- yaml 格式


从知识角度来说，要学，从开发角度来说，不用学。因为 Spring Boot 的配置在 Idea 工具下有提示，跟着提示走就行了。下面列举三种不同文件格式配置相同的属性范例，了解即可。

application.properties（properties 格式）

```properties
server.port=80
```

application.yml（yml 格式）

```YML
server:
  port: 81
```

application.yaml（yaml 格式）

```yaml
server:
  port: 82
```

yml 格式和 yaml 格式只是文件名后缀不一样，格式完全一样。<b style="color:orange">基本上都是用 yml 格式，要重点掌握。</b>

<b>总结</b>

1. Spring Boot 提供了 3 种配置文件的格式
    - properties（传统格式/默认格式）
    - <b>yml</b>（主流格式）
    - yaml

<b>思考：</b>如果三种格式的文件都写了，按什么优先级来？

#### 配置文件优先级

三个文件共存出现的情况很少，但是这个知识还是可以学习一下。我们在三个配置文件书写同样的信息，比如都配置端口，然后我们让每个文件配置的端口号都不一样，最后启动程序后看启动端口是多少就知道谁的加载优先级比较高了。

- application.properties（properties 格式）

```properties
server.port=80
```

- application.yml（yml 格式）

```YML
server:
  port: 81
```

- application.yaml（yaml 格式）

```yaml
server:
  port: 82
```

启动后发现目前的启动端口为 80，把 80 对应的文件删除掉，然后再启动，现在端口又改成了 81。现在我们就已经知道了 3 个文件的配置信息<b>使用</b>的优先顺序是什么。

```XML
application.properties  >  application.yml  >  application.yaml
```

虽然得到了一个知识结论，但是我们实际开发的时候还是要看最终的效果为准。也就是你要的最终效果是什么自己是明确的，上述结论只能帮助你分析结论产生的原因。这个知识了解一下就行了，因为以后同时写多种配置文件格式的情况实在是较少。

最后我们把配置文件内容给修改一下

- application.properties（properties 格式）

```properties
server.port=80
Spring.main.banner-mode=off
```

- application.yml（yml 格式）

```YML
server:
  port: 81
logging: 
  level: 
    root: debug
```

- application.yaml（yaml 格式）

```yaml
server:
  port: 82
```

我们发现不仅端口生效了，最终显示 80，同时其他两条配置也生效了，<b style="color:orange">看来每个配置文件中的项都会生效，只不过如果多个配置文件中有相同类型的配置会优先级高的文件覆盖优先级的文件中的配置。如果配置项不同的话，所有的配置项都会生效。</b>

<b>总结</b>

1. 配置文件间的加载优先级 properties（最高）>  yml  >  yaml（最低）
2. 不同配置文件中相同配置按照加载优先级相互覆盖，不同配置文件中不同配置全部保留 

#### 自动提示功能消失解决方案

自动提示功能不是 Spring Boot 技术给我们提供的，是我们在 Idea 工具下编程，这个编程工具给我们提供的。自动提示功能消失的原因还是蛮多的，如果想解决这个问题，就要知道为什么会消失，大体原因有如下 2 种：

1. Idea 认为你现在写配置的文件不是个配置文件，所以拒绝给你提供提示功能

2. Idea 认定你是合理的配置文件，但是 Idea 加载不到对应的提示信息

    这里我们主要解决第一个现象，第二种现象到原理篇再讲解。第一种现象的解决方式如下：

<b>步骤①</b>：打开设置，【Files】→【Project Structure...】

<div align="center"><img src="img/boot/image-20211126160548690.png"></div>

<b>步骤②</b>：在弹出窗口中左侧选择【Facets】，右侧选中 Spring 路径下对应的模块名称，也就是你自动提示功能消失的那个模块

<div align="center"><img src="img/boot/image-20211126160844372.png"></div>

<b>步骤③</b>：点击 Customize Spring Boot 按钮，此时可以看到当前模块对应的配置文件是哪些了。如果没有你想要称为配置文件的文件格式，就有可能无法弹出提示

<div align="center"><img src="img/boot/image-20211126160946448.png"></div>

<b>步骤④</b>：选择添加配置文件，然后选中要作为配置文件的具体文件就 OK 了

<div align="center"><img src="img/boot/image-20211126161145082.png" alt="image-20211126161145082" style="zoom:80%;" /></div><div align="center"><img src="img/boot/image-20211126161156324.png" alt="image-20211126161156324" style="zoom: 67%;" /></div>

到这里就做完了，其实就是 Idea 的一个小功能

<div align="center"><img src="img/boot/image-20211126161301699.png"></div>

<b>总结</b>

1. 指定 Spring Boot 配置文件

    - Setting → Project Structure → Facets
    - 选中对应项目/工程
    - Customize Spring Boot
    - 选择配置文件


### yaml文件

Spring Boot 的配置以后主要使用 yml 结尾的这种文件格式，并且在书写时可以通过提示的形式加载正确的格式。但是这种文件还是有严格的书写格式要求的。下面就来说一下具体的语法格式。

YAML（YAML Ain't Markup Language），一种数据序列化格式。具有容易阅读、容易与脚本语言交互、以数据为核心，重数据轻格式的特点。常见的文件扩展名有两种：

- .yml 格式（主流）
- .yaml 格式

> 具体的语法格式要求如下

1. 大小写敏感
2. 属性层级关系使用多行描述，<b>每行结尾使用冒号结束</b>
3. 使用缩进表示层级关系，同层级左侧对齐，只允许使用空格（不允许使用 Tab 键）
4. 属性值前面添加空格（属性名与属性值之间使用冒号+空格作为分隔）
5. #号 表示注释

上述规则不要死记硬背，按照书写习惯慢慢适应，并且在 Idea 下由于具有提示功能，慢慢适应着写格式就行了。核心的一条规则要记住，<font color="#ff0000"><b>数据前面要加空格与冒号隔开</b></font>。

> 下面列出常见的数据书写格式，熟悉一下

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

此外，yaml 格式中也可以表示数组，在属性名书写位置的下方使用减号作为数据开始符号，每行书写一个数据，减号与数据间空格分隔。

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

<b>总结</b>

1. yaml 语法规则
    - 大小写敏感
    - 属性层级关系使用多行描述，每行结尾使用冒号结束
    - 使用缩进表示层级关系，同层级左侧对齐，只允许使用空格（不允许使用Tab键）
    - 属性值前面添加空格（属性名与属性值之间使用冒号+空格作为分隔）
    - `#号`表示注释
2. 注意属性名冒号后面与数据之间有一个<b>空格</b>
3. 字面值、对象数据格式、数组数据格式

<b>思考</b>

现在我们已经知道了 yaml 具有严格的数据格式要求，并且已经可以正确的书写 yaml 文件了，那这些文件书写后其实是在定义一些数据。这些数据是给谁用的呢？大部分是 Spring Boot 框架内部使用，但是如果我们想配置一些数据自己使用，能不能用呢？答案是可以的，那如何读取 yaml 文件中的数据呢？

### yaml数据读取

对于 yaml 文件中的数据，其实你就可以想象成这就是一个小型的数据库，里面保存有若干数据，每个数据都有一个独立的名字，如果你想读取里面的数据，肯定是支持的，下面就介绍 3 种读取数据的方式。

#### 读取单一数据

<span style="color:orange">yaml 中保存的单个数据，可以使用 Spring 中的注解 @Value 读取单个数据</span>，属性名引用方式：<font color="#ff0000"><b>${一级属性名.二级属性名……}</b></font>

<div align="center"><img src="img/boot/image-20211126180433356.png" style="zoom:80%;" /></div>

记得使用 @Value 注解时，要将该注解写在某一个指定的 Spring 管控的 bean 的属性名上方，这样当 bean 进行初始化时候就可以读取到对应的单一数据了。

<b>总结</b>

1. 使用 @Value 配合 SpEL 读取单个数据
2. 如果数据存在多层级，依次书写层级名称即可

#### 读取全部数据

读取单一数据可以解决读取数据的问题，但是如果定义的数据量过大 @Value 就过于麻烦。Spring Boot 提供了一个对象，能够把所有的数据都封装到这一个对象中，这个对象叫做 Environment，使用自动装配注解可以将所有的 yaml 数据封装到这个对象中

<div align="center"><img src="img/boot/image-20211126180738569.png" style="zoom:80%;" /></div>

数据封装到了 Environment 对象中，获取属性时，通过 Environment 的接口操作进行，具体方法是 getProperties(String)，参数填写属性名即可

<b>总结</b>

1. 使用 Environment 对象封装全部配置信息
2. 使用 @Autowired 自动装配数据到 Environment 对象中

#### 读取对象数据

单一数据读取书写比较繁琐，全数据读取封装的太厉害了，每次拿数据还要一个一个的 getProperties(), 总之用起来都不是很舒服。由于 Java 是一个面向对象的语言，很多情况下，我们会将一组数据封装成一个对象。<span style="color:orange">Spring Boot 也提供了可以将一组 yaml 对象数据封装一个 Java 对象的操作。</span>

首先定义一个对象，并将该对象纳入 Spring 管控的范围，也就是定义成一个 bean，然后使用注解 @ConfigurationProperties 指定该对象加载哪一组 yaml 中配置的信息。

<div align="center"><img src="img/boot/image-20211126181126382.png" alt="image-20211126181126382" style="zoom:80%;" /></div>

<span style="color:orange">这个 @ConfigurationProperties 必须告诉他加载的数据前缀是什么，这样指定前缀下的所有属性就封装到这个对象中。</span>记得数据属性名要与对象的变量名一一对应，不然没法封装。其实以后如果你要定义一组数据自己使用，就可以先写一个对象，然后定义好属性，下面到配置中根据这个格式书写即可。

<div align="center"><img src="img/boot/image-20211126181423432.png"></div>

<font color="#f0f"><b>温馨提示</b></font>
自定义的这种数据在 yaml 文件中书写时没有弹出提示，在原理篇再揭秘如何弹出提示。

<b>总结</b>

1. 使用 @ConfigurationProperties 注解绑定配置信息到封装类中
2. 封装类需要定义为 Spring 管理的 bean，否则无法进行属性注入

#### yaml文件中的数据引用

如果你在书写 yaml 数据时，经常出现如下现象，比如很多个文件都具有相同的目录前缀

```YAML
center:
	dataDir: /usr/local/fire/data
    tmpDir: /usr/local/fire/tmp
    logDir: /usr/local/fire/log
    msgDir: /usr/local/fire/msgDir
```

或者

```YAML
center:
	dataDir: D:/usr/local/fire/data
    tmpDir: D:/usr/local/fire/tmp
    logDir: D:/usr/local/fire/log
    msgDir: D:/usr/local/fire/msgDir
```

这个时候你可以使用引用格式来定义数据，其实就是搞了个变量名，然后引用变量了，格式如下：

```YAML
baseDir: /usr/local/fire
center:
    dataDir: ${baseDir}/data
    tmpDir: ${baseDir}/tmp
    logDir: ${baseDir}/log
    msgDir: ${baseDir}/msgDir
```

还有一个注意事项，在书写字符串时，如果需要使用转义字符，需要将数据字符串使用双引号包裹起来

```YAML
lesson: "Spring\tboot\nlesson"
```

<b>总结</b>

1. 在配置文件中可以使用 `${属性名}` 方式引用属性值
2. 如果属性中出现特殊字符，可以使用双引号包裹起来作为字符解析

### 配置信息的加载顺序

- 先加载 yml
- 再加载 yaml
- 再加载 properties
- <span style="color:red">后加载的覆盖先加载的哦~</span>

```xml
<resource>
    <directory>${basedir}/src/main/resources</directory>
    <filtering>true</filtering>
    <includes>
        <include><b>/application*.yml</include>
        <include><b>/application*.yaml</include>
        <include><b>/application*.properties</include>
    </includes>
</resource>
```

# 基本原理

## 概述

Spring Boot 父项目做依赖管理，配置了常用的三方 jar 的版本。

Spring Boot 注解的层次关系如下，以指定启动类注解 `@SpringBootApplication` 为例：

`@SpringBootApplication` 上有这些注解

- `@SpringBootConfiguration`
    - @Configuration, 不就是 Spring 的那个注解吗。加上这个注解就是指定它是一个配置类
- `@EnableAutoConfiguration`，开启自动配置
    - `@AutoConfigurationPackage`
    - `@Import(AutoConfigurationImportSelector.class)`，Import 的作用是当前配置文件引入其他配置类
    - `AutoConfigurationImportSelector` 这个类很重要。单独拎出来讲！
- `@ComponentScan`，组件扫描。有这个注解的，会以该注解标注的类所在的包为基础路径，进行类的扫描。`@SpringBootApplication` 类上打了 `@ComponentScan` 注解，相当于 `@SpringBootApplication` 也有这个扫描的功能。

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited // 该注解表明注解是否会被子类继承，缺省情况是不继承的。 当注解在声明时，使用了@Inherited注解，则该注解会被使用了该注解的类的子类所继承。
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(excludeFilters = { @Filter(type = FilterType.CUSTOM, classes = TypeExcludeFilter.class),
      @Filter(type = FilterType.CUSTOM, classes = AutoConfigurationExcludeFilter.class) })
public @interface SpringBootApplication {}
```

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@AutoConfigurationPackage
@Import(AutoConfigurationImportSelector.class)
public @interface EnableAutoConfiguration {}
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
        // META-INF/Spring.factories  一般这个META-INF是当前类所在的那个jar包的META-INF下
        // 我们去看META-INF下的这个类org.Springframework.boot.autoconfigure.web.servlet.ServletWebServerFactoryAutoConfiguration,\
        // 找这个类的注解 @EnableConfigurationProperties(ServerProperties.class)
        // 点进这个类 ServerProperties
        // 定位到 @ConfigurationProperties(prefix = "server", ignoreUnknownFields = true)
        // public class ServerProperties { }
        // 结论 去 Spring-configuration-metadata.json里找配置信息！
        Assert.notEmpty(configurations, "No auto configuration classes found in META-INF/Spring.factories. If you "
            + "are using a custom packaging, make sure that file is correct.");
      return configurations;
   }
}
```

## 原理

> 依赖管理

父项目做依赖管理

```xml
<!-- 依赖管理 -->    
<parent>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-parent</artifactId>
    <version>2.3.4.RELEASE</version>
</parent>

<!-- 他的父项目 -->   
<parent>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-dependencies</artifactId>
    <version>2.3.4.RELEASE</version>
</parent>

<!-- 几乎声明了所有开发中常用的依赖的版本号,自动版本仲裁机制 -->   
```

开发导入 starter 场景启动器

- 见到很多 Spring-boot-starter-* ： *就某种场景
- 只要引入 starter，这个场景的所有常规需要的依赖 Spring Boot 都会自动引入
- Spring Boot 所有支持的场景
    https://docs.Spring.io/Spring-boot/docs/current/reference/html/using-Spring-boot.html#using-boot-starter
- 见到的  *-Spring-boot-starter：第三方为我们提供的简化开发的场景启动器。
- 所有场景启动器最底层的依赖

```xml
<dependency>
  <groupId>org.Springframework.boot</groupId>
  <artifactId>Spring-boot-starter</artifactId>
  <version>2.3.4.RELEASE</version>
  <scope>compile</scope>
</dependency>
```

无需关注版本号，自动版本仲裁

- 引入依赖默认都可以不写版本
- 引入非版本仲裁的 jar，要写版本号。

可以修改默认版本号

- 查看 Spring-boot-dependencies 里面规定当前依赖的版本用的 key。
- 在当前项目里面重写配置

```xml
<properties>
	<mysql.version>5.1.43</mysql.version>
</properties>
```

### 自动配置流程

1️⃣Spring Boot 启动的时候加载主配置类，开启了自动配置功能 @EnableAutoConfifiguration

2️⃣<b>@EnableAutoConfifiguration 的作用：</b>

- 利用 EnableAutoConfifigurationImportSelector 给容器中导入一些组件？

- 可以查看 selectImports() 方法的内容；

- List confifigurations = getCandidateConfifigurations(annotationMetadata, attributes); 获取候选的配置

    ```shell
    SpringFactoriesLoader.loadFactoryNames() 
    扫描所有jar包类路径下 META-INF/Spring.factories
    把扫描到的这些文件的内容包装成properties对象 
    从properties中获取到EnableAutoConfiguration.class类（类名）对应的值，然后 把他们添加在容器中
    ```

<b>将类路径下 META-INF/Spring.factories 里面配置的所有 EnableAutoConfifiguration 的值加入到了容器中；</b>

```shell
# Auto Configure 
org.Springframework.boot.autoconfigure.EnableAutoConfiguration=\ 
org.Springframework.boot.autoconfigure.admin.SpringApplicationAdminJmxAutoCo nfiguration,\ 
org.Springframework.boot.autoconfigure.aop.AopAutoConfiguration,\ 
org.Springframework.boot.autoconfigure.amqp.RabbitAutoConfiguration,\
org.Springframework.boot.autoconfigure.batch.BatchAutoConfiguration,\
org.Springframework.boot.autoconfigure.cache.CacheAutoConfiguration,\
```

每一个这样的 xxxAutoConfifiguration 类都是容器中的一个组件，都加入到容器中；用他们来做自动配置； 

3️⃣每一个自动配置类进行自动配置功能；

4️⃣以 <b>HttpEncodingAutoConfifiguration</b> (HTTP 编码自动配置）为例解释自动配置原理；

```java
// 表示这是一个配置类，以前编写的配置文件一样，也可以给容器中添加组件 
@AutoConfiguration 

// 启动指定类的 ConfigurationProperties 功能；
// 将配置文件中对应的值和 HttpEncodingProperties 绑定起来；
// 并把 HttpEncodingProperties 加入到 ioc 容器中
@EnableConfigurationProperties(HttpEncodingProperties.class) 

// Spring 底层 @Conditional 注解（Spring 注解版），
// 根据不同的条件，如果满足指定的条件，整个配置类里面的配置就会生效； 
// 判断当前应用是否是 web 应用，如果是，当前配置类生效
@ConditionalOnWebApplication

// 判断当前项目有没有这个类 CharacterEncodingFilter；SpringMVC 中进行乱码解决的过滤器；
@ConditionalOnClass(CharacterEncodingFilter.class)

// 判断配置文件中是否存在某个配置 Spring.http.encoding.enabled；
// 如果不存在，判断也是成立的 
// 即使我们配置文件中不配置 Spring.http.encoding.enabled=true，也是默认生效的；
@ConditionalOnProperty(prefix = "Spring.http.encoding", value = "enabled", matchIfMissing = true)
public class HttpEncodingAutoConfiguration {
    
    // 它已经和 Spring Boot 的配置文件映射了
    private final Encoding properties;

    // 只有一个有参构造的情况下，参数的值就会从容器中拿
    public HttpEncodingAutoConfiguration(ServerProperties properties) {
        this.properties = properties.getServlet().getEncoding();
    }

    @Bean //给容器中添加一个组件，这个组件的某些值需要从properties中获取
    @ConditionalOnMissingBean //判断容器没有这个组件？
    public CharacterEncodingFilter characterEncodingFilter() {
        CharacterEncodingFilter filter = new OrderedCharacterEncodingFilter();
        filter.setEncoding(this.properties.getCharset().name());
        filter.setForceRequestEncoding(this.properties.shouldForce(org.Springframework.boot.web.servlet.server.Encoding.Type.REQUEST));
        filter.setForceResponseEncoding(this.properties.shouldForce(org.Springframework.boot.web.servlet.server.Encoding.Type.RESPONSE));
        return filter;
    }

    @Bean
    public HttpEncodingAutoConfiguration.LocaleCharsetMappingsCustomizer localeCharsetMappingsCustomizer() {
        return new HttpEncodingAutoConfiguration.LocaleCharsetMappingsCustomizer(this.properties);
    }

    static class LocaleCharsetMappingsCustomizer implements WebServerFactoryCustomizer<ConfigurableServletWebServerFactory>, Ordered {
        private final Encoding properties;

        LocaleCharsetMappingsCustomizer(Encoding properties) {
            this.properties = properties;
        }

        public void customize(ConfigurableServletWebServerFactory factory) {
            if (this.properties.getMapping() != null) {
                factory.setLocaleCharsetMappings(this.properties.getMapping());
            }
        }

        public int getOrder() {
            return 0;
        }
    }
}
```

根据当前不同的条件判断，决定这个配置类是否生效。一但这个配置类生效；这个配置类就会给容器中添加各种组件；这些组件的属性是从对应的 properties 类中获取的，这些类里面的每一个属性又是和配置文件绑定的；

5️⃣<b style="color:red">所有在配置文件中能配置的属性都封装在 xxxxProperties 类中；配置文件能配置什么就可以参照某个功能对应的属性类</b>

```java
@ConfigurationProperties(prefix = "Spring.http.encoding") 
//从配置文件中获取指 定的值和bean的属性进行绑定 
public class HttpEncodingProperties { 
    public static final Charset DEFAULT_CHARSET = Charset.forName("UTF-8");
}
```

<b>精髓：</b>

- Spring Boot 启动会加载大量的自动配置类
- 我们看我们需要的功能有没有 Spring Boot 默认写好的自动配置类；
- 我们再来看这个自动配置类中到底配置了哪些组件；（只要我们要用的组件有，我们就不需要再来配置了）
- 给容器中自动配置类添加组件的时候，会从 properties 类中获取某些属性。我们就可以在配置

xxxxAutoConfifigurartion：自动配置类；

给容器中添加组件

xxxxProperties：封装配置文件中相关属性；

> 细节

<b>@Conditional 派生注解（Spring 注解版原生的 @Conditional 作用）</b>

作用：必须是 @Conditional 指定的条件成立，才给容器中添加组件，配置配里面的所有内容才生效；

| @Conditional扩展注解            | 作用（判断是否满足当前指定条件）                |
| ------------------------------- | ----------------------------------------------- |
| @ConditionalOnJava              | 系统的 Java 版本是否符合要求                    |
| @ConditionalOnBean              | 容器中存在指定 Bean                             |
| @ConditionalOnMissingBean       | 容器中不存在指定 Bean；                         |
| @ConditionalOnExpression        | 满足 SpEL 表达式指定                            |
| @ConditionalOnClass             | 系统中有指定的类                                |
| @ConditionalOnMissingClass      | 系统中没有指定的类                              |
| @ConditionalOnSingleCandidate   | 容器中只有一个指定的 Bean，或者这个 Bean 是首选 |
| @ConditionalOnProperty          | Bean 系统中指定的属性是否有指定的值             |
| @ConditionalOnResource          | 类路径下是否存在指定资源文件                    |
| @ConditionalOnWebApplication    | 当前是 web 环境                                 |
| @ConditionalOnNotWebApplication | 当前不是 web 环境                               |
| @ConditionalOnJndi              | JNDI 存在指定项                                 |

<b>自动配置类必须在一定的条件下才能生效；</b>我们怎么知道哪些自动配置类生效；

<b>可以通过启用 debug=true 属性</b>；来让控制台打印自动配置报告，这样我们就可以很方便的知道哪些自动配置类生效；

```
========================= 
AUTO-CONFIGURATION REPORT 
=========================

Positive matches:（自动配置类启用的）
------------------------------------
	DispatcherServletAutoConfiguration matched:
	- @ConditionalOnClass found required class 'org.Springframework.web.servlet.DispatcherServlet';

Negative matches:（没有启动，没有匹配成功的自动配置类）
------------------------------------
	ActiveMQAutoConfiguration:
		Did not match:
			- @ConditionalOnClass did not find required classes
'javax.jms.ConnectionFactory',
'org.apache.activemq.ActiveMQConnectionFactory' (OnClassCondition)
```

### 自动配置原理

[Spring Boot Reference Guide](https://docs.Spring.io/Spring-boot/docs/1.5.9.RELEASE/reference/htmlsingle/#common-application-properties)

#### 引导加载自动配置类

- @SpringBootConfiguration 表示是一个配置类
- @ComponentScan 指定扫描哪些，Spring 注解；
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
public @interface EnableAutoConfiguration {}
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

<div align="center"><img src="img/boot/AutoConfigurationImportSelector.png"></div>

```java
protected List<String> getCandidateConfigurations(AnnotationMetadata metadata, AnnotationAttributes attributes) {
    // 利用工厂加载，得到所有加载。loadFactoryNames 调用了 loadSpringFactories 方法
    List<String> configurations = SpringFactoriesLoader.loadFactoryNames(getSpringFactoriesLoaderFactoryClass(),
                                                                         getBeanClassLoader());
    Assert.notEmpty(configurations, "No auto configuration classes found in META-INF/Spring.factories. If you "
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
        // 从 FACTORIES_RESOURCE_LOCATION=META-INF/Spring.factories 加载数据
        // 默认扫描我们当前系统里面所有 META-INF/Spring.factories 位置的文件
        // 文件里写死了，Spring-boot一启动就要给容器中加载的所有配置类
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
@ConditionalOnProperty(prefix = "Spring.aop", name = "auto", havingValue = "true", matchIfMissing = true)
public class AopAutoConfiguration {

	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass(Advice.class)
	// some code
}
```

#### 修改默认配置

配置文件中是否存在 Spring.aop 的配置，如果存在 Spring.aop.auto，且值为 true，就失效。即便没有配，也认为配置了。

```java
@Configuration(proxyBeanMethods = false)
@ConditionalOnProperty(prefix = "Spring.aop", name = "auto", havingValue = "true", matchIfMissing = true)
public class AopAutoConfiguration {
    	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass(Advice.class) // 存在 Advice.class 才失效
	static class AspectJAutoProxyingConfiguration {}

	@Configuration(proxyBeanMethods = false)
	@ConditionalOnMissingClass("org.aspectj.weaver.Advice") // 系统里没有这个类才生效。
	@ConditionalOnProperty(prefix = "Spring.aop", name = "proxy-target-class", havingValue = "true",
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

Spring Boot 默认会在底层配好所有的组件。但是如果用户自己配置了以用户的优先

```java
@Bean
@ConditionalOnMissingBean
public CharacterEncodingFilter characterEncodingFilter() {
}
```

#### 总结

- Spring Boot 先加载所有的自动配置类  xxxxxAutoConfiguration
- 每个自动配置类按照条件进行生效，默认都会绑定配置文件指定的值。xxxxProperties 里面拿。xxxProperties 和配置文件进行了绑定

- 生效的配置类就会给容器中装配很多组件
- 只要容器中有这些组件，相当于这些功能就有了
- 定制化配置
    - 用户直接自己 @Bean 替换底层的组件
    - 用户去看这个组件是获取的配置文件什么值就去修改。

<b>xxxxxAutoConfiguration ---> 把组件装配进去  ---></b> <b>组件从 xxxxProperties 里面拿值 ----> application.properties</b>

## 自动配置举例

- 自动配好 Tomcat

    - 引入 Tomcat 依赖。
    - 配置 Tomcat

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-tomcat</artifactId>
    <version>2.3.4.RELEASE</version>
    <scope>compile</scope>
</dependency>
```

- 自动配好 Spring MVC

    - 引入 Spring MVC 全套组件
    - 自动配好 Spring MVC 常用组件（功能）

- 自动配好 Web 常见功能，如：字符编码问题

    - Spring Boot 帮我们配置好了所有 Web 开发的常见场景

- 默认的包结构

    - 主程序所在包及其下面的所有子包里面的组件都会被默认扫描进来
    - 无需以前的包扫描配置
    - 想要改变扫描路径，@SpringBootApplication(scanBasePackages=<b>"com.xx"</b>)
    - 或者 @ComponentScan 指定扫描路径

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

- 非常多的 starter
- 引入了哪些场景这个场景的自动配置才会开启
- Spring Boot 所有的自动配置功能都在 Spring-boot-autoconfigure 包里面
- ......

## 开发小技巧

### Lombok

简化 JavaBean 开发

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
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-devtools</artifactId>
    <optional>true</optional>
</dependency>
```

项目或者页面修改以后：Ctrl+F9；

### 配置问题提示

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-configuration-processor</artifactId>
    <optional>true</optional>
</dependency>
```

# Web 开发基础

<div align="center"><img src="img/boot/yuque_diagram.jpg"></div>

## 自动配置概览

Spring Boot provides auto-configuration for Spring MVC that <b>works well with most applications.(大多场景我们都无需自定义配置)</b>

The auto-configuration adds the following features on top of Spring’s defaults:

- Inclusion of `ContentNegotiatingViewResolver` and `BeanNameViewResolver` beans.
    - 内容协商视图解析器和 BeanName 视图解析器
    - 自动配置了 ViewResolver（视图解析器：根据方法的返回值得到视图对象（View），视图对象决定如何渲染（转发、重定向））
    - ContentNegotiatingViewResolver：组合所有的视图解析器的；
    - 如何定制：我们可以自己给容器中添加一个视图解析器；自动的将其组合进来；
- Support for serving static resources, including support for WebJars (covered [later in this document](https://docs.Spring.io/Spring-boot/docs/current/reference/html/Spring-boot-features.html#boot-features-Spring-mvc-static-content))).【静态资源（包括 webjars）】
- Automatic registration of `Converter`, `GenericConverter`, and `Formatter` beans.
    - 自动注册 `Converter，GenericConverter，Formatter `
    - Converter：转换器； public String hello(User user)：类型转换使用Converter
    - Formatter 格式化器； 2017.12.17===Date；
- Support for `HttpMessageConverters` (covered [later in this document](https://docs.Spring.io/Spring-boot/docs/current/reference/html/Spring-boot-features.html#boot-features-Spring-mvc-message-converters)).【支持 `HttpMessageConverters` （后来我们配合内容协商理解原理）】
- Automatic registration of `MessageCodesResolver` (covered [later in this document](https://docs.Spring.io/Spring-boot/docs/current/reference/html/Spring-boot-features.html#boot-features-Spring-message-codes)).【自动注册 `MessageCodesResolver` （国际化用）】
- Static `index.html` support.【静态 index.html 页支持】
- Custom `Favicon` support (covered [later in this document](https://docs.Spring.io/Spring-boot/docs/current/reference/html/Spring-boot-features.html#boot-features-Spring-mvc-favicon)).【自定义 `Favicon`  】
- Automatic use of a `ConfigurableWebBindingInitializer` bean (covered [later in this document](https://docs.Spring.io/Spring-boot/docs/current/reference/html/Spring-boot-features.html#boot-features-Spring-mvc-web-binding-initializer)).【自动使用 `ConfigurableWebBindingInitializer` ，（DataBinder 负责将请求数据绑定到 JavaBean 上）】

> If you want to keep those Spring Boot MVC customizations and make more [MVC customizations](https://docs.Spring.io/Spring/docs/5.2.9.RELEASE/Spring-framework-reference/web.html#mvc) (interceptors, formatters, view controllers, and other features), you can add your own `@Configuration` class of type `WebMvcConfigurer` but <b>without</b> `@EnableWebMvc`.
>
> <b>不用@EnableWebMvc注解。使用</b> `@Configuration` <b>+</b> `WebMvcConfigurer` <b>自定义规则</b>

> If you want to provide custom instances of `RequestMappingHandlerMapping`, `RequestMappingHandlerAdapter`, or `ExceptionHandlerExceptionResolver`, and still keep the Spring Boot MVC customizations, you can declare a bean of type `WebMvcRegistrations` and use it to provide custom instances of those components.
>
> <b>声明</b> `WebMvcRegistrations` <b>改变默认底层组件</b>

> If you want to take complete control of Spring MVC, you can add your own `@Configuration` annotated with `@EnableWebMvc`, or alternatively add your own `@Configuration`-annotated `DelegatingWebMvcConfiguration` as described in the Javadoc of `@EnableWebMvc`.
>
> <b>使用</b> `@EnableWebMvc+@Configuration+DelegatingWebMvcConfiguration 全面接管 SpringMVC`

## 简单功能分析

### 静态资源访问

#### 静态资源目录

默认情况下，Spring Boot 下的静态资源都统一放在类路径下：名为 /static (or /public or /resources or /META-INF/resources）目录下。

访问 url 为：<span style="color:green">当前项目根路径/ + 静态资源名</span>

> 原理： 静态映射/.

请求进来，先去找 Controller 看能不能处理。不能处理的所有请求又都交给静态资源处理器。静态资源也找不到则响应 404 页面

> 改变默认的静态资源路径

```yaml
Spring:
  mvc:
    static-path-pattern: /res/** # 改变资源的 url 映射路径，从 项目根目录/资源名 ==> 项目根目录/res/资源名
  resources:
    static-locations: [classpath:/haha/]
```

#### 静态资源访问前缀

默认无前缀，为其增加前缀后可以方便的拦截静态资源了（资源的拦截是先匹配 Controller、Controller 匹配不到再匹配静态资源）

```yaml
Spring:
  mvc:
    static-path-pattern: /res/**
```

<span style="color:green">当前项目 + static-path-pattern + 静态资源名 = 静态资源文件夹下找（只是修改了下 url 的访问规则，资源的存放位置并没有改变）</span>

#### 改变默认静态资源

```yaml
Spring:
	resources:
		static-locations: classpath:/haha/ # 2.4.x 无效了。
```

### 欢迎页

Spring Boot supports both static and templated welcome pages. It first looks for an <b>index.html file in the configured static content locations.</b> If one is not found, it then looks for an index template. If either is found, it is automatically used as the welcome page of the application.

- 静态资源路径下 index.html
    - 可以配置静态资源路径
    - 但是不可以配置静态资源的访问前缀。否则导致 index.html 不能被默认访问

```yaml
Spring:
#  mvc:
#    static-path-pattern: /res/*   这个会导致welcome page功能失效
  resources:
    static-locations: [classpath:/haha/]
```

## 静态资源配置原理

- Spring Boot 启动默认加载  xxxAutoConfiguration 类（自动配置类）
- Spring MVC 功能的自动配置类 WebMvcAutoConfiguration，生效

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
			org.Springframework.boot.autoconfigure.web.ResourceProperties.class, WebProperties.class })
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
        org.Springframework.boot.autoconfigure.web.ResourceProperties resourceProperties,
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
    - WebMvcProperties 和 Spring.mvc 前缀相关的配置文件进行匹配。
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

#### Rest 使用与原理

- @xxxMapping；
- Rest 风格支持（使用 <b>HTTP</b> 请求方式动词来表示对资源的操作）
    - 以前：/getUser  获取用户    /deleteUser 删除用户   /editUser  修改用户   /saveUser  保存用户
    - 现在： /user  GET-获取用户    DELETE-删除用户    PUT-修改用户  POST-保存用户
    - 核心 Filter；HiddenHttpMethodFilter
        - 用法： 表单 method=post，隐藏域 _method=put
        - Spring Boot 中手动开启，`Spring.mvc.hiddenmethod.filter.enabled=true`
    - 扩展：如何把 _method 这个名字换成我们自己喜欢的

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
@ConditionalOnProperty(prefix = "Spring.mvc.hiddenmethod.filter", name = "enabled")
public OrderedHiddenHttpMethodFilter hiddenHttpMethodFilter() {
    return new OrderedHiddenHttpMethodFilter();
}
```

Rest 原理（表单提交需要使用 Rest 的时候，debug 看 HiddenHttpMethodFilter 的 doFilter 方法；如果是自己实现，也可以考虑用 Filter 对请求进行拦截，判断，然后选择对应的 Controller 方法）

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

<div align="center"><img src="img/boot/image-20211023111242760.png"></div>

总结：Spring MVC 功能分析都从 org.Springframework.web.servlet.DispatcherServlet ---> doDispatch() 开始

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

getHandler 方法；RequestMappingHandlerMapping 中存储了所有 @RequestMapping 和 handler 的映射规则。

<div align="center">
    <img src="img/boot/image-20211023114049369.png">
    <br>
	<img src="img/boot/image-20211023114945974.png">
</div>


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

所有的请求映射都在 HandlerMapping 中。

- Spring Boot 自动配置欢迎页的 WelcomePageHandlerMapping 。访问 /能访问到 index.html；
- Spring Boot 自动配置了默认的 RequestMappingHandlerMapping

- 请求进来，挨个尝试所有的 HandlerMapping 看是否有请求信息。

- - 如果有就找到这个请求对应的 handler
  - 如果没有就是下一个 HandlerMapping

- 我们需要一些自定义的映射处理，我们也可以自己给容器中放 <b>HandlerMapping</b>。自定义 <b>HandlerMapping</b>

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
    //2、Spring Boot默认是禁用了矩阵变量的功能
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

### 参数处理原理

#### HandlerAdapter

- HandlerMapping 中找到能处理请求的 Handler（Controller#method）
- 为当前 Handler 找一个适配器 HandlerAdapter
    - HandlerAdapter 是 Spring 设计的一个接口，
        - supports 支持那些方法
        - handle 定义了如何处理的逻辑

<div align="center"><img src="img/boot/image-20211024200517771-16350771187701.png"></div>

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
protected ModelAndView handleInternal(HttpServletRequest request, HttpServletResponse response, HandlerMethod handlerMethod) throws Exception {

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

Spring MVC 目标方法能写多少种参数类型。取决于参数解析器。

<div align="center"><img src="img/boot/image-20211024201053219.png"></div>

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

<div align="center"><img src="img/boot/image-20211024201539502.png"></div>

详细内容参考 Spring MVC 相关笔记。

## 扩展 MVC

编写一个配置类（@Configuration），是 WebMvcConfigurerAdapter 类型；注意！不能标注 <b>@EnableWebMvc</b>;

既保留了所有的自动配置，也能用我们扩展的配置；

```java
// 使用WebMvcConfigurerAdapter可以来扩展SpringMVC的功能
// WebMvcAutoConfiguration是SpringMVC的自动配置类
@Configuration 
public class MyMvcConfig extends WebMvcConfigurerAdapter {
    @Override 
    public void addViewControllers(ViewControllerRegistry registry) { 
        // super.addViewControllers(registry); 
        //浏览器发送 /xx 请求来到 success 
        registry.addViewController("/atguigu").setViewName("success"); 
    }
}
```

> 全面接管 Spring MVC

Spring Boot 对 Spring MVC 的自动配置不需要了，所有都是我们自己配置；所有的 Spring MVC 的自动配置都失效了。

<span style="color:orange">我们需要在配置类中添加 @EnableWebMvc 即可；</span>

```java
@Configuration 
@EnableWebMvc
public class MyMvcConfig extends WebMvcConfigurerAdapter {
    @Override 
    public void addViewControllers(ViewControllerRegistry registry) { 
        // super.addViewControllers(registry); 
        //浏览器发送 /xx 请求来到 success 
        registry.addViewController("/atguigu").setViewName("success"); 
    }
}
```

为什么加了 @EnableWebMvc 注解自动配置就失效了？

```java
// @EnableWebMvc 
@Import(DelegatingWebMvcConfiguration.class) 
public @interface EnableWebMvc {}
// 而 DelegatingWebMvcConfiguration 继承了 WebMvcConfigurationSupport

@Configuration 
@ConditionalOnWebApplication 
@ConditionalOnClass({ Servlet.class, DispatcherServlet.class, 
                     WebMvcConfigurerAdapter.class }) 
//容器中没有这个组件的时候，这个自动配置类才生效 
@ConditionalOnMissingBean(WebMvcConfigurationSupport.class) 
@AutoConfigureOrder(Ordered.HIGHEST_PRECEDENCE + 10) 
@AutoConfigureAfter({ DispatcherServletAutoConfiguration.class, 
                     ValidationAutoConfiguration.class }) 
public class WebMvcAutoConfiguration {}
```

## 错误处理机制

Spring Boot 有一个默认的错误页面。

### 错误处理概述

<div align="center"><img src="img/boot/image-20220702190541159.png"></div>

如果是其他客户端，会响应一个 json 数据。

错误处理机制的原理可以参考 ErrorMvcAutoConfiguration（错误处理的自动配置）

### 定制错误响应

1️⃣如果有模板引擎，将错误页面命名为<b>”错误状态码.html“</b>放在模板引擎文件夹里面的  error 文件夹下】，发生此状态码的错误就会来到对应的页面；

我们可以使用 4xx 和 5xx 作为错误页面的文件名来匹配这种类型的所有错误，精确优先（优先寻找精确的状态码.html）；

页面能获取的信息如下；

timestamp：时间戳

status：状态码

error：错误提示

exception：异常对象

message：异常消息

errors：JSR303数据校验的错误都在这里

2️⃣没有模板引擎（模板引擎找不到这个错误页面），静态资源文件夹下找

3️⃣以上都没有错误页面，就是默认来到 Spring Boot 默认的错误提示页面；

> 错误页面代码

目录结构

```
src/
 +- main/
     +- java/
     |   + <source code>
     +- resources/
         +- public/
             +- error/
             |   +- 404.html
             +- <other public assets>
```

在 error/ 下放置页面代码即可。

> 也可以使用 Spring MVC 特点的错误配置页面

@ControllerAdvice 标注在类上，表明这个是用来处理错误的类。

@ExceptionHandler(错误.class) 标注在方法上，表明这个方法处理什么类型异常。

```java
// 自定义异常处理 & 返回定制 json 数据；
@ControllerAdvice
public class MyExceptionHandler{
    
    @ResponseBody
    @ExceptionHandler(UserNotFoundException.class)
    public Map<String,Object> handleException(Exception e){
        Map<String,Object> map = new HashMap<>(); 
        map.put("code","user.notexist"); 
        map.put("message",e.getMessage()); 
        return map;
    }
    
    // 转发到 /error 进行自适应响应效果处理
    public String handleException(Exception e, HttpServletRequest request){
        Map<String,Object> map = new HashMap<>();
        //传入我们自己的错误状态码 4xx 5xx，否则就不会进入定制错误页面的解析流程
        request.setAttribute("javax.servlet.error.status_code",500);
        map.put("code","user.notexist"); map.put("message",e.getMessage()); 
        //转发到/error 
        return "forward:/error"; } 
    }
}
```

[(11条消息) Spring Boot错误页配置及项目全局错误处理_逆风GLORY的博客-CSDN博客](https://blog.csdn.net/csdn_0911/article/details/82794426)

## 嵌入式 Servlet 容器

Spring Boot 默认使用 Tomcat 作为嵌入式的 Servlet 容器；

### 修改 Servlet 容器配置

可以直接通过配置文件进行修改：

```properties
server.port=8081 
server.context-path=/crud 
server.tomcat.uri-encoding=UTF-8 
# 通用的Servlet容器设置 
server.xxx 
# Tomcat的设置 
server.tomcat.xxx
```

也可以通过编写一个 EmbeddedServletContainerCustomizer：嵌入式的 Servlet 容器的定制器；来修改 Servlet 容器的配置

```java
@Configuration
public class XXConfig{
    @Bean 
    //一定要将这个定制器加入到容器中 
    public EmbeddedServletContainerCustomizer embeddedServletContainerCustomizer(){ 
        return new EmbeddedServletContainerCustomizer() { 
            //定制嵌入式的Servlet容器相关的规则 
            @Override public void customize(ConfigurableEmbeddedServletContainer container) {
                container.setPort(8083); 
            } 
        }; 
    }
}
```

### 注册 Servlet 三大组件

由于 Spring Boot 默认是以 jar 包的方式启动嵌入式的 Servlet 容器来启动 Spring Boot 的 web 应用，没有 web.xml 文件。因此注册三大组件用以下方式

ServletRegistrationBean

```java
@Bean 
public ServletRegistrationBean myServlet(){ 
    ServletRegistrationBean registrationBean = new ServletRegistrationBean(new MyServlet(),"/myServlet"); 
    return registrationBean; 
}

@Bean 
public FilterRegistrationBean myFilter(){ 
    FilterRegistrationBean registrationBean = new FilterRegistrationBean(); 
    registrationBean.setFilter(new MyFilter()); 
    registrationBean.setUrlPatterns(Arrays.asList("/hello","/myServlet")); 
    return registrationBean; 
}

@Bean 
public ServletListenerRegistrationBean myListener(){ 
    ServletListenerRegistrationBean<MyListener> registrationBean = new ServletListenerRegistrationBean<>(new MyListener()); 
    return registrationBean; 
}
```

Spring Boot 帮我们启动 Spring MVC 的时候，会自动注册 Spring MVC 的前端控制器 DispatcherServlet； 

```java
// DispatcherServletAutoConfiguration中注册前端控制器的代码

@Configuration(proxyBeanMethods = false)
@Conditional(DispatcherServletRegistrationCondition.class)
@ConditionalOnClass(ServletRegistration.class)
@EnableConfigurationProperties(WebMvcProperties.class)
@Import(DispatcherServletConfiguration.class)
protected static class DispatcherServletRegistrationConfiguration {
    @Bean(name = DEFAULT_DISPATCHER_SERVLET_REGISTRATION_BEAN_NAME)
    @ConditionalOnBean(value = DispatcherServlet.class, name = DEFAULT_DISPATCHER_SERVLET_BEAN_NAME)
    public DispatcherServletRegistrationBean dispatcherServletRegistration(DispatcherServlet dispatcherServlet,
                                                                           WebMvcProperties webMvcProperties, ObjectProvider<MultipartConfigElement> multipartConfig) {
        DispatcherServletRegistrationBean registration = new DispatcherServletRegistrationBean(dispatcherServlet,
                                                                                               webMvcProperties.getServlet().getPath());
        registration.setName(DEFAULT_DISPATCHER_SERVLET_BEAN_NAME);
        registration.setLoadOnStartup(webMvcProperties.getServlet().getLoadOnStartup());
        multipartConfig.ifAvailable(registration::setMultipartConfig);
        return registration;
    }
}
```

Spring Boot 也可以用其他 Servlet 容器

### 嵌入式容器

默认支持：Tomcat（默认使用）、Jetty、Undertow；使用 Jetty、Undertow 的时候需要先排除 Tomcat。

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-web</artifactId>
    <exclusions>
        <exclusion>
            <artifactId>Spring-boot-starter-tomcat</artifactId>
            <groupId>org.Springframework.boot</groupId>
        </exclusion>
    </exclusions>
    
</dependency> <!--引入其他的Servlet容器-->
<dependency>
    <artifactId>Spring-boot-starter-jetty</artifactId>
    <groupId>org.Springframework.boot</groupId>
</dependency>
```

嵌入式容器的原理

```java
@AutoConfiguration
@ConditionalOnWebApplication
@EnableConfigurationProperties(ServerProperties.class)
public class EmbeddedWebServerFactoryCustomizerAutoConfiguration {

	/**
	 * Nested configuration if Tomcat is being used.
	 */
	@Configuration(proxyBeanMethods = false)
    // 如果有 Tomcat 依赖，就启用 Tomcat 容器
	@ConditionalOnClass({ Tomcat.class, UpgradeProtocol.class })
	public static class TomcatWebServerFactoryCustomizerConfiguration {

		@Bean
		public TomcatWebServerFactoryCustomizer tomcatWebServerFactoryCustomizer(Environment environment,
				ServerProperties serverProperties) {
			return new TomcatWebServerFactoryCustomizer(environment, serverProperties);
		}
	}

	/**
	 * Nested configuration if Jetty is being used.
	 */
	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass({ Server.class, Loader.class, WebAppContext.class })
	public static class JettyWebServerFactoryCustomizerConfiguration {

		@Bean
		public JettyWebServerFactoryCustomizer jettyWebServerFactoryCustomizer(Environment environment,
				ServerProperties serverProperties) {
			return new JettyWebServerFactoryCustomizer(environment, serverProperties);
		}
	}

	/**
	 * Nested configuration if Undertow is being used.
	 */
	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass({ Undertow.class, SslClientAuthMode.class })
	public static class UndertowWebServerFactoryCustomizerConfiguration {

		@Bean
		public UndertowWebServerFactoryCustomizer undertowWebServerFactoryCustomizer(Environment environment,
				ServerProperties serverProperties) {
			return new UndertowWebServerFactoryCustomizer(environment, serverProperties);
		}
	}

	/**
	 * Nested configuration if Netty is being used.
	 */
	@Configuration(proxyBeanMethods = false)
	@ConditionalOnClass(HttpServer.class)
	public static class NettyWebServerFactoryCustomizerConfiguration {

		@Bean
		public NettyWebServerFactoryCustomizer nettyWebServerFactoryCustomizer(Environment environment,
				ServerProperties serverProperties) {
			return new NettyWebServerFactoryCustomizer(environment, serverProperties);
		}
	}

}
```

# 基本框架整合

https://github.com/Spring-projects/Spring-boot/tree/master/Spring-boot-samples

## 整合JDBC

Spring Boot 整合 JDBC 只要引入对应数据库的连接驱动包（如 mysql 的驱动），然后再 properties/yaml 配置文件中书写配置即可。

> 依赖

- Spring Boot 的 jdbc api
- mysql 驱动

> properties 文件

```properties
# jdbc基础配置
Spring.datasource.username=root
Spring.datasource.password=root
Spring.datasource.url=jdbc:mysql://localhost:3306/blog
Spring.datasource.driver-class-name=com.mysql.jdbc.Driver
```

> pom 文件

```xml
<dependencies>
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter</artifactId>
    </dependency>
    <!--jdbc api-->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-jdbc</artifactId>
    </dependency>

    <!--mysql驱动-->
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>

    <!--引入了它 配置文件书写时会有提示-->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-configuration-processor</artifactId>
        <optional>true</optional>
    </dependency>

    <!--单元测试依赖-->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-test</artifactId>
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
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

## 整合 Druid

> 概述

Spring Boot 整合 Druid，需要导入数据库的连接驱动包（如 mysql 的驱动），Druid 包，然后书写对应的配置文件。注意：由于数据库连接相关的配置文件是在 Druid 中进行设置的，所以前缀名要一致。

> 依赖

- mysql 驱动
- Spring Boot jdbc api
- druid 依赖
- Spring Boot-web 依赖，用来注册 servlet，filter 启用 druid 的控制台
- log4j，我们使用这个日志框架进行记录

> pom 文件

```xml
<dependencies>
    <!-- jdbc aip -->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-jdbc</artifactId>
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
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-web</artifactId>
    </dependency>
    <!-- 配置文件书写提示 -->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-configuration-processor</artifactId>
        <optional>true</optional>
    </dependency>
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-test</artifactId>
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
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

> properties 文件

```properties
# jdbc基础配置
Spring.datasource.username=root
Spring.datasource.password=root
Spring.datasource.url=jdbc:mysql://localhost:3306/blog
Spring.datasource.driver-class-name=com.mysql.jdbc.Driver
Spring.datasource.type=com.alibaba.druid.pool.DruidDataSource
# druid详细配置
Spring.datasource.initialSize=5
Spring.datasource.minIdle=5
Spring.datasource.maxActive=20
Spring.datasource.maxWait=60000
Spring.datasource.timeBetweenEvictionRunsMillis=60000
Spring.datasource.minEvictableIdleTimeMillis=300000
Spring.datasource.validationQuery=SELECT 1 FROM DUAL
Spring.datasource.testWhileIdle=true
Spring.datasource.testOnBorrow=false
Spring.datasource.testOnReturn=false
Spring.datasource.poolPreparedStatements=true
#配置监控统计拦截的filters，去掉后监控界面sql无法统计，'wall'用于防火墙
Spring.datasource.filters=stat,wall,log4j
Spring.datasource.maxPoolPreparedStatementPerConnectionSize=20
Spring.datasource.useGlobalDataSourceStat=true
Spring.datasource.connectionProperties=druid.stat.mergeSql=true;druid.stat.slowSqlMillis=500
```

> 代码

```java
// 表明这是一个JavaConfig配置类
@Configuration
public class DruidConfig {

    @Bean
    @ConfigurationProperties(prefix = "Spring.datasource")
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

http://www.mybatis.org/Spring-boot-starter/mybatis-Spring-boot-autoconfifigure/

> 概述

数据库采用 MySQL，连接池采用 Druid。整合方式有 SQL 用纯注解和 SQL采用 xml 两个版本。

> 依赖

- mysql 驱动
- Spring Boot jdbc api
- druid 依赖
- Spring Boot-web 依赖，用来注册 servlet，filter 启用 druid 的控制台
- log4j，我们使用这个日志框架进行记录
- mybatis 和 Spring 的整合包
- 其他的会自动帮我们导入的，不必担心

> pom文件

```xml
<dependencies>
    <!-- jdbc aip -->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-jdbc</artifactId>
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
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-web</artifactId>
    </dependency>
    <!--mybatis与Spring的整合包-->
    <dependency>
        <groupId>org.mybatis.Spring.boot</groupId>
        <artifactId>mybatis-Spring-boot-starter</artifactId>
        <version>2.1.2</version>
	</dependency>
    <!-- 配置文件书写提示 -->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-configuration-processor</artifactId>
        <optional>true</optional>
    </dependency>
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-test</artifactId>
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
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

> properties 文件

```properties
# 基础的jdbc配置
Spring.datasource.username=root
Spring.datasource.password=root
Spring.datasource.url=jdbc:mysql://localhost:3306/blog
Spring.datasource.driver-class-name=com.mysql.jdbc.Driver
Spring.datasource.type=com.alibaba.druid.pool.DruidDataSource
# 初始化druid配置
Spring.datasource.initialSize=5
Spring.datasource.minIdle=5
Spring.datasource.maxActive=20
Spring.datasource.maxWait=60000
Spring.datasource.timeBetweenEvictionRunsMillis=60000
Spring.datasource.minEvictableIdleTimeMillis=300000
Spring.datasource.validationQuery=SELECT 1 FROM DUAL
Spring.datasource.testWhileIdle=true
Spring.datasource.testOnBorrow=false
Spring.datasource.testOnReturn=false
Spring.datasource.poolPreparedStatements=true
# 开启druid的监控
Spring.datasource.filters=stat,wall,log4j
Spring.datasource.maxPoolPreparedStatementPerConnectionSize=20
Spring.datasource.useGlobalDataSourceStat=true
Spring.datasource.connectionProperties=druid.stat.mergeSql=true;druid.stat.slowSqlMillis=500
# mybatis采用xml书写SQL的话需要这一行，指定xml文件的位置
mybatis.mapper_locations=classpath:mapper/*.xml
# 开启驼峰命名
mybatis.configuration.map-underscore-to-camel-case=true
# 配置包别名
mybatis.type-aliases-package=com.bbxx.boot02.pojo
```

> SQL 纯注解

每个 dao 接口都加上注解 mapper 或在启动引导类位置处用 @MapperScan("扫描的包全名")

> SQL 书写

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

也可以在配置文件中开启：`mybatis.configuration.map-underscore-to-camel-case=true`

```java
@org.Springframework.context.annotation.Configuration
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

> SQL xml 版本

xml 放在那里？放在 resources 目录下，路径和 Mapper 接口的路径要一致。比如 MapperXX 在 com.xx.dao 那么对应的 xml 文件就应该放在 resources/com/xx/dao 下。

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
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-data-jpa</artifactId>
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
Spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
Spring.datasource.url=jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC
Spring.datasource.username=root
Spring.datasource.password=root

# jpa信息配置
Spring.jpa.database=MySQL
Spring.jpa.show-sql=true
Spring.jpa.generate-ddl=true
Spring.jpa.hibernate.ddl-auto=update
Spring.jpa.hibernate.naming_strategy=org.hibernate.cfg.ImprovedNamingStrategy
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
	// 省略 setter/getter toString
}
```

```java
package cn.baobaoxuxu.community.repository;

import cn.baobaoxuxu.community.pojo.Role;
import org.Springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface RoleRepository extends JpaRepository<Role, Long> {
    public List<Role> findAll();
}
```

## 整合MVC

> 概述

Spring Boot 有默认的配置，它替我们设置了有关 MVC 的一些默认配置。我们可以不使用这些默认配置，全面接管相关配置（全部由我们自行定义），也可以只修改必要的部分，其他的仍采用 Spring Boot 为我们提供的默认配置。一般是不采用全面接管。

> 依赖

Spring Boot-web 模块

> pom 文件

```xml
<dependencies>
   <!--web模块-->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-web</artifactId>
    </dependency>
    <!-- thymeleaf Spring Boot默认的模板引擎，顺带一起导入了。高版本boot，写这个即可，其他的不用写-->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-thymeleaf</artifactId>
    </dependency>

    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-configuration-processor</artifactId>
        <optional>true</optional>
    </dependency>
    <dependency>
        <groupId>junit</groupId>
        <artifactId>junit</artifactId>
        <version>4.13</version>
        <scope>test</scope>
    </dependency>

    <dependency>
        <groupId>org.mybatis.Spring.boot</groupId>
        <artifactId>mybatis-Spring-boot-starter</artifactId>
        <version>2.1.2</version>
    </dependency>

    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-test</artifactId>
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
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-maven-plugin</artifactId>
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

    // 重写方法式 的配置
    @Override
    public void addViewControllers(ViewControllerRegistry registry) {
        // 浏览器发送 /demo1 请求到success
        registry.addViewController("/demo1").setViewName("/success");
        registry.addViewController("/demo2").setViewName("/success");
    }

    // 组件式 配置
    @Bean
    public WebMvcConfigurer webMvcConfigurer() {
        WebMvcConfigurer web = new WebMvcConfigurer() {
            @Override
            public void addInterceptors(InterceptorRegistry registry) {
                registry.addInterceptor(new LoginInterceptor()).addPathPatterns("/<b>").
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

Spring Boot 的 SSM 整合配置

> 依赖

- mysql 驱动
- Spring Boot jdbc api
- druid 依赖
- Spring Boot-web 依赖，用来注册 servlet，filter 启用 druid 的控制台
- log4j，我们使用这个日志框架进行记录
- mybatis 和 Spring 的整合包
- 其他的会自动帮我们导入的，不必担心

> pom 文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-parent</artifactId>
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
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-starter-jdbc</artifactId>
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
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-starter-web</artifactId>
        </dependency>

        <!-- thymeleaf导入-->
        <dependency>
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-starter-thymeleaf</artifactId>
        </dependency>

        <!-- 配置文件提示 -->
        <dependency>
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-configuration-processor</artifactId>
            <optional>true</optional>
        </dependency>
        
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13</version>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>org.mybatis.Spring.boot</groupId>
            <artifactId>mybatis-Spring-boot-starter</artifactId>
            <version>2.1.2</version>
        </dependency>

        <dependency>
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-starter-test</artifactId>
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
                <groupId>org.Springframework.boot</groupId>
                <artifactId>Spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

</project>
```

> properties 文件

```properties
# 禁用缓存
Spring.thymeleaf.cache=false
# 基础的jdbc配置
Spring.datasource.username=root
Spring.datasource.password=root
Spring.datasource.url=jdbc:mysql://localhost:3306/blog
Spring.datasource.driver-class-name=com.mysql.jdbc.Driver
Spring.datasource.type=com.alibaba.druid.pool.DruidDataSource
# druid相关配置
Spring.datasource.initialSize=5
Spring.datasource.minIdle=5
Spring.datasource.maxActive=20
Spring.datasource.maxWait=60000
Spring.datasource.timeBetweenEvictionRunsMillis=60000
Spring.datasource.minEvictableIdleTimeMillis=300000
Spring.datasource.validationQuery=SELECT 1 FROM DUAL
Spring.datasource.testWhileIdle=true
Spring.datasource.testOnBorrow=false
Spring.datasource.testOnReturn=false
Spring.datasource.poolPreparedStatements=true
#   配置监控统计拦截的filters，去掉后监控界面sql无法统计，'wall'用于防火墙
Spring.datasource.filters=stat,wall,log4j
Spring.datasource.maxPoolPreparedStatementPerConnectionSize=20
Spring.datasource.useGlobalDataSourceStat=true
Spring.datasource.connectionProperties=druid.stat.mergeSql=true;druid.stat.slowSqlMillis=500
# mybatis xml方式书写SQL
mybatis.mapper_locations=classpath:mapper/*.xml
mybatis.configuration.map-underscore-to-camel-case=true
mybatis.type-aliases-package=com.bbxx.boot02.pojo
```

> Config 代码

druid

```java
@Configuration
public class DruidConfig {

    @ConfigurationProperties(prefix = "Spring.datasource")
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
    public void postHandle(HttpServletRequest request, HttpServletResponse response, Object handler, ModelAndView modelAndView) throws Exception {}

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) throws Exception {}
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
                registry.addInterceptor(new LoginInterceptor()).addPathPatterns("/<b>").
                        excludePathPatterns("/index.html", "/login.html","/index","/list");
            }
        };
        return web;
    }
}
```

mybatis 的 sql 文件

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
import org.Springframework.beans.factory.annotation.Autowired;
import org.Springframework.boot.test.context.SpringBootTest;
import org.Springframework.data.redis.core.RedisTemplate;
import org.Springframework.test.context.junit4.SpringRunner;

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
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-test</artifactId>
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

基于 Spring Boot 实现 SSMP 整合，学习如下技术的整合方式

- 整合 JUnit
- 整合 MyBatis
- 整合 MyBatis-Plus
- 整合 Druid


## 整合JUnit

先来看一下不使用 Spring Boot 技术时，Spring 整合 JUnit 的制作方式

```JAVA
//加载Spring整合junit专用的类运行器
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

其中核心代码是前两个注解

1️⃣第一个注解 `@RunWith` 是设置 Spring 专用的测试类运行器，简单说就是 Spring 程序执行程序有自己的一套独立的运行程序的方式，不能使用 JUnit 提供的类运行方式了，必须指定一下，但是格式是固定的，琢磨一下，<font color="#ff0000"><b>每次都指定一样的东西，这个东西写起来没有技术含量啊</b></font>，

2️⃣第二个注解 @ContextConfiguration 是用来设置 Spring 核心配置文件或配置类的，简单说就是加载 Spring 的环境你要告诉 Spring 具体的环境配置是在哪里写的，虽然每次加载的文件都有可能不同，但是仔细想想，如果文件名是固定的，这个貌似也是一个固定格式。既然<font color="#ff0000"><b>有可能是固定格式，那就有可能每次都写一样的东西，也是一个没有技术含量的内容书写</b></font>

Spring Boot 就抓住上述两条没有技术含量的内容书写进行开发简化，能走默认值的走默认值，能不写的就不写，具体格式如下

```JAVA
@SpringBootTest
class Spring Boot04JunitApplicationTests {
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

这次简化，一个注解就搞定了，而且还没有参数，Spring Boot 整合其他技术的优势就在两个字——<font color="#ff0000"><b>简化</b></font>。使用一个注解 @SpringBootTest 替换了前面两个注解。其内部和之前一样，只不过都走默认值。

代码加载的配置类或者配置文件是哪一个？就是我们前面启动程序使用的引导类。如果想手工指定引导类有两种方式，第一种方式使用属性的形式进行，在注解 @SpringBootTest 中添加 classes 属性指定配置类

```JAVA
@SpringBootTest(classes = Spring Boot04JunitApplication.class)
class Spring Boot04JunitApplicationTests {
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

第二种方式回归原始配置方式，仍然使用 @ContextConfiguration 注解进行，效果是一样的

```JAVA
@SpringBootTest
@ContextConfiguration(classes = Spring Boot04JunitApplication.class)
class Spring Boot04JunitApplicationTests {
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

使用 Spring Boot 整合 JUnit 需要保障导入 test 对应的 starter，由于初始化项目时此项是默认导入的，所以此处没有提及，其实和之前学习的内容一样，用什么技术导入对应的 starter 即可。

<b>总结</b>

1. 导入测试对应的 starter
2. 测试类使用 @SpringBootTest 修饰
3. 使用自动装配的形式添加要测试的对象
4. 测试类如果存在于引导类所在包或子包中无需指定引导类
5. 测试类如果不存在于引导类所在的包或子包中需要通过 classes 属性指定引导类

## 整合MyBatis

整合完 JUnit 下面再来说一下整合 MyBatis，下面列举出原始整合的全部内容，以配置类的形式为例进行

导入 MyBatis 坐标，Spring 整合 MyBatis 还有自己专用的坐标，此外 Spring 进行数据库操作的 jdbc 坐标是必须的，剩下还有 mysql 驱动坐标，本例中使用的 Druid 数据源可以不要

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
    <!--1.导入mybatis与Spring整合的jar包-->
    <dependency>
        <groupId>org.mybatis</groupId>
        <artifactId>mybatis-Spring</artifactId>
        <version>1.3.0</version>
    </dependency>
    <!--导入Spring操作数据库必选的包-->
    <dependency>
        <groupId>org.Springframework</groupId>
        <artifactId>Spring-jdbc</artifactId>
        <version>5.2.10.RELEASE</version>
    </dependency>
</dependencies>
```

Spring 核心配置

```JAVA
@Configuration
@ComponentScan("com.itheima")
@PropertySource("jdbc.properties")
public class SpringConfig {
}
```

MyBatis 要交给 Spring 接管的 bean

```JAVA
//定义mybatis专用的配置类
@Configuration
public class MyBatisConfig {
//    定义创建SqlSessionFactory对应的bean
    @Bean
    public SqlSessionFactoryBean sqlSessionFactory(DataSource dataSource){
        //SqlSessionFactoryBean是由mybatis-Spring包提供的，专用于整合用的对象
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

数据源对应的 bean，此处使用 Druid 数据源

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

数据库连接信息（properties 格式）

```properties
jdbc.driver=com.mysql.jdbc.Driver
jdbc.url=jdbc:mysql://localhost:3306/Spring_db?useSSL=false
jdbc.username=root
jdbc.password=root
```

上述格式基本上是最简格式了，要写的东西还真不少。下面看看 Spring Boot 整合 MyBaits 格式

<b>步骤①</b>：创建模块

<div align="center"><img src="img/boot/image-20211129092156020.png"></div>

<b>步骤②</b>：勾选要使用的技术，MyBatis，由于要操作数据库，还要勾选对应数据库

<div align="center"><img src="img/boot/image-20211129092210993.png"></div>

或者手工导入对应技术的 starter，和对应数据库的坐标

```XML
<dependencies>
    <!--1.导入对应的starter-->
    <dependency>
        <groupId>org.mybatis.Spring.boot</groupId>
        <artifactId>mybatis-Spring-boot-starter</artifactId>
        <version>2.2.0</version>
    </dependency>

    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

<b>步骤③</b>：配置数据源相关信息，没有这个信息你连接哪个数据库都不知道

```yaml
#2.配置相关信息
Spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db
    username: root
    password: root
```

结束了，就这么多。Spring Boot 把配置中所有可能出现的通用配置都简化了。下面写一个 MyBatis 程序运行需要的 Dao（或者 Mapper）就可以运行了

<b>实体类</b>

```JAVA
public class Book {
    private Integer id;
    private String type;
    private String name;
    private String description;
}
```

<b>映射接口（Dao）</b>

```JAVA
@Mapper
public interface BookDao {
    @Select("select * from tbl_book where id = #{id}")
    public Book getById(Integer id);
}
```

<b>测试类</b>

```JAVA
@SpringBootTest
class Spring Boot05MybatisApplicationTests {
    @Autowired
    private BookDao bookDao;
    @Test
    void contextLoads() {
        System.out.println(bookDao.getById(1));
    }
}
```

<font color="#ff0000"><b>注意</b></font>：当前使用的 Spring Boot 版本是 2.5.4，对应的坐标设置中 MySQL 驱动使用的是 8x 版本。使用 Spring Boot 2.4.3（不含）之前版本会出现一个小 BUG，就是 MySQL 驱动升级到 8 以后要求强制配置时区，如果不设置会出问题。解决方案很简单，驱动 url上面添加上对应设置就行了

```YAML
#2.配置相关信息
Spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
    username: root
    password: root
```

这里设置的 UTC 是全球标准时间，你也可以理解为是英国时间，中国处在东八区，需要在这个基础上加上 8 小时，这样才能和中国地区的时间对应的，也可以修改配置为 Asia/Shanghai，同样可以解决这个问题。

```YAML
#2.配置相关信息
Spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=Asia/Shanghai
    username: root
    password: root
```

如果不想每次都设置这个东西，也可以去修改 mysql 中的配置文件 mysql.ini，在 mysqld 项中添加 default-time-zone=+8:00 也可以解决这个问题。

此外在运行程序时还会给出一个提示，说数据库驱动过时的警告，根据提示修改配置即可，弃用<b>com.mysql.jdbc.Driver</b>，换用 <font color="#ff0000"><b>com.mysql.cj.jdbc.Driver</b></font>。前面的例子中已经更换了驱动了，在此说明一下。

```tex
Loading class `com.mysql.jdbc.Driver'. 
This is deprecated. The new driver class is `com.mysql.cj.jdbc.Driver'. 
The driver is automatically registered via the SPI and manual loading of the driver class is generally unnecessary.
```

<b>总结</b>

1. 整合操作需要勾选 MyBatis 技术，也就是导入 MyBatis 对应的 starter

2. 数据库连接相关信息转换成配置

3. 数据库 SQL 映射需要添加 @Mapper 被容器识别到

4. MySQL 8.X 驱动强制要求设置时区

    - 修改 url，添加 serverTimezone 设定
    - 修改 MySQL 数据库配置

5. 驱动类过时，提醒更换为 com.mysql.cj.jdbc.Driver


## 整合MyBatis-Plus

整合的核心

- 导入对应技术的 starter 坐标

- 根据对应技术的要求做配置

    虽然看起来有点虚，但是确实是这个理儿，下面趁热打铁，再换一个技术，看看是不是上面这两步。

接下来在 MyBatis 的基础上再升级一下，整合 MyBaitsPlus（简称 MP），国人开发的技术，符合中国人开发习惯。

<b>步骤①</b>：导入对应的 starter

```XML
<dependency>
    <groupId>com.baomidou</groupId>
    <artifactId>mybatis-plus-boot-starter</artifactId>
    <version>3.4.3</version>
</dependency>
```

关于这个坐标，此处要说明一点，之前我们看的 starter 都是 Spring-boot-starter-？？？，也就是说都是下面的格式

```tex
Spring-boot-start-***
```

而 MyBatis 与 MyBatisPlus 这两个坐标的名字书写比较特殊，是第三方技术名称在前，boot 和 starter 在后。此处简单提一下命名规范，后期原理篇会再详细讲解

| starter 所属 | 命名规则                                                    | 示例                                                      |
| ------------ | ----------------------------------------------------------- | --------------------------------------------------------- |
| 官方提供     | Spring-boot-starter-技术名称                                | Spring-boot-starter-web <br/>Spring-boot-starter-test     |
| 第三方提供   | 第三方技术名称-Spring-boot-starter                          | mybatis-Spring-boot-starter<br/>druid-Spring-boot-starter |
| 第三方提供   | 第三方技术名称-boot-starter（第三方技术名称过长，简化命名） | mybatis-plus-boot-starter                                 |

<font color="#f0f"><b>温馨提示</b></font>

截止目前，Spring Boot 官网还未收录此坐标，而我们 Idea 创建模块时读取的是 Spring Boot 官网的 Spring Initializr，所以也没有。如果换用阿里云的 url 创建项目可以找到对应的坐标。

<b>步骤②</b>：配置数据源相关信息

```yaml
#2.配置相关信息
Spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db
    username: root
    password: root
```

没了，就这么多，剩下的就是写 MyBaitsPlus 的程序了

<b>映射接口（Dao）</b>

```JAVA
@Mapper
public interface BookDao extends BaseMapper<Book> {}
```

核心在于 Dao 接口继承了一个 BaseMapper 的接口，这个接口中帮助开发者预定了若干个常用的 API 接口，简化了通用 API 接口的开发工作。

<div align="center"><img src="img/boot/image-20211129100313919.png" alt="image-20211129100313919" style="zoom:80%;" /></div>

下面就可以写一个测试类进行测试了，此处省略。

<font color="#f0f"><b>温馨提示</b></font>

目前数据库的表名定义规则是tbl_模块名称，为了能和实体类相对应，需要做一个配置，相关知识各位小伙伴可以到 MyBatisPlus 课程中去学习，此处仅给出解决方案。配置 application.yml 文件，添加如下配置即可，设置所有表名的通用前缀名

```yaml
mybatis-plus:
  global-config:
    db-config:
      table-prefix: tbl_		#设置所有表的通用前缀名称为tbl_
```

<b>总结</b>

1. 手工添加 MyBatis-Plus 对应的 starter
2. 数据层接口使用 BaseMapper 简化开发
3. 需要使用的第三方技术无法通过勾选确定时，需要手工添加坐标

## 整合Druid

使用 Spring Boot 整合了 3 个技术了，发现套路基本相同，导入对应的 starter，然后做配置，我们需要一直强化这套思想。下面再整合一个技术，继续深入强化此思想。

前面整合 MyBatis 和 MyBatisPlus 的时候，使用的数据源对象都是 Spring Boot 默认的数据源对象，下面我们手工控制一下，自己指定了一个数据源对象，Druid。

在没有指定数据源时，我们的配置如下：

```YAML
#2.配置相关信息
Spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=Asia/Shanghai
    username: root
    password: root
```

此时虽然没有指定数据源，但是根据 Spring Boot 的特点，肯定帮我们选了一个它认为最好的数据源对象，这就是 HiKari。通过启动日志可以查看到对应的身影。

```tex
2021-11-29 09:39:15.202  INFO 12260 --- [           main] com.zaxxer.hikari.HikariDataSource       : HikariPool-1 - Starting...
2021-11-29 09:39:15.208  WARN 12260 --- [           main] com.zaxxer.hikari.util.DriverDataSource  : Registered driver with driverClassName=com.mysql.jdbc.Driver was not found, trying direct instantiation.
2021-11-29 09:39:15.551  INFO 12260 --- [           main] com.zaxxer.hikari.HikariDataSource       : HikariPool-1 - Start completed.
```

上述信息中每一行都有 HiKari 的身影，如果需要更换数据源，其实只需要两步即可。

1. 导入对应的技术坐标

2. 配置使用指定的数据源类型

    下面就切换一下数据源对象

<b>步骤①</b>：导入对应的坐标（注意，是坐标，此处不是 starter）

```XML
<dependencies>
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid</artifactId>
        <version>1.1.16</version>
    </dependency>
</dependencies>
```

<b>步骤②</b>：修改配置，在数据源配置中有一个 type 属性，专用于指定数据源类型

```YAML
Spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
    username: root
    password: root
    type: com.alibaba.druid.pool.DruidDataSource
```

这里其实要提出一个问题的，目前的数据源配置格式是一个通用格式，不管你换什么数据源都可以用这种形式进行配置。但是新的问题又来了，如果对数据源进行个性化的配置，例如配置数据源对应的连接数量，这个时候就有新的问题了。每个数据源技术对应的配置名称都一样吗？肯定不是啊，各个厂商不可能提前商量好都写一样的名字啊，怎么办？就要使用专用的配置格式了。这个时候上面这种通用格式就不能使用了，怎么办？还能怎么办？按照 Spring Boot 整合其他技术的通用规则来套啊，导入对应的 starter，进行相应的配置即可。

<b>步骤①</b>：导入对应的 starter

```XML
<dependencies>
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid-Spring-boot-starter</artifactId>
        <version>1.2.6</version>
    </dependency>
</dependencies>
```

<b>步骤②</b>：修改配置

```YAML
Spring:
  datasource:
    druid:
      driver-class-name: com.mysql.cj.jdbc.Driver
      url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
      username: root
      password: root
```

注意观察，配置项中，在 datasource 下面并不是直接配置 url 这些属性的，而是先配置了一个 druid 节点，然后再配置的 url 这些东西。言外之意，url 这个属性是 druid 下面的属性，那你能想到什么？除了这 4 个常规配置外，还有 druid 专用的其他配置。通过提示功能可以打开 druid 相关的配置查阅

<div align="center"><img src="img/boot/image-20211129112610729.png" alt="image-20211129112610729" style="zoom:80%;" /></div>

与 druid 相关的配置超过 200 条以上，这就告诉你，如果想做 druid 相关的配置，使用这种格式就可以了，这里就不展开描述了，太多了。

这是我们做的第 4 个技术的整合方案，还是那两句话：<font color="#ff0000"><b>导入对应 starter，使用对应配置</b></font>。没了，Spring Boot 整合其他技术就这么简单粗暴。

<b>总结</b>

1. 整合 Druid 需要导入 Druid 对应的 starter
2. 根据 Druid 提供的配置方式进行配置
3. 整合第三方技术通用方式
    - 导入对应的 starter
    - 根据提供的配置格式，配置非默认值对应的配置项

## SSMP整合综合案例

将所有知识贯穿起来，同时做一个小功能

<b>主页面</b>

<div align="center"><img src="img/boot/image-20211129113447844.png"></div>

<b>添加</b>

<div align="center"><img src="img/boot/image-20211129113522459.png"></div>

<b>删除</b>

<div align="center"><img src="img/boot/image-20211129113550829.png"></div>

<b>修改</b>

<b>分页</b>

<div align="center"><img src="img/boot/image-20211129113628969.png"></div>

<b>条件查询</b>

<div align="center"><img src="img/boot/image-20211129113650369.png"></div>

整体案例中需要采用的技术如下

1. 实体类开发————使用 Lombok 快速制作实体类
2. Dao 开发————整合 MyBatisPlus，制作数据层测试
3. Service 开发————基于 MyBatisPlus 进行增量开发，制作业务层测试类
4. Controller 开发————基于 Restful 开发，使用 PostMan 测试接口功能
5. Controller 开发————前后端开发协议制作
6. 页面开发————基于 VUE+ElementUI 制作，前后端联调，页面数据处理，页面消息处理
    - 列表
    - 新增
    - 修改
    - 删除
    - 分页
    - 查询
7. 项目异常处理
8. 按条件查询————页面功能调整、Controller 修正功能、Service 修正功能

#### 0.模块创建

​		对于这个案例如果按照企业开发的形式进行应该制作后台微服务，前后端分离的开发。

<div align="center"><img src="img/boot/image-20211129114306062.png" alt="image-20211129114306062" style="zoom:80%;" /></div>

​		我知道这个对初学的小伙伴要求太高了，咱们简化一下。后台做单体服务器，前端不使用前后端分离的制作了。

<div align="center"><img src="img/boot/image-20211129114328967.png" alt="image-20211129114328967" style="zoom:80%;" /></div>

一个服务器即充当后台服务调用，又负责前端页面展示，降低学习的门槛。

下面我们创建一个新的模块，加载要使用的技术对应的 starter，修改配置文件格式为 yml 格式，并把 web 访问端口先设置成 80。

<b>pom.xml</b>

```XML
<dependencies>
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-test</artifactId>
        <scope>test</scope>
    </dependency>
</dependencies>
```

<b>application.yml</b>

```yaml
server:
  port: 80
```

#### 1.实体类开发

本案例对应的模块表结构如下：

```mysql
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

根据上述表结构，制作对应的实体类

<b>实体类</b>

```JAVA
public class Book {
    private Integer id;
    private String type;
    private String name;
    private String description;
}
```

实体类的开发可以自动通过工具手工生成 get/set 方法，然后覆盖 toString() 方法，方便调试，等等。不过这一套操作书写很繁琐，有对应的工具可以帮助我们简化开发，介绍一个小工具，lombok。

Lombok，一个 Java 类库，提供了一组注解，简化 POJO 实体类开发，Spring Boot 目前默认集成了 lombok 技术，并提供了对应的版本控制，所以只需要提供对应的坐标即可，在 pom.xml 中添加 lombok 的坐标。

```XML
<dependencies>
    <!--lombok-->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
    </dependency>
</dependencies>
```

使用 lombok 可以通过一个注解 @Data 完成一个实体类对应的 getter，setter，toString，equals，hashCode 等操作的快速添加

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

<b>总结</b>

1. 实体类制作
2. 使用 lombok 简化开发
    - 导入 lombok 无需指定版本，由 Spring Boot 提供版本
    - @Data 注解

#### 2.数据层开发——基础CRUD

数据层开发本次使用 MyBatisPlus 技术，数据源使用前面学习的 Druid，学都学了都用上。

<b>步骤①</b>：导入 MyBatisPlus 与 Druid 对应的 starter，当然 mysql 的驱动不能少

```xml
<dependencies>
    <dependency>
        <groupId>com.baomidou</groupId>
        <artifactId>mybatis-plus-boot-starter</artifactId>
        <version>3.4.3</version>
    </dependency>
    <dependency>
        <groupId>com.alibaba</groupId>
        <artifactId>druid-Spring-boot-starter</artifactId>
        <version>1.2.6</version>
    </dependency>
    <dependency>
        <groupId>mysql</groupId>
        <artifactId>mysql-connector-java</artifactId>
        <scope>runtime</scope>
    </dependency>
</dependencies>
```

<b>步骤②</b>：配置数据库连接相关的数据源配置

```YAML
server:
  port: 80

Spring:
  datasource:
    druid:
      driver-class-name: com.mysql.cj.jdbc.Driver
      url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
      username: root
      password: root
```

<b>步骤③</b>：使用 MyBatisPlus 的标准通用接口 BaseMapper 加速开发，别忘了 @Mapper 和泛型的指定

```JAVA
@Mapper
public interface BookDao extends BaseMapper<Book> {}
```

<b>步骤④</b>：制作测试类测试结果

```JAVA
package com.itheima.dao;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.itheima.domain.Book;
import org.junit.jupiter.api.Test;
import org.Springframework.beans.factory.annotation.Autowired;
import org.Springframework.boot.test.context.SpringBootTest;

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

MyBatisPlus 技术默认的主键生成策略为雪花算法，生成的主键 ID 长度较大，和目前的数据库设定规则不相符，需要配置一下使 MyBatisPlus 使用数据库的主键生成策略。在 application.yml 中添加对应配置即可，具体如下

```yaml
server:
  port: 80

Spring:
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

在进行数据层测试的时候，因为基础的 CRUD 操作均由 MyBatisPlus 给我们提供了，所以就出现了一个局面，开发者不需要书写 SQL 语句了，这样程序运行的时候总有一种感觉，一切的一切都是黑盒的，作为开发者我们啥也不知道就完了。如果程序正常运行还好，如果报错了，这个时候就很崩溃，你甚至都不知道从何下手，因为传递参数、封装 SQL 语句这些操作完全不是你开发出来的，所以查看执行期运行的 SQL 语句就成为当务之急。

Spring Boot 整合 MyBatisPlus 的时候充分考虑到了这点，通过配置的形式就可以查阅执行期 SQL 语句，配置如下

```YAML
mybatis-plus:
  global-config:
    db-config:
      table-prefix: tbl_
      id-type: auto
  configuration:
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
```

再来看运行结果，此时就显示了运行期执行 SQL 的情况。

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

其中清晰的标注了当前执行的 SQL 语句是什么，携带了什么参数，对应的执行结果是什么，所有信息应有尽有。此处设置的是日志的显示形式，当前配置的是控制台输出，当然还可以由更多的选择，根据需求切换即可

<div align="center"><img src="img/boot/image-20211129143207295.png" alt="image-20211129143207295" style="zoom:80%;" /></div>

<b>总结</b>

1. 手工导入 starter 坐标（2个），mysql 驱动（1个）

2. 配置数据源与 MyBatisPlus 对应的配置

3. 开发 Dao 接口（继承 BaseMapper）

4. 制作测试类测试 Dao 功能是否有效

5. 使用配置方式开启日志，设置日志输出方式为标准输出即可查阅 SQL 执行日志


#### 3.数据层开发——分页功能制作

前面仅仅是使用了 MyBatisPlus 提供的基础 CRUD 功能，实际上 MyBatisPlus 给我们提供了几乎所有的基础操作，这一节说一下如何实现数据库端的分页操作。

MyBatisPlus 提供的分页操作 API 如下：

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

其中 selectPage 方法需要传入一个封装分页数据的对象，可以通过 new 的形式创建这个对象，当然这个对象也是 MyBatisPlus 提供的，别选错包了。创建此对象时需要指定两个分页的基本数据

- 当前显示第几页
- 每页显示几条数据


可以通过创建 Page 对象时利用构造方法初始化这两个数据。

```JAVA
IPage page = new Page(2,5);
```

将该对象传入到查询方法 selectPage 后，可以得到查询结果，但是我们会发现当前操作查询结果返回值仍然是一个 IPage 对象

```JAVA
IPage page = bookDao.selectPage(page, null);
```

IPage 对象中封装了若干个数据，而查询的结果作为 IPage 对象封装的一个数据存在的，可以理解为查询结果得到后，又塞到了这个 IPage 对象中，其实还是为了高度的封装，一个 IPage 描述了分页所有的信息。下面 5 个操作就是 IPage 对象中封装的所有信息了。

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

到这里就知道这些数据如何获取了，但是当你去执行这个操作时，你会发现并不像我们分析的这样，实际上这个分页功能当前是无效的。为什么这样呢？这个要源于 MyBatisPlus 的内部机制。

对于 MySQL 的分页操作使用 limit 关键字进行，而并不是所有的数据库都使用 limit 关键字实现的，这个时候 MyBatisPlus 为了制作的兼容性强，将分页操作设置为基础查询操作的升级版。

基础操作中有查询全部的功能，而在这个基础上只需要升级一下（PLUS）就可以得到分页操作。所以 MyBatisPlus 将分页操作做成了一个开关，你用分页功能就把开关开启，不用就不需要开启这个开关。而我们现在没有开启这个开关，所以分页操作是没有的。这个开关是通过 MyBatisPlus 的拦截器的形式存在的，具体设置方式如下：

<b>定义 MyBatisPlus 拦截器并将其设置为 Spring 管控的 bean</b>

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

上述代码第一行是创建 MyBatisPlus 的拦截器栈，这个时候拦截器栈中没有具体的拦截器，第二行是初始化了分页拦截器，并添加到拦截器栈中。如果后期开发其他功能，需要添加全新的拦截器，按照第二行的格式继续 add 进去新的拦截器就可以了。

<b>总结</b>

1. 使用 IPage 封装分页数据
2. 分页操作依赖 MyBatisPlus 分页拦截器实现功能
3. 借助 MyBatisPlus 日志查阅执行 SQL 语句

#### 4.数据层开发——条件查询功能制作

除了分页功能，MyBatisPlus 还提供有强大的条件查询功能。以往我们写条件查询要自己动态拼写复杂的 SQL 语句，现在简单了，MyBatisPlus 将这些操作都制作成 API 接口，调用一个又一个的方法就可以实现各种条件的拼装。

下面的操作就是执行一个模糊匹配对应的操作，由 like 条件书写变为了 like 方法的调用。

```JAVA
@Test
void testGetBy(){
    QueryWrapper<Book> qw = new QueryWrapper<>();
    qw.like("name","Spring");
    bookDao.selectList(qw);
}
```

其中第一句 QueryWrapper 对象是一个用于封装查询条件的对象，该对象可以动态使用 API 调用的方法添加条件，最终转化成对应的 SQL 语句。第二句就是一个条件了，需要什么条件，使用 QueryWapper 对象直接调用对应操作即可。比如做大于小于关系，就可以使用 lt 或 gt 方法，等于使用 eq 方法，等等。

这组 API 使用还是比较简单的，但是关于属性字段名的书写存在着安全隐患，比如查询字段 name，当前是以字符串的形态书写的，万一写错，编译器还没有办法发现，只能将问题抛到运行器通过异常堆栈告诉开发者，不太友好。

MyBatisPlus 针对字段检查进行了功能升级，全面支持 Lambda 表达式，就有了下面这组 API。由 QueryWrapper 对象升级为 LambdaQueryWrapper 对象，这下就避免了上述问题的出现。

```JAVA
@Test
void testGetBy2(){
    String name = "1";
    LambdaQueryWrapper<Book> lqw = new LambdaQueryWrapper<Book>();
    lqw.like(Book::getName,name);
    bookDao.selectList(lqw);
}
```

为了便于开发者动态拼写 SQL，防止将 null 数据作为条件使用，MyBatisPlus 还提供了动态拼装 SQL 的快捷书写方式。

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

<b>总结</b>

1. 使用 QueryWrapper 对象封装查询条件

2. 推荐使用 LambdaQueryWrapper 对象

3. 所有查询操作封装成方法调用

4. 查询条件支持动态条件拼装


#### 5.业务层开发

数据层开发告一段落，下面进行业务层开发，其实标准业务层开发很多初学者认为就是调用数据层，怎么说呢？这个理解是没有大问题的，更精准的说法应该是<font color="#ff0000"><b>组织业务逻辑功能，并根据业务需求，对数据持久层发起调用</b></font>。有什么差别呢？目标是为了组织出符合需求的业务逻辑功能，至于调不调用数据层还真不好说，有需求就调用，没有需求就不调用。

一个常识性的知识普及一下，业务层的方法名定义一定要与业务有关，例如登录操作

```JAVA
login(String username,String password);
```

而数据层的方法名定义<span style="color:red">一定与业务无关</span>，例如根据用户名密码查询

```JAVA
selectByUserNameAndPassword(String username,String password);
```

我们在开发的时候是可以根据完成的工作不同划分成不同职能的开发团队的。比如一个哥们制作数据层，他就可以不知道业务是什么样子，拿到的需求文档要求可能是这样的

```tex
接口：传入用户名与密码字段，查询出对应结果，结果是单条数据
接口：传入ID字段，查询出对应结果，结果是单条数据
接口：传入离职字段，查询出对应结果，结果是多条数据
```

但是进行业务功能开发的哥们，拿到的需求文档要求差别就很大。接口：传入用户名与密码字段，对用户名字段做长度校验，4-15 位，对密码字段做长度校验，8 到 24 位，对密码字段做特殊字符校验，不允许存在空格，查询结果为对象。如果为 null，返回 BusinessException，封装消息码 INFO_LOGON_USERNAME_PASSWORD_ERROR

比较一下，能是一回事吗？差别太大了，所以说业务层方法定义与数据层方法定义差异化很大，只不过有些入门级的开发者手懒或者没有使用过公司相关的 ISO 标准化文档而已。

业务层接口定义如下：

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

业务层实现类如下，转调数据层即可：

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

别忘了对业务层接口进行测试，测试类如下：

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

<b>总结</b>

1. Service 接口名称定义成业务名称，并与 Dao 接口名称进行区分
2. 制作测试类测试 Service 功能是否有效

##### 业务层快速开发

其实 MyBatisPlus 技术不仅提供了数据层快速开发方案，业务层 MyBatisPlus 也给了一个通用接口，个人观点不推荐使用。其实就是一个封装+继承的思想，代码给出，实际开发慎用。

业务层接口快速开发

```JAVA
public interface IBookService extends IService<Book> {
    //添加非通用操作API接口
}
```

业务层接口实现类快速开发，关注继承的类需要传入两个泛型，一个是数据层接口，另一个是实体类。

```JAVA
@Service
public class BookServiceImpl extends ServiceImpl<BookDao, Book> implements IBookService {
    @Autowired
    private BookDao bookDao;
	//添加非通用操作API
}
```

如果感觉 MyBatisPlus 提供的功能不足以支撑你的使用需要（其实是一定不能支撑的，因为需求不可能是通用的），在原始接口基础上接着定义新的 API 接口就行了，但是不要和已有的 API 接口名冲突即可。

<b>总结</b>

1. 使用通用接口（ISerivce\<T\>）快速开发 Service
2. 使用通用实现类（ServiceImpl<M,T>）快速开发 ServiceImpl
3. 可以在通用接口基础上做功能重载或功能追加
4. 注意重载时不要覆盖原始操作，避免原始提供的功能丢失

#### 6.表现层开发

终于做到表现层了，做了这么多都是基础工作。其实你现在回头看看，哪里还有什么 Spring Boot 的影子？前面 1,2 步就搞完了。继续完成表现层制作吧，咱们表现层的开发使用基于 Restful 的表现层接口开发，功能测试通过 Postman 工具进行。

表现层接口如下:

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

在使用 Postman 测试时关注提交类型，对应上即可，不然就会报 405 的错误码了。

<b>普通 GET 请求</b>

<div align="center"><img src="img/boot/image-20211129152237371.png"></div>

<b>PUT 请求传递 json 数据，后台实用 @RequestBody 接收数据</b>

<div align="center"><img src="img/boot/image-20211129152300177.png"></div>

<b>GET 请求传递路径变量，后台实用 @PathVariable 接收数据</b>

<div align="center"><img src="img/boot/image-20211129152320073.png"></div>

<b>总结</b>

1. 基于 Restful 制作表现层接口
    - 新增：POST
    - 删除：DELETE
    - 修改：PUT
    - 查询：GET
2. 接收参数
    - 实体数据：@RequestBody
    - 路径变量：@PathVariable



#### 7.表现层消息一致性处理

目前我们通过 Postman 测试后业务层接口功能是通的，但是这样的结果给到前端开发者会出现一个小问题。不同的操作结果所展示的数据格式差异化严重。

<b>增删改操作结果</b>

```tex
true
```

<b>查询单个数据操作结果</b>

```json
{
    "id": 1,
    "type": "计算机理论",
    "name": "Spring实战 第5版",
    "description": "Spring入门经典教程"
}
```

<b>查询全部数据操作结果</b>

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

每种不同操作返回的数据格式都不一样，而且还不知道以后还会有什么格式，这样的结果让前端人员看了是很容易让人崩溃的，必须将所有操作的操作结果数据格式统一起来，需要设计表现层返回结果的模型类，用于后端与前端进行数据格式统一，也称为<b>前后端数据协议</b>

```JAVA
@Data
public class R {
    private Boolean flag;
    private Object data;
}
```

其中 flag 用于标识操作是否成功，data 用于封装操作数据，现在的数据格式就变了

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

表现层开发格式也需要转换一下

<div align="center"><img src="img/boot/image-20211129153301262.png" alt="image-20211129153301262" style="zoom:80%;" /></div>

<div align="center"><img src="img/boot/image-20211129153319052.png" alt="image-20211129153319052" style="zoom:80%;" /></div>

<div align="center"><img src="img/boot/image-20211129153332499.png" alt="image-20211129153332499" style="zoom:80%;" /></div>

<b>总结</b>

1. 设计统一的返回值结果类型便于前端开发读取数据

2. 返回值结果类型可以根据需求自行设定，没有固定格式

3. 返回值结果模型类用于后端与前端进行数据格式统一，也称为前后端数据协议


#### 8.前后端联通性测试

后端的表现层接口开发完毕，就可以进行前端的开发了。

将前端人员开发的页面保存到 resources 目录下的 static 目录中，建议执行 maven 的 clean 生命周期，避免缓存的问题出现。

<div align="center"><img src="img/boot/image-20211129153831990.png"></div>

在进行具体的功能开发之前，先做联通性的测试，通过页面发送异步提交（axios），这一步调试通过后再进行进一步的功能开发。

```js
//列表
getAll() {
	axios.get("/books").then((res)=>{
		console.log(res.data);
	});
},
```

只要后台代码能够正常工作，前端能够在日志中接收到数据，就证明前后端是通的，也就可以进行下一步的功能开发了。

<b>总结</b>

1. 单体项目中页面放置在 resources/static 目录下
2. created 钩子函数用于初始化页面时发起调用
3. 页面使用 axios 发送异步请求获取数据后确认前后端是否联通

#### 9.页面基础功能开发

##### 	F-1.列表功能（非分页版）

列表功能主要操作就是加载完数据，将数据展示到页面上，此处要利用 VUE 的数据模型绑定，发送请求得到数据，然后页面上读取指定数据即可。

<b>页面数据模型定义</b>

```js
data:{
	dataList: [],		//当前页要展示的列表数据
	...
},
```

异步请求获取数据

```JS
//列表
getAll() {
    axios.get("/books").then((res)=>{
        this.dataList = res.data.data;
    });
},
```

这样在页面加载时就可以获取到数据，并且由 VUE 将数据展示到页面上了。

<b>总结</b>：将查询数据返回到页面，利用前端数据绑定进行数据展示

##### 	F-2.添加功能

添加功能用于收集数据的表单是通过一个弹窗展示的，因此在添加操作前首先要进行弹窗的展示，添加后隐藏弹窗即可。因为这个弹窗一直存在，因此当页面加载时首先设置这个弹窗为不可显示状态，需要展示，切换状态即可。

<b>默认状态</b>

```js
data:{
	dialogFormVisible: false,	//添加表单是否可见
	...
},
```

<b>切换为显示状态</b>

```JS
//弹出添加窗口
handleCreate() {
	this.dialogFormVisible = true;
},
```

由于每次添加数据都是使用同一个弹窗录入数据，所以每次操作的痕迹将在下一次操作时展示出来，需要在每次操作之前清理掉上次操作的痕迹。

<b>定义清理数据操作</b>

```js
//重置表单
resetForm() {
    this.formData = {};
},
```

<b>切换弹窗状态时清理数据</b>

```js
//弹出添加窗口
handleCreate() {
    this.dialogFormVisible = true;
    this.resetForm();
},
```

至此准备工作完成，下面就要调用后台完成添加操作了。

<b>添加操作</b>

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

1. 将要保存的数据传递到后台，通过 post 请求的第二个参数传递 json 数据到后台
2. 根据返回的操作结果决定下一步操作
    - 如何是 true 就关闭添加窗口，显示添加成功的消息
    - 如果是 false 保留添加窗口，显示添加失败的消息
3. 无论添加是否成功，页面均进行刷新，动态加载数据（对 getAll 操作发起调用）

<b>取消添加操作</b>

```JS
//取消
cancel(){
    this.dialogFormVisible = false;
    this.$message.info("操作取消");
},
```

<b>总结</b>

1. 请求方式使用 POST 调用后台对应操作
2. 添加操作结束后动态刷新页面加载数据
3. 根据操作结果不同，显示对应的提示信息
4. 弹出添加 Div 时清除表单数据

##### 	F-3.删除功能

模仿添加操作制作删除功能，差别之处在于删除操作仅传递一个待删除的数据 id 到后台即可。

<b>删除操作</b>

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

<b>删除操作提示信息</b>

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

<b>总结</b>

1. 请求方式使用 Delete 调用后台对应操作
2. 删除操作需要传递当前行数据对应的id值到后台
3. 删除操作结束后动态刷新页面加载数据
4. 根据操作结果不同，显示对应的提示信息
5. 删除操作前弹出提示框避免误操作

##### 	F-4.修改功能

修改功能可以说是列表功能、删除功能与添加功能的合体。几个相似点如下：

1. 页面也需要有一个弹窗用来加载修改的数据，这一点与添加相同，都是要弹窗
2. 弹出窗口中要加载待修改的数据，而数据需要通过查询得到，这一点与查询全部相同，都是要查数据
3. 查询操作需要将要修改的数据 id 发送到后台，这一点与删除相同，都是传递 id 到后台
4. 查询得到数据后需要展示到弹窗中，这一点与查询全部相同，都是要通过数据模型绑定展示数据
5. 修改数据时需要将被修改的数据传递到后台，这一点与添加相同，都是要传递数据

所以整体上来看，修改功能就是前面几个功能的大合体

<b>查询并展示数据</b>

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

<b>修改操作</b>

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

<b>总结</b>

1. 加载要修改数据通过传递当前行数据对应的id值到后台查询数据（同删除与查询全部）
2. 利用前端双向数据绑定将查询到的数据进行回显（同查询全部）
3. 请求方式使用 PUT 调用后台对应操作（同新增传递数据）
4. 修改操作结束后动态刷新页面加载数据（同新增）
5. 根据操作结果不同，显示对应的提示信息（同新增）

#### 10.业务消息一致性处理

目前的功能制作基本上达成了正常使用的情况，后台手工抛出一个异常，看看前端接收到的数据什么样子。

```json
{
    "timestamp": "2021-09-15T03:27:31.038+00:00",
    "status": 500,
    "error": "Internal Server Error",
    "path": "/books"
}
```

面对这种情况，前端的同学又不会了，这又是什么格式？怎么和之前的格式不一样？

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

看来不仅要对正确的操作数据格式做处理，还要对错误的操作数据格式做同样的格式处理。

首先在当前的数据结果中添加消息字段，用来兼容后台出现的操作消息。

```JAVA
@Data
public class R{
    private Boolean flag;
    private Object data;
    private String msg;		//用于封装消息
}
```

后台代码也要根据情况做处理，当前是模拟的错误。

```JAVA
@PostMapping
public R save(@RequestBody Book book) throws IOException {
    Boolean flag = bookService.insert(book);
    return new R(flag , flag ? "添加成功^_^" : "添加失败-_-!");
}
```

然后在表现层做统一的异常处理，使用 Spring MVC 提供的异常处理器做统一的异常处理。

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

页面上得到数据后，先判定是否有后台传递过来的消息，标志就是当前操作是否成功，如果返回操作结果 false，就读取后台传递的消息。

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

<b>总结</b>

1. 使用注解 @RestControllerAdvice 定义 SpringMVC 异常处理器用来处理异常的
2. 异常处理器必须被扫描加载，否则无法生效
3. 表现层返回结果的模型类中添加消息属性用来传递消息到页面

#### 11.页面功能开发

##### 	F-5.分页功能

分页功能的制作用于替换前面的查询全部，其中要使用到 elementUI 提供的分页组件。

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

为了配合分页组件，封装分页对应的数据模型。

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

修改查询全部功能为分页查询，通过路径变量传递页码信息参数。

```JS
getAll() {
    axios.get("/books/"+this.pagination.currentPage+"/"+this.pagination.pageSize).then((res) => {
    });
},
```

后台提供对应的分页功能。

```JAVA
@GetMapping("/{currentPage}/{pageSize}")
public R getAll(@PathVariable Integer currentPage,@PathVariable Integer pageSize){
    IPage<Book> pageBook = bookService.getPage(currentPage, pageSize);
    return new R(null != pageBook ,pageBook);
}
```

页面根据分页操作结果读取对应数据，并进行数据模型绑定。

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

对切换页码操作设置调用当前分页操作。

```JS
//切换页码
handleCurrentChange(currentPage) {
    this.pagination.currentPage = currentPage;
    this.getAll();
},
```

<b>总结</b>

1. 使用 el 分页组件
2. 定义分页组件绑定的数据模型
3. 异步调用获取分页数据
4. 分页数据页面回显

##### 	F-6.删除功能维护

由于使用了分页功能，当最后一页只有一条数据时，删除操作就会出现 BUG，最后一页无数据但是独立展示，对分页查询功能进行后台功能维护，如果当前页码值大于最大页码值，重新执行查询。其实这个问题解决方案很多，这里给出比较简单的一种处理方案。

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

最后一个功能来做条件查询，其实条件查询可以理解为分页查询的时候除了携带分页数据再多带几个数据的查询。这些多带的数据就是查询条件。比较一下不带条件的分页查询与带条件的分页查询差别之处，这个功能就好做了

- 页面封装的数据：带不带条件影响的仅仅是一次性传递到后台的数据总量，由传递 2 个分页相关数据转换成 2 个分页数据加若干个条件

- 后台查询功能：查询时由不带条件，转换成带条件，反正不带条件的时候查询条件对象使用的是 null，现在换成具体条件，差别不大

- 查询结果：不管带不带条件，出来的数据只是有数量上的差别，其他都差别，这个可以忽略

    经过上述分析，看来需要在页面发送请求的格式方面做一定的修改，后台的调用数据层操作时发送修改，其他没有区别。

    页面发送请求时，两个分页数据仍然使用路径变量，其他条件采用动态拼装 url 参数的形式传递。

<b>页面封装查询条件字段</b>

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

将查询条件组织成 url 参数，添加到请求 url 地址中，这里可以借助其他类库快速开发，当前使用手工形式拼接，降低学习要求

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

<b>总结</b>

1. 定义查询条件数据模型（当前封装到分页数据模型中）
2. 异步调用分页功能并通过请求参数传递数据到后台

# 运维实用篇

实用篇共分为两块内容，分别是运维实用篇和开发实用篇。

运维实用篇的定位是玩转配置，为开发实用篇中做各种技术的整合做好准备工作。与开发实用篇相比，运维实用篇的内容显得略微单薄，并且有部分知识模块在运维实用篇和开发实用篇中都要讲一部分，这些内容都后置到开发实用篇中了。运维实用篇中都包含的内容：

- Spring Boot 程序的打包与运行
- 配置高级
- 多环境开发
- 日志

## 打包与运行

1. 开发部门使用 Git、SVN 等版本控制工具上传工程到版本服务器
2. 服务器使用版本控制工具下载工程
3. 服务器上使用 Maven 工具在当前真机环境下重新构建项目
4. 启动服务

打包是指将程序转换成一个可执行的文件，所谓运行指不依赖开发环境执行打包产生的文件。上述两个操作都有对应的命令可以快速执行。

### 程序打包

Spring Boot 程序是基于 Maven 创建的，在 Maven 中提供有打包的指令，叫做 package。本操作可以在 Idea 环境下执行。

```JAVA
mvn package
```

打包后会产生一个与工程名类似的 jar 文件，其名称是由模块名+版本号+.jar 组成的。

### 程序运行

程序包打好以后，就可以直接执行了。在程序包所在路径下，执行指令。

```JAVA
java -jar 工程包名.jar
```

执行程序打包指令后，程序正常运行，与在 Idea 下执行程序没有区别。

<font color="#ff0000"><b>特别关注</b></font>：如果你的计算机中没有安装 java 的 jdk 环境，是无法正确执行上述操作的，因为程序执行使用的是 java 指令。

<font color="#ff0000"><b>特别关注</b></font>：在使用向导创建 Spring Boot 工程时，pom.xml 文件中会有如下配置，这一段配置千万不能删除，否则打包后无法正常执行程序。

```XML
<build>
    <plugins>
        <plugin>
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

<b>总结</b>

1. Spring Boot 工程可以基于 java 环境下独立运行 jar 文件启动服务
2. Spring Boot 工程执行 mvn 命令 package 进行打包
3. 执行 jar 命令：java –jar 工程名.jar

### 打包失败处理

打包以后执行会出现一些问题，导致程序无法正常执行，例如下面的现象

<div align="center"><img src="img/boot/image-20211201094223991.png" alt="image-20211201094223991" style="zoom:67%;" /></div>

要想搞清楚这个问题就要说说 .jar 文件的工作机制了，知道了这个东西就知道如何避免此类问题的发生了。

搞 java 开发平时会接触很多 jar 包，比如 mysql 的驱动 jar 包，而上面我们打包程序后得到的也是一个 jar 文件。这个时候如果你使用上面的 java -jar 指令去执行 mysql 的驱动 jar 包就会出现上述不可执行的现象，而我们的 Spring Boot 项目为什么能执行呢？其实是因为打包方式不一样。

在 Spring Boot 工程的 pom.xml 中有下面这组配置，这组配置决定了打包出来的程序包是否可以执行。

```XML
<build>
    <plugins>
        <plugin>
            <groupId>org.Springframework.boot</groupId>
            <artifactId>Spring-boot-maven-plugin</artifactId>
        </plugin>
    </plugins>
</build>
```

我们分别开启这段配置和注释掉这段配置分别执行两次打包，然后观察两次打包后的程序包的差别，共有 3 处比较明显的特征

- 打包后文件的大小不同
- 打包后所包含的内容不同
- 打包程序中个别文件内容不同

先看第一个现象，文件大小不同。带有配置时打包生成的程序包大小如下：

<div align="center"><img src="img/boot/image-20211201095610270.png" alt="image-20211201095610270" style="zoom:150%;" /></div>

不难看出，带有配置的程序包体积比不带配置的大了 30 倍。

大的程序包中除了包含小的程序包中的内容，还有别的东西。都有什么呢？回到 BOOT-INF 目录下，打开 lib 目录，里面显示了很多个 jar 文件。

<div align="center"><img src="img/boot/image-20211201102025791.png" alt="image-20211201102025791" style="zoom:50%;" /></div>

仔细翻阅不难发现，这些 jar 文件都是我们制作这个工程时导入的坐标对应的文件。大概可以想明白了，Spring Boot 程序为了让自己打包生成的程序可以独立运行，不仅将项目中自己开发的内容进行了打包，还把当前工程运行需要使用的 jar 包全部打包进来了。为什么这样做呢？就是为了可以独立运行。不依赖程序包外部的任何资源可以独立运行当前程序。这也是为什么大的程序包容量是小的程序包容量的 30 倍的主要原因。

再看看大程序包还有什么不同之处，在最外层目录包含一个 org 目录，进入此目录，目录名是 org\Springframework\boot\loader，在里面可以找到一个 <font color="#ff0000"><b>JarLauncher.class</b></font> 的文件，先记得这个文件。再看这套目录名，明显是一个 Spring 的目录名，为什么要把 Spring 框架的东西打包到这个程序包中呢？不清楚。

回到两个程序包的最外层目录，查看名称相同的文件夹 META-INF 下都有一个叫做 MANIFEST.MF 的文件，但是大小不同，打开文件，比较内容区别

小容量文件的 MANIFEST.MF

```shell
Manifest-Version: 1.0
Implementation-Title: Springboot_08_ssmp
Implementation-Version: 0.0.1-SNAPSHOT
Build-Jdk-Spec: 1.8
Created-By: Maven Jar Plugin 3.2.0
```

大容量文件的 MANIFEST.MF

```shell
Manifest-Version: 1.0
Spring-Boot-Classpath-Index: BOOT-INF/classpath.idx
Implementation-Title: Springboot_08_ssmp
Implementation-Version: 0.0.1-SNAPSHOT
Spring-Boot-Layers-Index: BOOT-INF/layers.idx
Start-Class: com.itheima.SSMPApplication
Spring-Boot-Classes: BOOT-INF/classes/
Spring-Boot-Lib: BOOT-INF/lib/
Build-Jdk-Spec: 1.8
Spring-Boot-Version: 2.5.4
Created-By: Maven Jar Plugin 3.2.0
Main-Class: org.Springframework.boot.loader.JarLauncher
```

大文件中明显比小文件中多了几行信息，其中最后一行信息是 Main-Class: org.Springframework.boot.loader.<font color="#ff0000"><b>JarLauncher</b></font>。这句话什么意思呢？如果使用 java -jar 执行此程序包，将执行 Main-Class 属性配置的类，这个类恰巧就是前面看到的那个文件。原来 Spring Boot 打包程序中出现 Spring 框架的东西是为这里服务的。而这个 org.Springframework.boot.loader.<font color="#ff0000"><b>JarLauncher</b></font> 类内部要查找 Start-Class 属性中配置的类，并执行对应的类。这个属性在当前配置中也存在，对应的就是我们的引导类类名。

现在这组设定的作用就搞清楚了

1. Spring Boot 程序添加配置后会打出一个特殊的包，包含 Spring 框架部分功能，原始工程内容，原始工程依赖的 jar 包
2. 首先读取 MANIFEST.MF 文件中的 Main-Class 属性，用来标记执行 java -jar 命令后运行的类
3. JarLauncher 类执行时会找到 Start-Class 属性，也就是启动类类名
4. 运行启动类时会运行当前工程的内容
5. 运行当前工程时会使用依赖的 jar 包，从 lib 目录中查找

Spring Boot 打出来了包为了能够独立运行，将所有需要使用的资源全部都添加到了这个包里。这就是为什么这个 jar 包能独立运行的原因。

再来看之前的报错信息：

<div align="center"><img src="img/boot/image-20211201094223991.png" alt="image-20211201094223991" style="zoom:67%;" /></div>

由于打包时没有使用那段配置，结果打包后形成了一个普通的 jar 包，在 MANIFEST.MF 文件中也就没有了 Main-Class 对应的属性了，所以运行时提示找不到主清单属性，这就是报错的原因。

上述内容搞清楚对我们编程意义并不大，但是对理清楚 Spring Boot 工程独立运行的机制是有帮助的。

<b>总结：</b>Spring-boot-maven-plugin 插件用于将当前程序打包成一个可以独立运行的程序包

### 命令行启动常见问题及解决方案

在 DOS 环境下启动 Spring Boot 工程时，可能会遇到端口占用的问题。

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

关于打包与运行程序其实还有一系列的配置和参数，下面的内容中遇到再说。

### Linux下快速启动

- `nohub java -jar xxx.jar > server.log`   后台启动 jar，日志保存在 server.log 里
- `ps -ef | grep "java -jar"` 搜索包含 “java -jar” 的命令
- `kill -9 pid编号` 杀死进程

也可以使用 tmux 命令。

## 配置高级

### 临时属性设置

程序包打好以后，里面的配置都已经是固定的了，比如配置了服务器的端口是 8080。如果我要启动项目，发现当前我的服务器上已经有应用启动起来并且占用了 8080 端口，这时候可以通过临时属性的方式快速修改某些配置。

<div align="center">
    <img src="img/boot/image-20211206095113771.png"><img src="img/boot/image-20211206095524343.png"><img src="img/boot/image-20211206095101581.png">
</div>
```shell
java –jar Spring Boot.jar –-server.port=80
```

在命令输入完毕后，空一格，然后输入两个-号。下面按照属性名=属性值的形式添加对应参数就可以了。当属性存在多级名称时，中间使用点分隔，和 properties 文件中的属性格式完全相同。

如果你发现要修改的属性不止一个，可以按照上述格式继续写，属性与属性之间使用空格分隔。

```shell
java –jar Spring Boot.jar –-server.port=80 --logging.level.root=debug
```

#### 属性加载优先级

现在我们的程序配置受两个地方控制了，第一配置文件，第二临时属性。并且我们发现临时属性的加载优先级要高于配置文件的。那是否还有其他的配置方式呢？其实是有的，而且还不少，打开官方文档中对应的内容，就可以查看配置读取的优先顺序：[https://docs.Spring.io/Spring-boot/docs/current/reference/html/Spring-boot-features.html#boot-features-external-config](https://docs.Spring.io/Spring-boot/docs/current/reference/html/Spring-boot-features.html)

<div align="center"><img src="img/boot/image-20211206100859236.png" alt="image-20211206100859236"></div>

我们可以看到，有 14 种配置的位置，而我们现在使用的是这里面的 2 个。第 3 条 Config data 说的就是使用配置文件，第 11 条 Command line arguments 说的就是使用命令行临时参数。而这 14 种配置的顺序就是 Spring Boot 加载配置的顺序，言外之意，命令行临时属性比配置文件的加载优先级高，所以这个列表<b>上面的优先级低，下面的优先级高</b>。其实这个东西不用背的，我们最终要什么效果，自己是知道的，不管这个顺序是怎么个高低排序，开发时一定要配置成你要的顺序为准。这个顺序只是在你想不明白问题的时候帮助你分析罢了。

比如你现在加载了一个 user.name 属性。结果你发现出来的结果和你想的不一样，那肯定是别的优先级比你高的属性覆盖你的配置属性了，那你就可以看着这个顺序挨个排查。哪个位置有可能覆盖了你的属性。

在 yaml 中配置了 user.name 属性值，然后读取出来的时候居然不是自己的配置值，因为在系统属性中有一个属性叫做 user.name，两个相互冲突了。而系统属性的加载优先顺序在上面这个列表中是 5 号，高于 3 号，所以 Spring Boot 最终会加载系统配置属性 user.name。

<b>总结</b>

1. 使用 jar 命令启动 Spring Boot 工程时可以使用临时属性替换配置文件中的属性
2. 临时属性添加方式：java –jar 工程名.jar –-属性名=值
3. 多个临时属性之间使用空格分隔
4. 临时属性必须是当前 boot 工程支持的属性，否则设置无效

#### 开发环境中使用临时属性

开发环境中如何使用临时属性，其实就是 Idea 界面下如何操作了。

打开 Spring Boot 引导类的运行界面，在里面找到配置项。其中 Program arguments 对应的位置就是添加临时属性的，可以加几个试试效果。

<div align="center"><img src="img/boot/image-20211206101947622.png" alt="image-20211206101947622" style="zoom:80%;" /></div>

我们运行 main 方法的时候，如果想使用 main 方法的参数，也就是下面的 args 参数，就是在上面这个位置添加的参数。

```JAVA
public static void main(String[] args) {}
```

通过这个 args 就可以获取到参数。再来看我们的引导类是如何书写的

```JAVA
public static void main(String[] args) {
    SpringApplication.run(SSMPApplication.class,args);
}
```

这个 args 参数居然传递给了 run 方法，看来在 Idea 中配置的临时参数就是通过这个位置传递到我们的程序中的。言外之意，这里如果不用这个 args 是不是就断开了外部传递临时属性的入口呢？是这样的，我们可以使用下面的调用方式，这样外部临时属性就无法进入到 Spring Boot 程序中了。

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

<b>总结</b>：启动 Spring Boot 程序时，可以选择是否使用命令行属性为 Spring Boot 程序传递启动属性

<b>思考</b>：现在使用临时属性可以在启动项目前临时更改配置了，如果需要变更的属性偏多，如何处理？

### 配置文件分类

Spring Boot 提供了配置文件和临时属性的方式来对程序进行配置。前面一直说的是临时属性，这一节要说说配置文件了。其实这个配置文件我们一直在使用，只不过我们用的是 Spring Boot 提供的 4 级配置文件中的其中一个级别。4 个级别分别是：

- 类路径下配置文件（一直使用的是这个，也就是 resources 目录中的 application.yml 文件）
- 类路径下 config 目录下配置文件
- 程序包所在目录中配置文件
- 程序包所在目录中 config 目录下配置文件

其实上述 4 种文件是提供给你了 4 种配置文件书写的位置，功能都是一样的，都是做配置的，就是优先级顺序不一样。上面 4 个文件的加载优先顺序为

- 1、file ：config/application.yml <b>【最高】</b>
- 2、file ：application.yml
- 3、classpath：config/application.yml
- 4、classpath：application.yml  <b>【最低】</b>

那为什么设计这种多种呢？说一个最典型的应用吧。

- 场景 A：你作为一个开发者，你做程序的时候为了方便自己写代码，配置的数据库肯定是连接你自己本机的，咱们使用 4 这个级别，也就是之前一直用的 application.yml。
- 场景 B：现在项目开发到了一个阶段，要联调测试了，连接的数据库是测试服务器的数据库，肯定要换一组配置吧。你可以选择把你之前的文件中的内容都改了，目前还不麻烦。
- 场景 C：测试完了，一切 OK。你继续写你的代码，你发现你原来写的配置文件被改成测试服务器的内容了，你要再改回来。现在明白了不？场景 B 中把你的内容都改掉了，你现在要重新改回来，以后呢？改来改去吗？

解决方案很简单，用上面的 3 这个级别的配置文件就可以快速解决这个问题，再写一个配置就行了。两个配置文件共存，因为 config 目录中的配置加载优先级比你的高，所以配置项如果和级别 4 里面的内容相同就覆盖了。级别 1 和 2 什么时候使用呢？程序打包以后就要用这个级别了，管你程序里面配置写的是什么？我的级别高，可以轻松覆盖你，就不用考虑这些配置冲突的问题了。

<b>总结</b>

配置文件分为 4 种

- 项目类路径配置文件：服务于开发人员本机开发与测试
- 项目类路径 config 目录中配置文件：服务于项目经理整体调控
- 工程路径配置文件：服务于运维人员配置涉密线上环境
- 工程路径 config 目录中配置文件：服务于运维经理整体调控

多层级配置文件间的属性采用叠加并覆盖的形式作用于程序。


### 自定义配置文件

之前咱们做配置使用的配置文件都是 application.yml，其实这个文件也是可以改名字的，这样方便维护。比如 2020 年 4 月 1 日搞活动，走了一组配置，2020 年 5 月 1 日活动取消，恢复原始配置，这个时候只需要重新更换一下配置文件就可以了。但是我们总不能在原始配置文件上修改吧，不然搞完活动以后，活动的配置就留不下来了，不利于维护。

自定义配置文件方式有如下两种：

<b>方式一：使用临时属性设置配置文件名，注意仅仅是名称，不要带扩展名</b>

<div align="center"><img src="img/boot/image-20211206105548238.png" alt="image-20211206105548238"/></div>

<b>方式二：使用临时属性设置配置文件路径，这个是全路径名</b>

<div align="center"><img src="img/boot/image-20211206105716450.png" alt="image-20211206105716450"/></div>

也可以设置加载多个配置文件

<div align="center"><img src="img/boot/image-20211206105750285.png" alt="image-20211206105750285" /></div>

使用的属性一个是 Spring.config.name，另一个是 Spring.config.location，这个一定要区别清楚。

<font color="#f0f"><b>温馨提示</b></font>
我们现在研究的都是 Spring Boot 单体项目，就是单服务器版本。其实企业开发现在更多的是使用基于 SpringCloud 技术的多服务器项目。这种配置方式和我们现在学习的完全不一样，所有的服务器将不再设置自己的配置文件，而是通过配置中心获取配置，动态加载配置信息。集中管理更为方便。

<b>总结</b>

1. 配置文件可以修改名称，通过启动参数设定
2. 配置文件可以修改路径，通过启动参数设定
3. 微服务开发中配置文件通过配置中心进行设置

## 多环境开发

什么是多环境？其实就是说你的电脑上写的程序最终要放到别人的服务器上去运行。每个计算机环境不一样，这就是多环境。常见的多环境开发主要兼顾3种环境设置，开发环境——自己用的，测试环境——自己公司用的，生产环境——甲方爸爸用的。因为这是绝对不同的三台电脑，所以环境肯定有所不同，比如连接的数据库不一样，设置的访问端口不一样等等。

<div align="center"><img src="img/boot/image-20211206110958819.png" alt="image-20211206110958819"/></div>

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
Spring:
	profiles: pro
server:
	port: 80
---
Spring:
	profiles: dev
server:
	port: 81
```

那用哪一个呢？设置默认启动哪个就可以了

```yaml
Spring:
	profiles:
		active: pro		# 启动pro
---
Spring:
	profiles: pro
server:
	port: 80
---
Spring:
	profiles: dev
server:
	port: 81
```

就这么简单，再多来一组环境也 OK

```yaml
Spring:
	profiles:
		active: pro		# 启动pro
---
Spring:
	profiles: pro
server:
	port: 80
---
Spring:
	profiles: dev
server:
	port: 81
---
Spring:
	profiles: test
server:
	port: 82
```

其中关于环境名称定义上述格式是过时格式，<font color="#f0f"><b>标准格式如下</b></font>

```yaml
Spring:
	config:
    	activate:
        	on-profile: pro
```

<b>总结</b>

1. 多环境开发需要设置若干种常用环境，例如开发、生产、测试环境
2. yaml 格式中设置多环境使用---区分环境设置边界
3. 每种环境的区别在于加载的配置属性不同
4. 启用某种环境时需要指定启动时使用该环境

### 多环境开发（yaml多文件版）	

将所有的配置都放在一个配置文件中，尤其是每一个配置应用场景都不一样，这显然不合理，于是就有了将一个配置文件拆分成多个配置文件的想法。拆分后，每个配置文件中写自己的配置，主配置文件中写清楚用哪一个配置文件就好了。

<b>主配置文件</b>

```yaml
Spring:
	profiles:
		active: pro		# 启动pro(active: pro 选择的配置文件的名称是 application-pro.yml)
```

<b>环境配置文件</b>

```yaml
server:
	port: 80
```

环境配置文件因为每一个都是配置自己的项，所以连名字都不用写里面了。那问题是如何区分这是哪一组配置呢？使用文件名区分。

<b>application-pro.yaml</b>

```yaml
server:
	port: 80
```

<b>application-dev.yaml</b>

```yaml
server:
	port: 81
```

文件的命名规则为：application-环境名.yml。

在配置文件中，如果某些配置项所有环境都一样，可以将这些项写入到主配置中，只有哪些有区别的项才写入到环境配置文件中。

- 主配置文件中设置公共配置（全局）
- 环境分类配置文件中常用于设置冲突属性（局部）

<b>总结</b>

1. 可以使用独立配置文件定义环境属性

2. 独立配置文件便于线上系统维护更新并保障系统安全性


### 多环境开发（properties多文件版）

Spring Boot 最早期提供的配置文件格式是 properties 格式的，这种格式的多环境配置也了解一下。

<b>主配置文件</b>

```properties
Spring.profiles.active=pro
```

<b>环境配置文件</b>

<b>application-pro.properties</b>

```properties
server.port=80
```

<b>application-dev.properties</b>

```properties
server.port=81
```

文件的命名规则为：application-环境名.properties。

<b>总结</b>：properties 文件多环境配置仅支持多文件格式

### 多环境开发独立配置文件书写技巧

作为程序员在搞配置的时候往往处于一种分久必合合久必分的局面。开始先写一起，后来为了方便维护就拆分。对于多环境开发也是如此，下面给大家说一下如何基于多环境开发做配置独立管理，务必掌握。

<b>准备工作</b>：将所有的配置根据功能对配置文件中的信息进行拆分，并制作成独立的配置文件，命名规则如下

- application-devDB.yml
- application-devRedis.yml
- application-devMVC.yml

<b>使用</b>：使用 include 属性在激活指定环境的情况下，同时对多个环境进行加载使其生效，多个环境间使用逗号分隔

```yaml
Spring:
	profiles:
    	active: dev
        include: devDB,devRedis,devMVC
```

比较一下，现在相当于加载 dev 配置时，再加载对应的3组配置，从结构上就很清晰，用了什么，对应的名称是什么

<b>注意</b>：当主环境 dev 与其他环境有相同属性时，主环境属性生效；其他环境中有相同属性时，最后加载的环境属性生效

<b>改良</b>：但是上/面的设置也有一个问题，比如我要切换 dev 环境为 pro 时，include 也要修改。因为 include 属性只能使用一次，这就比较麻烦了。Spring Boot 从 2.4 版开始使用 group 属性替代 include 属性，降低了配置书写量。简单说就是我先写好，你爱用哪个用哪个。

```yaml
Spring:
	profiles:
    	active: dev
        group:
        	"dev": devDB,devRedis,devMVC
      		"pro": proDB,proRedis,proMVC
      		"test": testDB,testRedis,testMVC
```

现在再来看，如果切换 dev 到 pro，只需要改一下是不是就结束了？完美！

<b>总结</b>：<span style="color:red"><b>多环境开发使用group属性设置配置文件分组，便于线上维护管理</b></span>

### 多环境开发控制

多环境开发到这里基本上说完了，最后说一个冲突问题。就是 maven 和 Spring Boot 同时设置多环境的话怎么搞。

要想处理这个冲突问题，你要先理清一个关系，究竟谁在多环境开发中其主导地位。也就是说如果现在都设置了多环境，谁的应该是保留下来的，另一个应该遵从相同的设置。

maven 是做什么的？项目构建管理的，最终生成代码包的，Spring Boot 是干什么的？简化开发的。简化，又不是其主导作用。最终还是要靠 maven 来管理整个工程，所以 Spring Boot 应该听 maven 的。整个确认后下面就好做了。大体思想如下：

- 先在 maven 环境中设置用什么具体的环境
- 在 Spring Boot 中读取 maven 设置的环境即可

<b>maven 中设置多环境（使用属性方式区分环境）</b>

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

<b>SpringBoot 中读取 maven 设置值</b>

```yaml
Spring:
	profiles:
    	active: @profile.active@
```

上面的@属性名@就是读取 maven 中配置的属性值的语法格式。

<b>总结</b>

1. 当 Maven 与 Spring Boot 同时对多环境进行控制时，以 Mavn 为主，Spring Boot 使用@..@占位符读取 Maven 对应的配置属性值
2. 基于 Spring Boot 读取 Maven 配置属性的前提下，如果在 Idea 下测试工程时 pom.xml 每次更新需要手动 compile 方可生效

## 日志

- 编程期调试代码
- 运营期记录信息
    - 记录日常运营重要信息（峰值流量、平均响应时长……）
    - 记录应用报错信息（错误堆栈）
    - 记录运维过程数据（扩容、宕机、报警……）

市面上的日志框架有很多：JUL、JCL、Jboss-logging、logback、log4j、log4j2、slf4j....

| 日志门面（日志的抽象层）                             | 日志实现                       |
| ---------------------------------------------------- | ------------------------------ |
| JCL（Jakarta Commons Logging） SLF4j（Simple Logging | Log4j JUL（java.util.logging） |

左边选一个门面（抽象层）、右边来选一个实现；

日志门面： SLF4J；

日志实现：Logback；

Spring Boot：底层是 Spring 框架，Spring 框架默认是用 JCL；Spring Boot 选用SLF4j 和 logback； 


### 代码中使用日志工具记录日志

日志的使用格式非常固定，直接上操作步骤：

<b>步骤①</b>：添加日志记录操作

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
        return "Spring Boot is running...2";
    }
}
```

上述代码中 log 对象就是用来记录日志的对象，下面的 log.debug，log.info 这些操作就是写日志的 API了。

<b>步骤②</b>：设置日志输出级别

日志设置好以后可以根据设置选择哪些参与记录。这里是根据日志的级别来设置的。日志的级别分为 6 种，分别是：

- TRACE：运行堆栈信息，使用率低
- DEBUG：程序员调试代码使用
- INFO：记录运维过程数据
- WARN：记录运维过程报警数据
- ERROR：记录错误堆栈信息
- FATAL：灾难信息，合并计入 ERROR

一般情况下，开发时候使用 DEBUG，上线后使用 INFO，运维信息记录使用 WARN 即可。下面就设置一下日志级别：

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

<b>步骤③</b>：<span style="color:red"><b>设置日志组，控制指定包对应的日志输出级别，也可以直接控制指定包对应的日志输出级别，推荐使用</b></span>

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

<b>总结</b>

1. 日志用于记录开发调试与运维过程消息
2. 日志的级别共 6 种，通常使用 4 种即可，分别是 DEBUG，INFO，WARN，ERROR
3. 可以通过日志组或代码包的形式进行日志显示级别的控制

### 优化日志对象创建代码

写代码的时候每个类都要写创建日志记录对象，这个可以优化一下，使用前面用过的 lombok 技术给我们提供的工具类即可。

```JAVA
@RestController
@RequestMapping("/books")
public class BookController extends BaseClass{
    private static final Logger log = LoggerFactory.getLogger(BookController.class);	//这一句可以不写了
}
```

导入 lombok 后使用注解搞定，日志对象名为 log

```JAVA
@Slf4j		//这个注解替代了下面那一行
@RestController
@RequestMapping("/books")
public class BookController extends BaseClass{
    private static final Logger log = LoggerFactory.getLogger(BookController.class);	//这一句可以不写了
}
```

<b>总结</b>：基于 lombok 提供的 `@Slf4j` 注解为类快速添加日志对象

### 日志输出格式控制

日志已经能够记录了，但是目前记录的格式是 Spring Boot 给我们提供的，如果想自定义控制就需要自己设置了。先分析一下当前日志的记录格式。

<div align="center"><img src="img/boot/image-20211206123431222.png"></div>

Spring Boot 重写后的信息，名称过长时，简化包名书写为首字母，甚至直接删除

对于单条日志信息来说，日期，触发位置，记录信息是最核心的信息。级别用于做筛选过滤，PID 与线程名用于做精准分析。了解这些信息后就可以 DIY 日志格式了。本课程不做详细的研究，有兴趣的小伙伴可以学习相关的知识。下面给出课程中模拟的官方日志模板的书写格式，便于大家学习。

```yaml
logging:
	pattern:
    	console: "%d %clr(%p) --- [%16t] %clr(%-40.40c){cyan} : %m %n"
```

<b>总结</b>：日志输出格式设置规则 `%d:日期`；`%m:消息`；`%n:换行`；`%clr:彩色`；`[%16t]:线程，占16位`；`%c:类名，-40是左对齐，.40是控制内容的截取`；`{cyan}:青色`

| 格式 | 描述   |
| ---- | ------ |
| %d   | 日期   |
| %m   | 消息   |
| %n   | 换行   |
| %clr | 彩色   |
| %t   | 线程   |
| %c   | 类名   |
| -40  | 左对齐 |

### 日志文件

日志信息显示，记录已经控制住了，下面就要说一下日志的转存了。日志不能仅显示在控制台上，要把日志记录到文件中，方便后期维护查阅。对于日志文件的使用存在各种各样的策略，例如每日记录，分类记录，报警后记录等。这里主要研究日志文件如何记录。记录日志到文件中格式非常简单，设置日志文件名即可。

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

以上格式是基于 logback 日志技术设置每日日志文件的设置格式，要求容量到达 3KB 以后就转存信息到第二个文件中。文件命名规则中的 %d 标识日期，%i 是一个递增变量，用于区分日志文件。

<b>总结</b>：日志记录到文件、日志文件格式设置

| logging.file | logging.path | Example  | Description                        |
| ------------ | ------------ | -------- | ---------------------------------- |
| （none）     | （none）     |          | 只在控制台输出                     |
| 指定文件名   | （none）     | my.log   | 输出日志到 my.log 文件             |
| （none）     | 指定目录     | /var/log | 输出到指定目录的 Spring.log 文件中 |

### 指定配置

给类路径下放上每个日志框架自己的配置文件即可；Spring Boot 就不使用他默认配置的了

| Logging System          | Customization                                                |
| ----------------------- | ------------------------------------------------------------ |
| Logback                 | logback-Spring.xml , logback-Spring.groovy , logback.xml or logback.groovy |
| Log4j2                  | log4j2-Spring.xml or log4j2.xml                              |
| JDK (Java Util Logging) | logging.properties                                           |

logback.xml：直接就被日志框架识别了；

<b>logback-Spring.xml</b>：日志框架就不直接加载日志的配置项，由 Spring Boot 解析日志配置，可以使用

Spring Boot 的高级 Profifile 功能

```xml
<SpringProfile name="staging"> 
    <!-- configuration to be enabled when the "staging" profile is active --> 
    可以指定某段配置只在某个环境下生效 
</SpringProfile>
```

### 遗留问题

a（slf4j+logback）: Spring（commons-logging）、Hibernate（jboss-logging）、MyBatis、xxxx 统一日志记录，即使是别的框架和我一起统一使用 slf4j 进行输出？

如何让系统中所有的日志都统一到 slf4j？

1、将系统中其他日志框架先排除出去；

2、用中间包来替换原有的日志框架；

3、我们导入 slf4j 其他的实现

```xml
<!-- Spring 日志包 -->
<dependency> 
    <groupId>org.Springframework.boot</groupId> 
    <artifactId>Spring-boot-starter-logging</artifactId> 
</dependency>
```

1）、Spring Boot 底层也是使用 slf4j+logback 的方式进行日志记录

2）、Spring Boot 也把其他的日志都替换成了 slf4j； 

3）、中间替换包？

```java
@SuppressWarnings("rawtypes") 
public abstract class LogFactory {
    static String UNSUPPORTED_OPERATION_IN_JCL_OVER_SLF4J = "http://www.slf4j.org/codes.html#unsupported_operation_in_jcl_over_slf4j";
    static LogFactory logFactory = new SLF4JLogFactory(); 
}
```

如果我们要引入其他框架？一定要把这个框架的默认日志依赖移除掉？

Spring 框架用的是 commons-logging；

```xml
<dependency>
	<groupId>org.Springframework</groupId>
    <artifactId>Spring-core</artifactId>
    <exclusions>
    	<exclusion>
        	<groupId>commons-logging</groupId>
    		<artifactId>commons-logging</artifactId>
        </exclusion>
    </exclusions>
</dependency>
```

Spring Boot 能自动适配所有的日志，而且底层使用 slf4j+logback 的方式记录日志，引入其他框架的时候，只需要把这个框架依赖的日志框架排除掉即可

# 开发实用篇

- 热部署
- 配置高级
- 测试
- 数据层解决方案
- 整合第三方技术
- 监控

## 热部署

什么是热部署？简单说就是你程序改了，不用重启，服务器会自己悄悄的把更新后的程序给重新加载一遍，这就是热部署。

热部署的功能是如何实现的呢？这就要分两种情况来说了，非 Spring Boot 工程和 Spring Boot 工程的热部署实现方式完全不一样。先说一下原始的非 Spring Boot 项目是如何实现热部署的。

<b>非 Spring Boot 项目热部署实现原理</b>

开发非 Spring Boot 项目时，我们要制作一个 web 工程并通过 tomcat 启动，通常需要先安装 tomcat 服务器到磁盘中，开发的程序配置发布到安装的 tomcat 服务器上。如果想实现热部署的效果，这种情况其实有两种做法，一种是在 tomcat 服务器的配置文件中进行配置，这种做法与你使用什么 IDE 工具无关，不管你使用 eclipse 还是 idea 都行。还有一种做法是通过 IDE 工具进行配置，比如在 idea 工具中进行设置，这种形式需要依赖 IDE 工具，每款 IDE 工具不同，对应的配置也不太一样。<b>但是核心思想是一样的，就是使用服务器去监控其中加载的应用，发现产生了变化就重新加载一次。</b>

上面所说的非 Spring Boot 项目实现热部署看上去是一个非常简单的过程，最简单的思路如下：例如启动一个定时任务，任务启动时记录每个文件的大小，以后每 5 秒比对一下每个文件的大小是否有改变，或者是否有新文件。如果没有改变，放行，如果有改变，刷新当前记录的文件信息，然后重新启动服务器，这就可以实现热部署了。当然，这个过程肯定不能这么做，比如我把一个打印输出的字符串 "abc" 改成 "cba"，比对大小是没有变化的，但是内容缺实变了，所以这么做肯定不行，而且重启服务器这就是冷启动了，不能算热部署。

看上去这个过程也没多复杂，在 Spring Boot 项目中难道还有其他的弯弯绕吗？还真有。

<b>Springboot项目热部署实现原理</b>

基于 Spring Boot 开发的 web 工程其实有一个显著的特征，就是 tomcat 服务器内置了，还记得内嵌服务器吗？服务器是以一个对象的形式在 Spring 容器中运行的。本来我们期望于 tomcat 服务器加载程序后由 tomcat 服务器盯着程序，你变化后我就重新启动重新加载，但是现在 tomcat 和我们的程序是平级的了，都是 Spring 容器中的组件，这下就麻烦了，缺乏了一个直接的管理权，那该怎么做呢？简单，再搞一个程序 X 在 Spring 容器中盯着你原始开发的程序 A 不就行了吗？确实，搞一个盯着程序 A 的程序 X 就行了，如果你自己开发的程序 A 变化了，那么程序 X 就命令 tomcat 容器重新加载程序 A 就 OK 了。并且这样做有一个好处，Spring 容器中东西不用全部重新加载一遍，只需要重新加载你开发的程序那一部分就可以了，这下效率又高了，挺好。

下面就说说，怎么搞出来这么一个程序 X，肯定不是我们自己手写了，Spring Boot 早就做好了，搞一个坐标导入进去就行了。

### 手动启动热部署

<b>步骤①</b>：导入开发者工具对应的坐标

```XML
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-devtools</artifactId>
    <optional>true</optional>
</dependency>
```

<b>步骤②</b>：构建项目，可以使用快捷键激活此功能

<div align="center"><img src="img/boot/image-20220222121257218.png"></div>

对应的快捷键 `<CTR>L+<F9>`，以上过程就实现了 Spring Boot 工程的热部署。底层的工作过程如下。

<b>重启与重载</b>

一个 Spring Boot 项目在运行时实际上是分两个过程进行的，根据加载的东西不同，划分成 base 类加载器与 restart 类加载器。

- base 类加载器：用来加载 jar 包中的类，jar 包中的类和配置文件由于不会发生变化，因此不管加载多少次，加载的内容不会发生变化
- restart 类加载器：用来加载开发者自己开发的类、配置文件、页面等信息，这一类文件受开发者影响

当 Spring Boot 项目启动时，base 类加载器执行，加载 jar 包中的信息后，restart类加载器执行，加载开发者制作的内容。当执行构建项目后，由于 jar 中的信息不会变化，因此 base 类加载器无需再次执行，所以仅仅运行 restart 类加载即可，也就是将开发者自己制作的内容重新加载就行了，这就完成了一次热部署的过程，也可以说热部署的过程实际上是重新加载 restart 类加载器中的信息。

<b>总结</b>

1. 使用开发者工具可以为当前项目开启热部署功能
2. 使用构建项目操作对工程进行热部署

<b>思考</b>

上述过程每次进行热部署都需要开发者手工操作，不管是点击按钮还是快捷键都需要开发者手工执行。这种操作的应用场景主要是在开发调试期，并且调试的代码处于不同的文件中，比如服务器启动了，我需要改 4 个文件中的内容，然后重启，等 4 个文件都改完了再执行热部署，使用一个快捷键就 OK 了。但是如果现在开发者要修改的内容就只有一个文件中的少量代码，这个时候代码修改完毕如果能够让程序自己执行热部署功能，就可以减少开发者的操作，也就是自动进行热部署，能这么做吗？是可以的。咱们下一节再说。

### 自动启动热部署

自动热部署其实就是设计一个开关，打开这个开关后，IDE工具就可以自动热部署。因此这个操作和 IDE 工具有关，以下以 idea 为例设置 idea 中启动热部署

<b>步骤①</b>：设置自动构建项目

打开【File】，选择【settings...】,在面板左侧的菜单中找到【Compile】选项，然后勾选【Build project automatically】，意思是自动构建项目

<div align="center"><img src="img/boot/image-20220222123543551.png"></div>

自动构建项目选项勾选后

<b>步骤②</b>：允许在程序运行时进行自动构建

使用快捷键【Ctrl】+【Alt】+【Shit】+【/】打开维护面板，选择第1项【Registry...】

<div align="center"><img src="img/boot/image-20220222124006910.png"></div>

在选项中搜索 comple，然后勾选对应项即可

<div align="center"><img src="img/boot/image-20220222124240069.png"></div>

这样程序在运行的时候就可以进行自动构建了，实现了热部署的效果。

<font color="#ff0000"><b>关注</b></font>：如果你每敲一个字母，服务器就重新构建一次，这未免有点太频繁了，所以 idea 设置当 idea 工具失去焦点 5 秒后进行热部署。其实就是你从 idea 工具中切换到其他工具时进行热部署，比如改完程序需要到浏览器上去调试，这个时候 idea 就自动进行热部署操作。

<b>总结</b>

1. 自动热部署要开启自动构建项目
2. 自动热部署要开启在程序运行时自动构建项目

<b>思考</b>

现在已经实现了热部署了，但是到企业开发的时候你会发现，为了便于管理，在你的程序目录中除了有代码，还有可能有文档，如果你修改了一下文档，这个时候会进行热部署吗？不管是否进行热部署，这个过程我们需要自己控制才比较合理，那这个东西能控制吗？

### 参与热部署监控的文件范围配置

通过修改项目中的文件，你可以发现其实并不是所有的文件修改都会激活热部署的，原因在于在开发者工具中有一组配置，当满足了配置中的条件后，才会启动热部署，配置中默认不参与热部署的目录信息如下

- /META-INF/maven
- /META-INF/resources
- /resources
- /static
- /public
- /templates

以上目录中的文件如果发生变化，是不参与热部署的。如果想修改配置，可以通过 application.yml 文件进行设定哪些文件不参与热部署操作

```yaml
Spring:
  devtools:
    restart:
      # 设置不参与热部署的文件或文件夹
      exclude: static/<b>,public/<b>,config/application.yml
```

<b>总结</b>

1. 通过配置可以修改不参与热部署的文件或目录

<b>思考</b>

热部署功能是一个典型的开发阶段使用的功能，到了线上环境运行程序时，这个功能就没有意义了。能否关闭热部署功能呢？

### 关闭热部署

线上环境运行时是不可能使用热部署功能的，所以需要强制关闭此功能，通过配置可以关闭此功能。

```yaml
Spring:
  devtools:
    restart:
      enabled: false
```

如果当心配置文件层级过多导致相符覆盖最终引起配置失效，可以提高配置的层级，在更高层级中配置关闭热部署。例如在启动容器前通过系统属性设置关闭热部署功能。

```JAVA
@SpringBootApplication
public class SSMPApplication {
    public static void main(String[] args) {
        System.setProperty("Spring.devtools.restart.enabled","false");
        SpringApplication.run(SSMPApplication.class);
    }
}
```

其实上述担心略微有点多余，因为线上环境的维护是不可能出现修改代码的操作的，这么做唯一的作用是降低资源消耗，毕竟那双盯着你项目是不是产生变化的眼睛只要闭上了，就不具有热部署功能了，这个开关的作用就是禁用对应功能。

<b>总结</b>

1. 通过配置可以关闭热部署功能降低线上程序的资源消耗

## 配置高级

### @ConfigurationProperties

在基础篇学习了 @ConfigurationProperties 注解，此注解的作用是用来为 bean 绑定属性的。开发者可以在 yml 配置文件中以对象的格式添加若干属性

```YML
servers:
  ip-address: 192.168.0.1 
  port: 2345
  timeout: -1
```

然后再开发一个用来封装数据的实体类，注意要提供属性对应的 setter 方法

```JAVA
@Component
@Data
public class ServerConfig {
    private String ipAddress;
    private int port;
    private long timeout;
}
```

使用 @ConfigurationProperties 注解就可以将配置中的属性值关联到开发的模型类上

```JAVA
@Component
@Data
@ConfigurationProperties(prefix = "servers")
public class ServerConfig {
    private String ipAddress;
    private int port;
    private long timeout;
}
```

这样加载对应 bean 的时候就可以直接加载配置属性值了。但是目前我们学的都是给自定义的 bean 使用这种形式加载属性值，如果是第三方的 bean 呢？能不能用这种形式加载属性值呢？为什么会提出这个疑问？原因就在于当前 @ConfigurationProperties 注解是写在类定义的上方，而第三方开发的 bean 源代码不是你自己书写的，你也不可能到源代码中去添加 @ConfigurationProperties 注解，这种问题该怎么解决呢？

<b>使用 @ConfigurationProperties 注解其实可以为第三方 bean 加载属性</b>，格式特殊一点而已。

<b>步骤①</b>：使用 @Bean 注解定义第三方 bean

```JAVA
@Bean
public DruidDataSource datasource(){
    DruidDataSource ds = new DruidDataSource();
    return ds;
}
```

<b>步骤②</b>：在 yml 中定义要绑定的属性，注意 datasource 此时全小写

```YAML
datasource:
  driverClassName: com.mysql.jdbc.Driver
```

<b>步骤③</b>：使用 @ConfigurationProperties 注解为第三方 bean 进行属性绑定，注意前缀是全小写的 datasource

```JAVA
@Bean
@ConfigurationProperties(prefix = "datasource")
public DruidDataSource datasource(){
    DruidDataSource ds = new DruidDataSource();
    return ds;
}
```

操作方式完全一样，只不过 @ConfigurationProperties 注解不仅能添加到类上，还可以添加到方法上，<b>添加到类上是为 Spring 容器管理的当前类的对象绑定属性，添加到方法上是为 Spring 容器管理的当前方法的返回值对象绑定属性</b>，其实本质上都一样。

做到这其实就出现了一个新的问题，目前我们定义 bean 不是通过类注解定义就是通过 @Bean 定义，使用 @ConfigurationProperties 注解可以为 bean 进行属性绑定，那在一个业务系统中，哪些 bean 通过注解 @ConfigurationProperties 去绑定属性了呢？因为这个注解不仅可以写在类上，还可以写在方法上，所以找起来就比较麻烦了。为了解决这个问题，Spring 给我们提供了一个全新的注解，专门标注使用 @ConfigurationProperties 注解绑定属性的 bean 是哪些。这个注解叫做 @EnableConfigurationProperties。具体如何使用呢？

<b>步骤①</b>：在配置类上开启 @EnableConfigurationProperties 注解，并标注要使用 @ConfigurationProperties 注解绑定属性的类

```java
@SpringBootApplication
@EnableConfigurationProperties(ServerConfig.class)
public class Spring Boot13ConfigurationApplication {}
```

<b>步骤②</b>：在对应的类上直接使用 @ConfigurationProperties 进行属性绑定

```JAVA
@Data
@ConfigurationProperties(prefix = "servers")
public class ServerConfig {
    private String ipAddress;
    private int port;
    private long timeout;
}
```

有人感觉这没区别啊？注意观察，现在绑定属性的 ServerConfig 类并没有声明 @Component 注解。<b>当使用 @EnableConfigurationProperties 注解时，Spring 会默认将其标注的类（注解中的参数）定义为 bean</b>，因此无需再次声明 @Component 注解了。

最后再说一个小技巧，使用 @ConfigurationProperties 注解时，会出现一个提示信息

<div align="center"><img src="img/boot/image-20220222145535749.png"></div>

出现这个提示后只需要添加一个坐标此提醒就消失了

```XML
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-configuration-processor</artifactId>
</dependency>
```

<b>总结</b>

1. 使用 @ConfigurationProperties 可以为使用 @Bean 声明的第三方 bean 绑定属性
2. 当使用 @EnableConfigurationProperties 声明进行属性绑定的 bean 后，无需使用 @Component 注解再次进行 bean 声明

### 宽松绑定/松散绑定

在进行属性绑定时，可能会遇到如下情况，为了进行标准命名，开发者会将属性名严格按照驼峰命名法书写，在 yml 配置文件中将 datasource 修改为 dataSource，如下：

```yaml
dataSource:
  driverClassName: com.mysql.jdbc.Driver
```

不修改 Java 代码中的前缀，此时程序可以正常运行；然后又将代码中的前缀 datasource 修改为 dataSource，如下：

```java
@Bean
@ConfigurationProperties(prefix = "dataSource")
public DruidDataSource datasource(){
    DruidDataSource ds = new DruidDataSource();
    return ds;
}
```

此时就发生了编译错误，而且并不是 idea 工具导致的，运行后依然会出现问题，配置属性名 dataSource 是无效的。

```CMD
Configuration property name 'dataSource' is not valid:

    Invalid characters: 'S'
    Bean: datasource
    Reason: Canonical names should be kebab-case ('-' separated), lowercase alpha-numeric characters and must start with a letter

Action:
Modify 'dataSource' so that it conforms to the canonical names requirements.
```

为什么会出现这种问题，这就要来说一说 Spring Boot 进行属性绑定时的一个重要知识点了，有关属性名称的宽松绑定，也可以称为宽松绑定。

什么是宽松绑定？实际上是 Spring Boot 进行编程时人性化设计的一种体现，即配置文件中的命名格式与变量名的命名格式可以进行格式上的最大化兼容。兼容到什么程度呢？几乎主流的命名格式都支持，例如：

在 ServerConfig 中的 ipAddress 属性名

```JAVA
@Component
@Data
@ConfigurationProperties(prefix = "servers")
public class ServerConfig {
    private String ipAddress;
}
```

可以与下面的配置属性名规则全兼容

```YML
servers:
  ipAddress: 192.168.0.2       # 驼峰模式
  ip_address: 192.168.0.2      # 下划线模式
  ip-address: 192.168.0.2      # 烤肉串模式
  IP_ADDRESS: 192.168.0.2      # 常量模式
```

也可以说，以上 4 种模式最终都可以匹配到 ipAddress 这个属性名。因为在进行匹配时，配置中的名称要去掉中划线和下划线后，忽略大小写的情况下去与 java 代码中的属性名进行忽略大小写的等值匹配，以上 4 种命名去掉下划线中划线忽略大小写后都是一个词 ipaddress，java 代码中的属性名忽略大小写后也是 ipaddress，这样就可以进行等值匹配了，这就是为什么这 4 种格式都能匹配成功的原因。不过 Spring Boot 官方推荐使用烤肉串模式，也就是中划线模式。

到这里我们掌握了一个知识点，就是命名的规范问题。再来看开始出现的编程错误信息

```CMD
Configuration property name 'dataSource' is not valid:

    Invalid characters: 'S'
    Bean: datasource
    Reason: Canonical names should be kebab-case ('-' separated), lowercase alpha-numeric characters and must start with a letter

Action:
Modify 'dataSource' so that it conforms to the canonical names requirements.
```

其中 Reason 描述了报错的原因，规范的名称应该是烤肉串(kebab)模式(case)，即使用-分隔，使用小写字母数字作为标准字符，且必须以字母开头。然后再看我们写的名称 dataSource，就不满足上述要求。闹了半天，<b>在书写前缀时，这个词不是随意支持的，必须使用上述标准</b>。

最后说一句，以上规则仅针对 Spring Boot 中 @ConfigurationProperties 注解进行属性绑定时有效，对 @Value 注解进行属性映射无效。有人就说，那我不用你不就行了？不用，你小看 Spring Boot 的推广能力了，到原理篇我们看源码时，你会发现内部全是这玩意儿。

<b>总结</b>

1. @ConfigurationProperties 绑定属性时支持属性名宽松绑定，这个宽松体现在属性名的命名规则上
2. @Value 注解不支持松散绑定规则
3. 绑定前缀名推荐采用烤肉串命名规则，即使用中划线做分隔符

### 常用计量单位绑定

在前面的配置中，我们书写了如下配置值，其中第三项超时时间 timeout 描述了服务器操作超时时间，当前值是 -1 表示永不超时。

```YML
servers:
  ip-address: 192.168.0.1 
  port: 2345
  timeout: -1
```

但是每个人都这个值的理解会产生不同，比如线上服务器完成一次主从备份，配置超时时间 240，这个 240 如果单位是秒就是超时时间4分钟，如果单位是分钟就是超时时间4小时。面对一次线上服务器的主从备份，设置 4 分钟，简直是开玩笑，别说拷贝过程，备份之前的压缩过程 4 分钟也搞不定，这个时候问题就来了，怎么解决这个误会？

除了加强约定之外，Spring Boot 充分利用了 JDK8 中提供的全新的用来表示计量单位的新数据类型，从根本上解决这个问题。以下模型类中添加了两个JDK8中新增的类，分别是 Duration 和 DataSize

```JAVA
@Component
@Data
@ConfigurationProperties(prefix = "servers")
public class ServerConfig {
    @DurationUnit(ChronoUnit.HOURS)
    private Duration serverTimeOut;
    @DataSizeUnit(DataUnit.MEGABYTES)
    private DataSize dataSize;
}
```

<b>Duration</b>：表示时间间隔，可以通过 @DurationUnit 注解描述时间单位，例如上例中描述的单位为小时（ChronoUnit.HOURS）

<b>DataSize</b>：表示存储空间，可以通过 @DataSizeUnit 注解描述存储空间单位，例如上例中描述的单位为 MB（DataUnit.MEGABYTES）

使用上述两个单位就可以有效避免因沟通不同步或文档不健全导致的信息不对称问题，从根本上解决了问题，避免产生误读。

Druation 常用单位如下：

<div align="center"><img src="img/boot/image-20220222173911102.png"></div>

DataSize 常用单位如下：

<div align="center"><img src="img/boot/image-20220222174130102.png"></div>

### 校验

目前我们在进行属性绑定时可以通过松散绑定规则在书写时放飞自我了，但是在书写时由于无法感知模型类中的数据类型，就会出现类型不匹配的问题，比如代码中需要 int 类型，配置中给了非法的数值，例如写一个 “a"，这种数据肯定无法有效的绑定，还会引发错误。		

Spring Boot 给出了强大的数据校验功能，可以有效的避免此类问题的发生。在 JAVAEE 的 JSR303 规范中给出了具体的数据校验标准，开发者可以根据自己的需要选择对应的校验框架，此处使用 Hibernate 提供的校验框架来作为实现进行数据校验。书写应用格式非常固定，话不多说，直接上步骤

<b>步骤①</b>：开启校验框架

```xml
<!--1.导入JSR303规范-->
<dependency>
    <groupId>javax.validation</groupId>
    <artifactId>validation-api</artifactId>
</dependency>
<!--使用hibernate框架提供的校验器做实现-->
<dependency>
    <groupId>org.hibernate.validator</groupId>
    <artifactId>hibernate-validator</artifactId>
</dependency>
```

<b>步骤②</b>：在需要开启校验功能的类上使用注解 @Validated 开启校验功能

```java
@Component
@Data
@ConfigurationProperties(prefix = "servers")
//开启对当前bean的属性注入校验
@Validated
public class ServerConfig {
}
```

<b>步骤③</b>：对具体的字段设置校验规则

```JAVA
@Component
@Data
@ConfigurationProperties(prefix = "servers")
//开启对当前bean的属性注入校验
@Validated
public class ServerConfig {
    //设置具体的规则
    @Max(value = 8888,message = "最大值不能超过8888")
    @Min(value = 202,message = "最小值不能低于202")
    private int port;
}
```

通过设置数据格式校验，就可以有效避免非法数据加载，其实使用起来还是挺轻松的，基本上就是一个格式。

<b>总结</b>

1.开启 Bean 属性校验功能一共3步：导入 JSR303 与 Hibernate 校验框架坐标、使用 @Validated 注解启用校验功能、使用具体校验规则规范数据校验格式

### 数据类型转换

看下这个问题。连接数据库是，运行程序提示密码错误

```shell
java.sql.SQLException: Access denied for user 'root'@'localhost' (using password: YES)
```

其实看到这个报错，看的出这是用户名和密码不匹配，但是问题就在于密码并没有输入错误，这就比较讨厌了。给的报错信息无法帮助你有效的分析问题，甚至会给你带到沟里。来看看用户名密码的配置是如何写的：

```yaml
Spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
    username: root
    password: 0127
```

这名开发者的生日是 1 月 27 日，所以密码就使用了 0127，其实问题就出在这里了。之前在基础篇讲属性注入时，提到过类型相关的知识，在整数相关知识中有这么一句话，<b>支持二进制，八进制，十六进制</b>

<div align="center"><img src="img/boot/image-20220222225748370.png"></div>

这个问题就处在这里了，因为 0127 在开发者眼中是一个字符串 “0127”，但是在 Spring Boot 看来，这就是一个数字，而且是一个八进制的数字。当后台使用 String 类型接收数据时，如果配置文件中配置了一个整数值，他是先按照整数进行处理，读取后再转换成字符串。巧了，0127 撞上了八进制的格式，所以最终以十进制数字 87 的结果存在了。

<span style="color:red">这里提两个注意点，第一，字符串标准书写加上引号包裹，养成习惯，第二，遇到 0 开头的数据多注意吧。</span>

<b>总结</b>

yaml 文件中对于数字的定义支持进制书写格式，如需使用字符串请使用引号明确标注。

## 测试

测试是保障程序正确性的唯一屏障。

### 加载测试专用属性

测试过程本身并不是一个复杂的过程，但是很多情况下测试时需要模拟一些线上情况，或者模拟一些特殊情况。如果当前环境按照线上环境已经设定好了，例如是下面的配置

```YAML
env:
  maxMemory: 32GB
  minMemory: 16GB
```

但是你现在想测试对应的兼容性，需要测试如下配置

```YAML
env:
  maxMemory: 16GB
  minMemory: 8GB
```

这个时候我们能不能每次测试的时候都去修改源码 application.yml 中的配置进行测试呢？显然是不行的。每次测试前改过来，每次测试后改回去，这太麻烦了。于是我们就想，需要在测试环境中创建一组临时属性，去覆盖我们源码中设定的属性，这样测试用例就相当于是一个独立的环境，能够独立测试，这样就方便多了。

<b>临时属性</b>

Spring Boot 已经为我们开发者早就想好了这种问题该如何解决，并且提供了对应的功能入口。在测试用例程序中，可以通过对注解 @SpringBootTest 添加属性来模拟临时属性，具体如下：

```JAVA
//properties属性可以为当前测试用例添加临时的属性配置
@SpringBootTest(properties = {"test.prop=testValue1"})
public class PropertiesAndArgsTest {

    @Value("${test.prop}")
    private String msg;
    
    @Test
    void testProperties(){
        System.out.println(msg);
    }
}
```

使用注解 @SpringBootTest 的 properties 属性就可以为当前测试用例添加临时的属性，覆盖源码配置文件中对应的属性值进行测试。

<b>临时参数</b>

除了上述这种情况，在前面讲解使用命令行启动 Spring Boot 程序时讲过，通过命令行参数也可以设置属性值。而且线上启动程序时，通常都会添加一些专用的配置信息。作为运维人员他们才不懂 java，更不懂这些配置的信息具体格式该怎么写，那如果我们作为开发者提供了对应的书写内容后，能否提前测试一下这些配置信息是否有效呢？当时是可以的，还是通过注解 @SpringBootTest 的另一个属性来进行设定。

```JAVA
//args属性可以为当前测试用例添加临时的命令行参数
@SpringBootTest(args={"--test.prop=testValue2"})
public class PropertiesAndArgsTest {
    
    @Value("${test.prop}")
    private String msg;
    
    @Test
    void testProperties(){
        System.out.println(msg);
    }
}
```

使用注解 @SpringBootTest 的 args 属性就可以为当前测试用例模拟命令行参数并进行测试。

如果两者共存呢？其实如果思考一下配置属性与命令行参数的加载优先级，这个结果就不言而喻了。在属性加载的优先级设定中，有明确的优先级设定顺序

<div align="center"><img src="img/boot/image-20211206100859236.png" alt="image-20211206100859236"></div>

在这个属性加载优先级的顺序中，明确规定了命令行参数的优先级排序是 11，而配置属性的优先级是 3，结果不言而喻了，args 属性配置优先于 properties 属性配置加载。

到这里我们就掌握了如果在测试用例中去模拟临时属性的设定。

<b>总结</b>

1. 加载测试临时属性可以通过注解 @SpringBootTest 的 properties 和 args 属性进行设定，此设定应用范围仅适用于当前测试用例

<b>思考</b>

应用于测试环境的临时属性解决了，如果想在测试的时候临时加载一些 bean 能不做呢？也就是说我测试时，想搞一些独立的 bean 出来，专门应用于测试环境，能否实现呢？

### 加载测试专用配置

上一节提出了临时配置一些专用于测试环境的 bean 的需求，这一节我们就来解决这个问题。

学习过 Spring 的知识，我们都知道，其实一个 Spring 环境中可以设置若干个配置文件或配置类，若干个配置信息可以同时生效。现在我们的需求就是在测试环境中再添加一个配置类，然后启动测试环境时，生效此配置就行了。其实做法和 Spring 环境中加载多个配置信息的方式完全一样。具体操作步骤如下：

<b>步骤①</b>：在测试包 test 中创建专用的测试环境配置类

```java
@Configuration
public class MsgConfig {
    @Bean
    public String msg(){
        return "bean msg";
    }
}
```

上述配置仅用于演示当前实验效果，实际开发可不能这么注入 String 类型的数据

<b>步骤②</b>：在启动测试环境时，导入测试环境专用的配置类，使用 @Import 注解即可实现

```java
@SpringBootTest
@Import({MsgConfig.class})
public class ConfigurationTest {

    @Autowired
    private String msg;

    @Test
    void testConfiguration(){
        System.out.println(msg);
    }
}
```

到这里就通过 @Import 属性实现了基于开发环境的配置基础上，对配置进行测试环境的追加操作，实现了 1+1 的配置环境效果。这样我们就可以实现每一个不同的测试用例加载不同的bean的效果，丰富测试用例的编写，同时不影响开发环境的配置。

<b>总结</b>

1. 定义测试环境专用的配置类，然后通过 @Import 注解在具体的测试中导入临时的配置，例如测试用例，方便测试过程，且上述配置不影响其他的测试类环境

<b>思考</b>

当前我们已经可以实现业务层和数据层的测试，并且通过临时配置，控制每个测试用例加载不同的测试数据。但是实际企业开发不仅要保障业务层与数据层的功能安全有效，也要保障表现层的功能正常。但是我们目的对表现层的测试都是通过 postman 手工测试的，并没有在打包过程中体现表现层功能被测试通过。能否在测试用例中对表现层进行功能测试呢？还真可以，咱们下一节再讲。

### Web环境模拟测试

在测试中对表现层功能进行测试需要一个基础和一个功能。所谓的一个基础是运行测试程序时，必须启动 web 环境，不然没法测试 web 功能。一个功能是必须在测试程序中具备发送 web 请求的能力，不然无法实现 web 功能的测试。所以在测试用例中测试表现层接口这项工作就转换成了两件事，一，如何在测试类中启动 web 测试，二，如何在测试类中发送 web 请求。下面一件事一件事进行，先说第一个

<b>测试类中启动 web 环境</b>

每一个 Spring Boot 的测试类上方都会标准 @SpringBootTest 注解，而注解带有一个属性，叫做 webEnvironment。通过该属性就可以设置在测试用例中启动 web 环境，具体如下：

```JAVA
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class WebTest {	}
```

测试类中启动 web 环境时，可以指定启动的 Web 环境对应的端口，Spring Boot 提供了 4 种设置值，分别如下：

<div align="center"><img src="img/boot/image-20220223125453317.png" alt="image-20220223125453317" style="zoom:80%;" /></div>

- MOCK：根据当前设置确认是否启动 web 环境，例如使用了 Servlet 的 API 就启动 web 环境，属于适配性的配置
- DEFINED_PORT：使用自定义的端口作为 web 服务器端口
- RANDOM_PORT：使用随机端口作为 web 服务器端口
- NONE：不启动 web 环境

通过上述配置，现在启动测试程序时就可以正常启用 web 环境了，建议大家测试时使用 RANDOM_PORT，避免代码中因为写死设定引发线上功能打包测试时由于端口冲突导致意外现象的出现。就是说你程序中写了用 8080 端口，结果线上环境 8080 端口被占用了，结果你代码中所有写的东西都要改，这就是写死代码的代价。现在你用随机端口就可以测试出来你有没有这种问题的隐患了。

测试环境中的 web 环境已经搭建好了，下面就可以来解决第二个问题了，如何在程序代码中发送 web 请求。

<b>测试类中发送请求</b>

对于测试类中发送请求，其实 java 的 API 就提供对应的功能，只不过平时各位小伙伴接触的比较少，所以较为陌生。Spring Boot 为了便于开发者进行对应的功能开发，对其又进行了包装，简化了开发步骤，具体操作如下：

<b>步骤①</b>：在测试类中开启 web 虚拟调用功能，通过注解 @AutoConfigureMockMvc 实现此功能的开启

```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
//开启虚拟MVC调用
@AutoConfigureMockMvc
public class WebTest {}
```

<b>步骤②</b>：定义发起虚拟调用的对象 MockMVC，通过自动装配的形式初始化对象

```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
//开启虚拟MVC调用
@AutoConfigureMockMvc
public class WebTest {

    @Test
    void testWeb(@Autowired MockMvc mvc) {}
}
```

<b>步骤③</b>：创建一个虚拟请求对象，封装请求的路径，并使用 MockMVC 对象发送对应请求

```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
//开启虚拟MVC调用
@AutoConfigureMockMvc
public class WebTest {

    @Test
    void testWeb(@Autowired MockMvc mvc) throws Exception {
        //http://localhost:8080/books
        //创建虚拟请求，当前访问/books
        MockHttpServletRequestBuilder builder = MockMvcRequestBuilders.get("/books");
        //执行对应的请求
        mvc.perform(builder);
    }
}
```

执行测试程序，现在就可以正常的发送/books对应的请求了，注意访问路径不要写 http://localhost:8080/books，因为前面的服务器 IP 地址和端口使用的是当前虚拟的 web 环境，无需指定，仅指定请求的具体路径即可。

<b>总结</b>

1. 在测试类中测试 web 层接口要保障测试类启动时启动 web 容器，使用 @SpringBootTest 注解的 webEnvironment 属性可以虚拟 web 环境用于测试
2. 为测试方法注入 MockMvc 对象，通过 MockMvc 对象可以发送虚拟请求，模拟 web 请求调用过程

<b>思考</b>

目前已经成功的发送了请求，但是还没有起到测试的效果，测试过程必须出现预计值与真实值的比对结果才能确认测试结果是否通过，虚拟请求中能对哪些请求结果进行比对呢？

<b>web 环境请求结果比对</b>

上一节已经在测试用例中成功的模拟出了 web 环境，并成功的发送了 web 请求，本节就来解决发送请求后如何比对发送结果的问题。其实发完请求得到的信息只有一种，就是响应对象。至于响应对象中包含什么，就可以比对什么。常见的比对内容如下：

- 响应状态匹配

    ```JAVA
    @Test
    void testStatus(@Autowired MockMvc mvc) throws Exception {
        MockHttpServletRequestBuilder builder = MockMvcRequestBuilders.get("/books");
        ResultActions action = mvc.perform(builder);
        //设定预期值 与真实值进行比较，成功测试通过，失败测试失败
        //定义本次调用的预期值
        StatusResultMatchers status = MockMvcResultMatchers.status();
        //预计本次调用时成功的：状态200
        ResultMatcher ok = status.isOk();
        //添加预计值到本次调用过程中进行匹配
        action.andExpect(ok);
    }
    ```

- 响应体匹配（非 json 数据格式）

    ```JAVA
    @Test
    void testBody(@Autowired MockMvc mvc) throws Exception {
        MockHttpServletRequestBuilder builder = MockMvcRequestBuilders.get("/books");
        ResultActions action = mvc.perform(builder);
        //设定预期值 与真实值进行比较，成功测试通过，失败测试失败
        //定义本次调用的预期值
        ContentResultMatchers content = MockMvcResultMatchers.content();
        ResultMatcher result = content.string("Spring Boot2");
        //添加预计值到本次调用过程中进行匹配
        action.andExpect(result);
    }
    ```

- 响应体匹配（json 数据格式，开发中的主流使用方式）

    ```JAVA
    @Test
    void testJson(@Autowired MockMvc mvc) throws Exception {
        MockHttpServletRequestBuilder builder = MockMvcRequestBuilders.get("/books");
        ResultActions action = mvc.perform(builder);
        //设定预期值 与真实值进行比较，成功测试通过，失败测试失败
        //定义本次调用的预期值
        ContentResultMatchers content = MockMvcResultMatchers.content();
        ResultMatcher result = content.json("{\"id\":1,\"name\":\"Spring Boot2\",\"type\":\"Spring Boot\"}");
        //添加预计值到本次调用过程中进行匹配
        action.andExpect(result);
    }
    ```

- 响应头信息匹配

    ```JAVA
    @Test
    void testContentType(@Autowired MockMvc mvc) throws Exception {
        MockHttpServletRequestBuilder builder = MockMvcRequestBuilders.get("/books");
        ResultActions action = mvc.perform(builder);
        //设定预期值 与真实值进行比较，成功测试通过，失败测试失败
        //定义本次调用的预期值
        HeaderResultMatchers header = MockMvcResultMatchers.header();
        ResultMatcher contentType = header.string("Content-Type", "application/json");
        //添加预计值到本次调用过程中进行匹配
        action.andExpect(contentType);
    }
    ```

​		基本上齐了，头信息，正文信息，状态信息都有了，就可以组合出一个完美的响应结果比对结果了。以下范例就是三种信息同时进行匹配校验，也是一个完整的信息匹配过程。

```JAVA
@Test
void testGetById(@Autowired MockMvc mvc) throws Exception {
    MockHttpServletRequestBuilder builder = MockMvcRequestBuilders.get("/books");
    ResultActions action = mvc.perform(builder);

    StatusResultMatchers status = MockMvcResultMatchers.status();
    ResultMatcher ok = status.isOk();
    action.andExpect(ok);

    HeaderResultMatchers header = MockMvcResultMatchers.header();
    ResultMatcher contentType = header.string("Content-Type", "application/json");
    action.andExpect(contentType);

    ContentResultMatchers content = MockMvcResultMatchers.content();
    ResultMatcher result = content.json("{\"id\":1,\"name\":\"Spring Boot\",\"type\":\"Spring Boot\"}");
    action.andExpect(result);
}
```

<b>总结</b>

1. web 虚拟调用可以对本地虚拟请求的返回响应信息进行比对，分为响应头信息比对、响应体信息比对、响应状态信息比对

### 数据层测试回滚

当前我们的测试程序可以完美的进行表现层、业务层、数据层接口对应的功能测试了，但是测试用例开发完成后，在打包的阶段由于 test 生命周期属于必须被运行的生命周期，如果跳过会给系统带来极高的安全隐患，所以测试用例必须执行。但是新的问题就呈现了，测试用例如果测试时产生了事务提交就会在测试过程中对数据库数据产生影响，进而产生垃圾数据。这个过程不是我们希望发生的，作为开发者测试用例该运行运行，但是过程中产生的数据不要在我的系统中留痕，这样该如何处理呢？

Spring Boot 早就为开发者想到了这个问题，并且针对此问题给出了最简解决方案，在原始测试用例中添加注解 @Transactional 即可实现当前测试用例的事务不提交。当程序运行后，只要注解 @Transactional 出现的位置存在注解 @SpringBootTest，Spring Boot 就会认为这是一个测试程序，无需提交事务，所以也就可以避免事务的提交。

```JAVA
@SpringBootTest
@Transactional
@Rollback(true)
public class DaoTest {
    @Autowired
    private BookService bookService;

    @Test
    void testSave(){
        Book book = new Book();
        book.setName("Spring Boot3");
        book.setType("Spring Boot3");
        book.setDescription("Spring Boot3");

        bookService.save(book);
    }
}
```

如果开发者想提交事务，也可以，再添加一个 @RollBack 的注解，设置回滚状态为 false 即可正常提交事务。Spring Boot 在辅助开发者日常工作这一块展现出了惊人的能力。

<b>总结</b>

1. 在 Spring Boot 的测试类中通过添加注解 @Transactional 来阻止测试用例提交事务
2. 通过注解 @Rollback 控制 Spring Boot 测试类执行结果是否提交事务，需要配合注解 @Transactional 使用

<b>思考</b>

当前测试程序已经近乎完美了，但是由于测试用例中书写的测试数据属于固定数据，往往失去了测试的意义，开发者可以针对测试用例进行针对性开发，这样就有可能出现测试用例不能完美呈现业务逻辑代码是否真实有效的达成业务目标的现象，解决方案其实很容易想，测试用例的数据只要随机产生就可以了，能实现吗？

### 测试用例数据设定

对于测试用例的数据固定书写肯定是不合理的，Spring Boot 提供了在配置中使用随机值的机制，确保每次运行程序加载的数据都是随机的。具体如下：

```yaml
testcase:
  book:
    id: ${random.int}
    id2: ${random.int(10)}
    type: ${random.int!5,10!}
    name: ${random.value}
    uuid: ${random.uuid}
    publishTime: ${random.long}
```

当前配置就可以在每次运行程序时创建一组随机数据，避免每次运行时数据都是固定值的尴尬现象发生，有助于测试功能的进行。数据的加载按照之前加载数据的形式，使用 @ConfigurationProperties 注解即可

```JAVA
@Component
@Data
@ConfigurationProperties(prefix = "testcase.book")
public class BookCase {
    private int id;
    private int id2;
    private int type;
    private String name;
    private String uuid;
    private long publishTime;
}
```

对于随机值的产生，还有一些小的限定规则，比如产生的数值性数据可以设置范围等，具体如下：

<div align="center"><img src="img/boot/image-20220223135454862.png" alt="image-20220223135454862" style="zoom:80%;" /></div>

- ${random.int} 表示随机整数
- ${random.int(10)} 表示 10 以内的随机数
- ${random.int(10,20)} 表示 10 到 20 的随机数
- 其中 () 可以是任意字符，例如[]，!!均可

<b>总结</b>

1. 使用随机数据可以替换测试用例中书写的固定数据，提高测试用例中的测试数据有效性

## 数据层解决方案

开发实用篇前三章基本上是开胃菜，从第四章开始，开发实用篇进入到了噩梦难度了，从这里开始，不再是单纯的在 Spring Boot 内部搞事情了，要涉及到很多相关知识。本章节主要内容都是和数据存储与读取相关，前期学习的知识与数据层有关的技术基本上都围绕在数据库这个层面上，所以本章要讲的第一个大的分支就是 SQL 解决方案相关的内容，除此之外，数据的来源还可以是非 SQL 技术相关的数据操作，因此第二部分围绕着 NOSQL 解决方案讲解。

### SQL

回忆一下之前做 SSMP 整合的时候数据层解决方案涉及到了哪些技术？MySQL 数据库与 MyBatisPlus 框架，后面又学了 Druid 数据源的配置，所以现在数据层解决方案可以说是Mysql+Druid+MyBatisPlus。而三个技术分别对应了数据层操作的三个层面：

- 数据源技术：Druid
- 持久化技术：MyBatisPlus
- 数据库技术：MySQL

下面的研究就分为三个层面进行研究，对应上面列出的三个方面，从第一个数据源技术开始说起。

#### 数据源技术

目前我们使用的数据源技术是 Druid，运行时可以在日志中看到对应的数据源初始化信息，具体如下：

```CMD
INFO 28600 --- [           main] c.a.d.s.b.a.DruidDataSourceAutoConfigure : Init DruidDataSource
INFO 28600 --- [           main] com.alibaba.druid.pool.DruidDataSource   : {dataSource-1} inited
```

如果不使用 Druid 数据源，程序运行后是什么样子呢？是独立的数据库连接对象还是有其他的连接池技术支持呢？将 Druid 技术对应的 starter 去掉再次运行程序可以在日志中找到如下初始化信息：

```CMD
INFO 31820 --- [           main] com.zaxxer.hikari.HikariDataSource       : HikariPool-1 - Starting...
INFO 31820 --- [           main] com.zaxxer.hikari.HikariDataSource       : HikariPool-1 - Start completed.
```

虽然没有 DruidDataSource 相关的信息了，但是我们发现日志中有 HikariDataSource 这个信息，就算不懂这是个什么技术，看名字也能看出来，以 DataSource 结尾的名称，这一定是一个数据源技术。我们又没有手工添加这个技术，这个技术哪里来的呢？这个是 Spring Boot 内嵌的数据源。

数据层技术是每一个企业级应用程序都会用到的，而其中必定会进行数据库连接的管理。Spring Boot 根据开发者的习惯出发，开发者提供了数据源技术，就用你提供的，开发者没有提供，那总不能手工管理一个一个的数据库连接对象啊，怎么办？我给你一个默认的就好了，这样省心又省事，大家都方便。

Spring Boot 提供了 3 款内嵌数据源技术，分别如下：

- HikariCP
- Tomcat 提供 DataSource
- Commons DBCP

第一种，HikartCP，这是 Spring Boot 官方推荐的数据源技术，作为默认内置数据源使用。不配置数据源，那就用这个。

第二种，Tomcat 提供的 DataSource，如果不想用 HikartCP，并且使用 tomcat 作为 web 服务器进行 web 程序的开发，使用这个。为什么是 Tomcat，不是其他 web 服务器呢？因为 web 技术导入 starter 后，默认使用内嵌 tomcat，既然都是默认使用的技术了，那就一用到底，数据源也用它的。有人就提出怎么才能不使用 HikartCP 用 tomcat 提供的默认数据源对象呢？把 HikartCP 技术的坐标排除掉就 OK 了。

第三种，DBCP，这个使用的条件就更苛刻了，既不使用 HikartCP 也不使用 tomcat 的 DataSource 时，默认给你用这个。

怎么配置使用这些东西呢？之前我们配置 druid 时使用 druid 的 starter 对应的配置如下：

```YAML
Spring:
  datasource:
    druid:	
   	  url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
      driver-class-name: com.mysql.cj.jdbc.Driver
      username: root
      password: root
```

换成是默认的数据源 HikariCP 后，直接把 druid 删掉就行了，如下：

```YAML
Spring:
  datasource:
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
    driver-class-name: com.mysql.cj.jdbc.Driver
    username: root
    password: root
```

当然，也可以写上是对 hikari 做的配置，但是 url 地址要单独配置，如下：

```YAML
Spring:
  datasource:
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
    hikari:
      driver-class-name: com.mysql.cj.jdbc.Driver
      username: root
      password: root
```

这就是配置 hikari 数据源的方式。如果想对 hikari 做进一步的配置，可以继续配置其独立的属性。例如：

```yaml
Spring:
  datasource:
    url: jdbc:mysql://localhost:3306/ssm_db?serverTimezone=UTC
    hikari:
      driver-class-name: com.mysql.cj.jdbc.Driver
      username: root
      password: root
      maximum-pool-size: 50
```

如果不想使用 hikari 数据源，使用 tomcat 的数据源或者 DBCP 配置格式也是一样的。学习到这里，以后我们做数据层时，数据源对象的选择就不再是单一的使用 druid 数据源技术了，可以根据需要自行选择。

<b>总结</b>

1. Spring Boot 技术提供了 3 种内置的数据源技术，分别是 Hikari、tomcat 内置数据源、DBCP

#### 持久化技术

说完数据源解决方案，再来说一下持久化解决方案。Spring Boot 充分发挥其最强辅助的特征，给开发者提供了一套现成的数据层技术，叫做 JdbcTemplate。这个技术是 Spring 提供的，所以在 Spring Boot 技术范畴中，这个技术也是存在的。

这个技术其实就是回归到 jdbc 最原始的编程形式来进行数据层的开发，下面直接上操作步骤：

<b>步骤①</b>：导入 jdbc 对应的坐标，记得是 starter

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-jdbc</artifactId>
</dependency
```

<b>步骤②</b>：自动装配 JdbcTemplate 对象

```java
@SpringBootTest
class Spring Boot15SqlApplicationTests {
    @Test
    void testJdbcTemplate(@Autowired JdbcTemplate jdbcTemplate){}
}
```

<b>步骤③</b>：使用 JdbcTemplate 实现查询操作（非实体类封装数据的查询操作）

```java
@Test
void testJdbcTemplate(@Autowired JdbcTemplate jdbcTemplate){
    String sql = "select * from tbl_book";
    List<Map<String, Object>> maps = jdbcTemplate.queryForList(sql);
    System.out.println(maps);
}
```

<b>步骤④</b>：使用 JdbcTemplate 实现查询操作（实体类封装数据的查询操作）

```java
@Test
void testJdbcTemplate(@Autowired JdbcTemplate jdbcTemplate){

    String sql = "select * from tbl_book";
    RowMapper<Book> rm = new RowMapper<Book>() {
        @Override
        public Book mapRow(ResultSet rs, int rowNum) throws SQLException {
            Book temp = new Book();
            temp.setId(rs.getInt("id"));
            temp.setName(rs.getString("name"));
            temp.setType(rs.getString("type"));
            temp.setDescription(rs.getString("description"));
            return temp;
        }
    };
    List<Book> list = jdbcTemplate.query(sql, rm);
    System.out.println(list);
}
```

<b>步骤⑤</b>：使用 JdbcTemplate 实现增删改操作

```java
@Test
void testJdbcTemplateSave(@Autowired JdbcTemplate jdbcTemplate){
    String sql = "insert into tbl_book values(3,'Spring Boot1','Spring Boot2','Spring Boot3')";
    jdbcTemplate.update(sql);
}
```

如果想对 JdbcTemplate 对象进行相关配置，可以在 yml 文件中进行设定，具体如下：

```yaml
Spring:
  jdbc:
    template:
      query-timeout: -1   # 查询超时时间
      max-rows: 500       # 最大行数
      fetch-size: -1      # 缓存行数
```

<b>总结</b>

1. Spring Boot 内置 JdbcTemplate 持久化解决方案
2. 使用 JdbcTemplate 需要导入 Spring-boot-starter-jdbc 的坐标

#### 数据库技术

Spring Boot 提供了 3 款内置的数据库，分别是

- H2
- HSQL
- Derby

以上三款数据库除了可以独立安装之外，还可以像是 tomcat 服务器一样，采用内嵌的形式运行在 Spring Boot 容器中。这三款数据库底层都是使用 java 语言开发的。

这三个数据库都可以采用内嵌容器的形式运行，在应用程序运行后，如果我们进行测试工作，此时测试的数据无需存储在磁盘上，但是又要测试使用，内嵌数据库就很方便，运行在内存中，该测试测试，该运行运行，等服务器关闭后，一切烟消云散，不必维护外部数据库了。这也是内嵌数据库的最大优点，方便进行功能测试。

下面以 H2 数据库为例讲解如何使用这些内嵌数据库，操作步骤也非常简单

<b>步骤①</b>：导入 H2 数据库对应的坐标，一共 2 个

```xml
<dependency>
    <groupId>com.h2database</groupId>
    <artifactId>h2</artifactId>
</dependency>
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-data-jpa</artifactId>
</dependency>
```

<b>步骤②</b>：将工程设置为 web 工程，启动工程时启动 H2 数据库

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-web</artifactId>
</dependency>
```

<b>步骤③</b>：通过配置开启 H2 数据库控制台访问程序，也可以使用其他的数据库连接软件操作

```yaml
Spring:
  h2:
    console:
      enabled: true
      path: /h2
```

web 端访问路径 /h2，访问密码 123456，如果访问失败，先配置下列数据源，启动程序运行后再次访问 /h2 路径就可以正常访问了

```yaml
datasource:
  url: jdbc:h2:~/test
  hikari:
    driver-class-name: org.h2.Driver
    username: sa
    password: 123456
```

<b>步骤④</b>：使用 JdbcTemplate 或 MyBatisPlus 技术操作数据库

上线时，切记关闭内存级数据库，采用 MySQL 数据库作为数据持久化方案，关闭方式就是设置 enabled 属性为 false 即可。

<b>总结</b>

1. H2 内嵌式数据库启动方式，添加坐标，添加配置
2. H2 数据库线上运行时请务必关闭

到这里 SQL 相关的数据层解决方案就讲完了，现在的可选技术就丰富的多了。

- 数据源技术：Druid、Hikari、tomcat DataSource、DBCP
- 持久化技术：MyBatisPlus、MyBatis、JdbcTemplate
- 数据库技术：MySQL、H2、HSQL、Derby

现在开发程序时就可以在以上技术中任选一种组织成一套数据库解决方案了。

### NoSQL

NoSQL 是非关系型数据库解决方案。本节讲解的内容就是 Spring Boot 如何整合这些技术，在 Spring Boot 官方文档中提供了 10 种相关技术的整合方案，国内市场上最流行的几款 NoSQL 数据库整合方案，分别是 Redis、MongoDB、ES。

#### 整合Redis

Redis 是一款采用 key-value 数据存储格式的内存级 NoSQL 数据库，重点关注数据存储格式，是 key-value 格式，也就是键值对的存储形式。与 MySQL 数据库不同，MySQL 数据库有表、有字段、有记录，Redis 没有这些东西，就是一个名称对应一个值，并且数据以存储在内存中使用为主。什么叫以存储在内存中为主？其实 Redis 有它的数据持久化方案，分别是 RDB 和 AOF，但是 Redis 自身并不是为了数据持久化而生的，主要是在内存中保存数据，加速数据访问的，所以说是一款内存级数据库。

Redis 支持多种数据存储格式，比如可以直接存字符串，也可以存一个 map 集合，list 集合，后面会涉及到一些不同格式的数据操作，这个需要先学习一下才能进行整合，所以在基本操作中会介绍一些相关操作。下面就先安装，再操作，最后说整合

##### 安装

windows 版安装包下载地址：https://github.com/tporadowski/redis/releases

下载的安装包有两种形式，一种是一键安装的 msi 文件，还有一种是解压缩就能使用的 zip 文件，哪种形式都行，此处采用的是 msi 一键安装的 msi 文件进行安装的。

msi，其实就是一个文件安装包，不仅安装软件，还帮你把安装软件时需要的功能关联在一起，打包操作。比如如安装序列、创建和设置安装路径、设置系统依赖项、默认设定安装选项和控制安装过程的属性。

安装完毕后会得到如下文件，其中有两个文件对应两个命令，是启动 Redis 的核心命令，需要再 CMD 命令行模式执行。

<div align="center"><img src="img/boot/image-20220224091709173.png" alt="image-20220224091709173" style="zoom: 80%;" /></div>

<b>启动服务器</b>

```CMD
redis-server.exe redis.windows.conf
```

初学者无需调整服务器对外服务端口，默认 6379。

<b>启动客户端</b>

```CMD
redis-cli.exe
```

如果启动 redis 服务器失败，可以先启动客户端，然后执行 shutdown 操作后退出，此时 redis 服务器就可以正常执行了。

##### 基本操作

服务器启动后，使用客户端就可以连接服务器，类似于启动完 MySQL 数据库，然后启动 SQL 命令行操作数据库。		

放置一个字符串数据到 redis 中，先为数据定义一个名称，比如 name,age 等，然后使用命令 set 设置数据到 redis 服务器中即可

```CMD
set name itheima
set age 12
```

从 redis 中取出已经放入的数据，根据名称取，就可以得到对应数据。如果没有对应数据就会得到(nil)

```CMD
get name
get age
```

以上使用的数据存储是一个名称对应一个值，如果要维护的数据过多，可以使用别的数据存储结构。例如 hash，它是一种一个名称下可以存储多个数据的存储模型，并且每个数据也可以有自己的二级存储名称。向 hash 结构中存储数据格式如下：

```
hset a a1 aa1		#对外key名称是a，在名称为a的存储模型中，a1这个key中保存了数据aa1
hset a a2 aa2
```

获取 hash 结构中的数据命令如下

```CMD
hget a a1			#得到aa1
hget a a2			#得到aa2
```

有关 redis 的基础操作就普及到这里，需要全面掌握 redis 技术，请参看相关教程学习。

##### 整合

在进行整合之前先梳理一下整合的思想，Spring Boot 整合任何技术其实就是在 Spring Boot 中使用对应技术的 API。如果两个技术没有交集，就不存在整合的概念了。所谓整合其实就是使用 Spring Boot 技术去管理其他技术，几个问题是躲不掉的。

第一，需要先导入对应技术的坐标，而整合之后，这些坐标都有了一些变化

第二，任何技术通常都会有一些相关的设置信息，整合之后，这些信息如何写，写在哪是一个问题

第三，没有整合之前操作如果是模式A的话，整合之后如果没有给开发者带来一些便捷操作，那整合将毫无意义，所以整合后操作肯定要简化一些，那对应的操作方式自然也有所不同

按照上面的三个问题去思考 Spring Boot 整合所有技术是一种通用思想，在整合的过程中会逐步摸索出整合的套路，而且适用性非常强，经过若干种技术的整合后基本上可以总结出一套固定思维。

下面就开始 Spring Boot 整合 redis，操作步骤如下：

<b>步骤①</b>：导入 Spring Boot 整合 redis 的 starter 坐标

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-data-redis</artifactId>
</dependency>
```

上述坐标可以在创建模块的时候通过勾选的形式进行选择，归属 NoSQL 分类中

<div align="center"><img src="img/boot/image-20220224101142220.png" alt="image-20220224101142220" style="zoom:50%;" /></div>

<b>步骤②</b>：进行基础配置

```yaml
Spring:
  redis:
    host: localhost
    port: 6379
```

操作 redis，最基本的信息就是操作哪一台 redis 服务器，所以服务器地址属于基础配置信息，不可缺少。但是即便你不配置，目前也是可以用的。因为以上两组信息都有默认配置，刚好就是上述配置值。

<b>步骤③</b>：使用 Spring Boot 整合 redis 的专用客户端接口操作，此处使用的是 RedisTemplate

```java
@SpringBootTest
class Spring Boot16RedisApplicationTests {
    @Autowired
    private RedisTemplate redisTemplate;
    @Test
    void set() {
        ValueOperations ops = redisTemplate.opsForValue();
        ops.set("age",41);
    }
    @Test
    void get() {
        ValueOperations ops = redisTemplate.opsForValue();
        Object age = ops.get("name");
        System.out.println(age);
    }
    @Test
    void hset() {
        HashOperations ops = redisTemplate.opsForHash();
        ops.put("info","b","bb");
    }
    @Test
    void hget() {
        HashOperations ops = redisTemplate.opsForHash();
        Object val = ops.get("info", "b");
        System.out.println(val);
    }
}
```

在操作 redis 时，需要先确认操作何种数据，根据数据种类得到操作接口。例如使用 opsForValue() 获取 string 类型的数据操作接口，使用 opsForHash() 获取 hash 类型的数据操作接口，剩下的就是调用对应 api 操作了。各种类型的数据操作接口如下：

<div align="center"><img src="img/boot/image-20220224103104908.png" alt="image-20220224103104908" style="zoom:80%;" /></div>

<b>总结</b>

1. Spring Boot 整合 redis 步骤
    1. 导入 Spring Boot 整合 redis 的 starter 坐标
    2. 进行基础配置
    3. 使用 Spring Boot 整合 redis 的专用客户端接口 RedisTemplate 操作

<b>StringRedisTemplate</b>

由于 redis 内部不提供 java 对象的存储格式，因此当操作的数据以对象的形式存在时，会进行转码，转换成字符串格式后进行操作。为了方便开发者使用基于字符串为数据的操作，Spring Boot 整合 redis 时提供了专用的 API 接口 StringRedisTemplate，可以理解为这是 RedisTemplate 的一种指定数据泛型的操作 API。

```JAVA
@SpringBootTest
public class StringRedisTemplateTest {
    @Autowired
    private StringRedisTemplate stringRedisTemplate;
    @Test
    void get(){
        ValueOperations<String, String> ops = stringRedisTemplate.opsForValue();
        String name = ops.get("name");
        System.out.println(name);
    }
}
```

<b>redis客户端选择</b>

Spring Boot 整合 redis 技术提供了多种客户端兼容模式，默认提供的是 lettucs 客户端技术，也可以根据需要切换成指定客户端技术，例如 jedis 客户端技术，切换成 jedis 客户端技术操作步骤如下：

<b>步骤①</b>：导入 jedis 坐标

```xml
<dependency>
    <groupId>redis.clients</groupId>
    <artifactId>jedis</artifactId>
</dependency>
```

jedis 坐标受 Spring Boot 管理，无需提供版本号

<b>步骤②</b>：配置客户端技术类型，设置为 jedis

```yaml
Spring:
  redis:
    host: localhost
    port: 6379
    client-type: jedis
```

<b>步骤③</b>：根据需要设置对应的配置

```yaml
Spring:
  redis:
    host: localhost
    port: 6379
    client-type: jedis
    lettuce:
      pool:
        max-active: 16
    jedis:
      pool:
        max-active: 16
```

<b>lettcus 与 jedis 区别</b>

- jedis 连接 Redis 服务器是直连模式，当多线程模式下使用 jedis 会存在线程安全问题，解决方案可以通过配置连接池使每个连接专用，这样整体性能就大受影响
- lettcus 基于 Netty 框架进行与 Redis 服务器连接，底层设计中采用 StatefulRedisConnection。 StatefulRedisConnection 自身是线程安全的，可以保障并发访问安全问题，所以一个连接可以被多线程复用。当然 lettcus 也支持多连接实例一起工作

<b>总结</b>

1. Spring Boot 整合 redis 提供了 StringRedisTemplate 对象，以字符串的数据格式操作 redis
2. 如果需要切换 redis 客户端实现技术，可以通过配置的形式进行

#### 整合MongoDB

使用 Redis 技术可以有效的提高数据访问速度，但是由于 Redis 的数据格式单一性，无法操作结构化数据，当操作对象型的数据时，Redis 就显得捉襟见肘。在保障访问速度的情况下，如果想操作结构化数据，看来 Redis 无法满足要求了，此时需要使用全新的数据存储来解决此问题。

MongoDB 是一个开源、高性能、无模式的文档型数据库，它是 NoSQL 数据库产品中的一种，是最像关系型数据库的非关系型数据库。

上述描述中几个词，其中对于我们最陌生的词是无模式的。什么叫无模式呢？简单说就是作为一款数据库，没有固定的数据存储结构，第一条数据可能有 A、B、C 一共 3 个字段，第二条数据可能有 D、E、F 也是 3 个字段，第三条数据可能是 A、C、E 3 个字段，也就是说数据的结构不固定，这就是无模式。这有什么用？灵活，随时变更，不受约束。基于上述特点，MongoDB 的应用面也会产生一些变化。以下列出了一些可以使用 MongoDB 作为数据存储的场景，但是并不是必须使用 MongoDB 的场景：

- 淘宝用户数据
    - 存储位置：数据库
    - 特征：永久性存储，修改频度极低
- 游戏装备数据、游戏道具数据
    - 存储位置：数据库、Mongodb
    - 特征：永久性存储与临时存储相结合、修改频度较高
- 直播数据、打赏数据、粉丝数据
    - 存储位置：数据库、Mongodb
    - 特征：永久性存储与临时存储相结合，修改频度极高
- 物联网数据
    - 存储位置：Mongodb
    - 特征：临时存储，修改频度飞速

##### 安装

windows 版安装包下载地址：https://www.mongodb.com/try/download

下载的安装包也有两种形式，一种是一键安装的 msi 文件，还有一种是解压缩就能使用的 zip 文件，哪种形式都行，本课程采用解压缩 zip 文件进行安装。

解压缩完毕后会得到如下文件，其中 bin 目录包含了所有 mongodb 的可执行命令

<div align="center"><img src="img/boot/image-20220224111306933.png"/></div>

mongodb 在运行时需要指定一个数据存储的目录，所以创建一个数据存储目录，通常放置在安装目录中，此处创建 data 的目录用来存储数据，具体如下

<div align="center"><img src="img/boot/image-20220224111053408.png"/></div>

如果在安装的过程中出现了如下警告信息，就是告诉你，你当前的操作系统缺少了一些系统文件，这个不用担心。

<div align="center"><img src="img/boot/image-20220224113956882.png" style="zoom:50%;" /></div>

根据下列方案即可解决，在浏览器中搜索提示缺少的名称对应的文件，并下载，将下载的文件拷贝到 windows 安装目录的 system32 目录下，然后在命令行中执行 regsvr32 命令注册此文件。根据下载的文件名不同，执行命令前更改对应名称。

```CMD
regsvr32 vcruntime140_1.dll
```

<b>启动服务器</b>

```CMD
mongod --dbpath=..\data\db
```

启动服务器时需要指定数据存储位置，通过参数 --dbpath 进行设置，可以根据需要自行设置数据存储路径。默认服务端口 27017。

<b>启动客户端</b>

```CMD
mongo --host=127.0.0.1 --port=27017
```

##### 基本操作

MongoDB 虽然是一款数据库，但是有类似于 Navicat 的数据库客户端软件，能够便捷的操作 MongoDB，先安装一个客户端，再来操作 MongoDB。

同类型的软件较多，本次安装的软件时 Robo3t，Robot3t 是一款绿色软件，无需安装，解压缩即可。解压缩完毕后进入安装目录双击 robot3t.exe 即可使用。

<div align="center"><img src="img/boot/image-20220224114911573.png" alt="image-20220224114911573" style="zoom: 33%;" /></div>

打开软件首先要连接 MongoDB 服务器，选择【File】菜单，选择【Connect...】

<div align="center"><img src="img/boot/image-20220224115202422.png"></div>

进入连接管理界面后，选择左上角的【Create】链接，创建新的连接设置

<div align="center"><img src="img/boot/image-20220224115254200.png" alt="image-20220224115254200" style="zoom:80%;" /></div>

如果输入设置值即可连接（默认不修改即可连接本机 27017 端口）

<div align="center"><img src="img/boot/image-20220224115300266.png"></div>

连接成功后在命令输入区域输入命令即可操作 MongoDB。

创建数据库：在左侧菜单中使用右键创建，输入数据库名称即可

创建集合：在 Collections 上使用右键创建，输入集合名称即可，集合等同于数据库中的表的作用

新增文档：（文档是一种类似 json 格式的数据，初学者可以先把数据理解为就是 json 数据）	

```CMD
db.集合名称.insert/save/insertOne(文档)
```

删除文档：

```CMD
db.集合名称.remove(条件)
```

修改文档：

```cmd
db.集合名称.update(条件，{操作种类:{文档}})
```

查询文档：

```CMD
基础查询
查询全部：		   db.集合.find();
查第一条：		   db.集合.findOne()
查询指定数量文档：	db.集合.find().limit(10)					//查10条文档
跳过指定数量文档：	db.集合.find().skip(20)					//跳过20条文档
统计：			  	db.集合.count()
排序：				db.集合.sort({age:1})						//按age升序排序
投影：				db.集合名称.find(条件,{name:1,age:1})		 //仅保留name与age域

条件查询
基本格式：			db.集合.find({条件})
模糊查询：			db.集合.find({域名:/正则表达式/})		  //等同SQL中的like，比like强大，可以执行正则所有规则
条件比较运算：		   db.集合.find({域名:{$gt:值}})				//等同SQL中的数值比较操作，例如：name>18
包含查询：			db.集合.find({域名:{$in:[值1，值2]}})		//等同于SQL中的in
条件连接查询：		   db.集合.find({$and:[{条件1},{条件2}]})	   //等同于SQL中的and、or
```

有关 MongoDB 的基础操作就普及到这里，需要全面掌握 MongoDB 技术，请参看相关教程学习。

##### 整合

使用 Spring Boot 整合 MongDB 该如何进行呢？其实 Spring Boot 为什么使用的开发者这么多，就是因为他的套路几乎完全一样。导入坐标，做配置，使用 API 接口操作。整合 Redis 如此，整合 MongoDB 同样如此。

第一，先导入对应技术的整合 starter 坐标

第二，配置必要信息

第三，使用提供的 API 操作即可

下面就开始 Spring Boot 整合 MongoDB，操作步骤如下：

<b>步骤①</b>：导入 Spring Boot 整合 MongoDB 的 starter 坐标

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-data-mongodb</artifactId>
</dependency>
```

上述坐标也可以在创建模块的时候通过勾选的形式进行选择，同样归属 NoSQL 分类中

<div align="center"><img src="img/boot/image-20220224120721626.png" alt="image-20220224120721626" style="zoom: 67%;" /></div>

<b>步骤②</b>：进行基础配置

```yaml
Spring:
  data:
    mongodb:
      uri: mongodb://localhost/itheima
```

操作 MongoDB 需要的配置与操作 redis 一样，最基本的信息都是操作哪一台服务器，区别就是连接的服务器 IP 地址和端口不同，书写格式不同而已。

<b>步骤③</b>：使用 Spring Boot 整合 MongoDB 的专用客户端接口 MongoTemplate 来进行操作

```java
@SpringBootTest
class Spring Boot17MongodbApplicationTests {
    @Autowired
    private MongoTemplate mongoTemplate;
    @Test
    void contextLoads() {
        Book book = new Book();
        book.setId(2);
        book.setName("Spring Boot2");
        book.setType("Spring Boot2");
        book.setDescription("Spring Boot2");
        mongoTemplate.save(book);
    }
    @Test
    void find(){
        List<Book> all = mongoTemplate.findAll(Book.class);
        System.out.println(all);
    }
}
```

<b>总结</b>

1. Spring Boot 整合 MongoDB 步骤
    1. 导入 Spring Boot 整合 MongoDB 的 starter 坐标
    2. 进行基础配置
    3. 使用 Spring Boot 整合 MongoDB 的专用客户端接口 MongoTemplate 操作

#### 整合ES

NoSQL 解决方案已经讲完了两种技术的整合了，Redis 可以使用内存加载数据并实现数据快速访问，MongoDB 可以在内存中存储类似对象的数据并实现数据的快速访问，在企业级开发中对于速度的追求是永无止境的。下面要讲的内容也是一款 NoSQL 解决方案，只不过他的作用不是为了直接加速数据的读写，而是加速数据的查询的，叫做 ES 技术。

ES（Elasticsearch）是一个分布式全文搜索引擎，重点是全文搜索。

那什么是全文搜索呢？比如用户要买一本书，以 Java 为关键字进行搜索，不管是书名中还是书的介绍中，甚至是书的作者名字，只要包含 java 就作为查询结果返回给用户查看，上述过程就使用了全文搜索技术。搜索的条件不再是仅用于对某一个字段进行比对，而是在一条数据中使用搜索条件去比对更多的字段，只要能匹配上就列入查询结果，这就是全文搜索的目的。而 ES 技术就是一种可以实现上述效果的技术。

要实现全文搜索的效果，不可能使用数据库中 like 操作去进行比对，这种效率太低了。ES 设计了一种全新的思想，来实现全文搜索。具体操作过程如下：

1. 将被查询的字段的数据全部文本信息进行查分，分成若干个词

    - 例如“中华人民共和国”就会被拆分成三个词，分别是“中华”、“人民”、“共和国”，此过程有专业术语叫做分词。分词的策略不同，分出的效果不一样，不同的分词策略称为分词器。

2. 将分词得到的结果存储起来，对应每条数据的 id

    - 例如 id 为 1 的数据中名称这一项的值是“中华人民共和国”，那么分词结束后，就会出现“中华”对应 id 为 1，“人民”对应 id 为 1，“共和国”对应 id 为 1
    - 例如 id 为 2 的数据中名称这一项的值是“人民代表大会“，那么分词结束后，就会出现“人民”对应 id 为 2，“代表”对应 id 为 2，“大会”对应 id 为 2
    - 此时就会出现如下对应结果，按照上述形式可以对所有文档进行分词。需要注意分词的过程不是仅对一个字段进行，而是对每一个参与查询的字段都执行，最终结果汇总到一个表格中

    | 分词结果关键字 | 对应id |
    | -------------- | ------ |
    | 中华           | 1      |
    | 人民           | 1,2    |
    | 共和国         | 1      |
    | 代表           | 2      |
    | 大会           | 2      |

3. 当进行查询时，如果输入“人民”作为查询条件，可以通过上述表格数据进行比对，得到 id 值 1,2，然后根据id值就可以得到查询的结果数据了。

上述过程中分词结果关键字内容每一个都不相同，作用有点类似于数据库中的索引，是用来加速数据查询的。但是数据库中的索引是对某一个字段进行添加索引，而这里的分词结果关键字不是一个完整的字段值，只是一个字段中的其中的一部分内容。并且索引使用时是根据索引内容查找整条数据，全文搜索中的分词结果关键字查询后得到的并不是整条的数据，而是数据的id，要想获得具体数据还要再次查询，因此这里为这种分词结果关键字起了一个全新的名称，叫做<b>倒排索引</b>。

##### 安装

windows 版安装包下载地址：[https://](https://www.elastic.co/cn/downloads/elasticsearch)[www.elastic.co/cn/downloads/elasticsearch](https://www.elastic.co/cn/downloads/elasticsearch)

下载的安装包是解压缩就能使用的 zip 文件，解压缩完毕后会得到如下文件

<div align="center"><img src="img/boot/image-20220225132756400.png"></div>

- bin 目录：包含所有的可执行命令
- config 目录：包含 ES 服务器使用的配置文件
- jdk 目录：此目录中包含了一个完整的 jdk 工具包，版本 17，当 ES 升级时，使用最新版本的 jdk 确保不会出现版本支持性不足的问题
- lib 目录：包含 ES 运行的依赖 jar 文件
- logs 目录：包含 ES 运行后产生的所有日志文件
- modules 目录：包含 ES 软件中所有的功能模块，也是一个一个的 jar 包。和 jar 目录不同，jar 目录是 ES 运行期间依赖的 jar 包，modules 是 ES 软件自己的功能 jar 包
- plugins 目录：包含 ES 软件安装的插件，默认为空

<b>启动服务器</b>

```CMD
elasticsearch.bat
```

双击 elasticsearch.bat 文件即可启动ES服务器，默认服务端口 9200。通过浏览器访问 http://localhost:9200 看到如下信息视为ES服务器正常启动

```CMD
{
  "name" : "CZBK-<b><b><b><b><b>",
  "cluster_name" : "elasticsearch",
  "cluster_uuid" : "j137DSswTPG8U4Yb-0T1Mg",
  "version" : {
    "number" : "7.16.2",
    "build_flavor" : "default",
    "build_type" : "zip",
    "build_hash" : "2b937c44140b6559905130a8650c64dbd0879cfb",
    "build_date" : "2021-12-18T19:42:46.604893745Z",
    "build_snapshot" : false,
    "lucene_version" : "8.10.1",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  },
  "tagline" : "You Know, for Search"
}
```

##### 基本操作

ES 中保存有我们要查询的数据，只不过格式和数据库存储数据格式不同而已。在ES中我们要先创建倒排索引，这个索引的功能又点类似于数据库的表，然后将数据添加到倒排索引中，添加的数据称为文档。所以要进行ES的操作要先创建索引，再添加文档，这样才能进行后续的查询操作。

要操作 ES 可以通过 Rest 风格的请求来进行，也就是说发送一个请求就可以执行一个操作。比如新建索引，删除索引这些操作都可以使用发送请求的形式来进行。

- 创建索引，books 是索引名称，下同

    ```CMD
    PUT请求		http://localhost:9200/books
    ```

    发送请求后，看到如下信息即索引创建成功

    ```json
    {
        "acknowledged": true,
        "shards_acknowledged": true,
        "index": "books"
    }
    ```

    重复创建已经存在的索引会出现错误信息，reason 属性中描述错误原因

    ```json
    {
        "error": {
            "root_cause": [
                {
                    "type": "resource_already_exists_exception",
                    "reason": "index [books/VgC_XMVAQmedaiBNSgO2-w] already exists",
                    "index_uuid": "VgC_XMVAQmedaiBNSgO2-w",
                    "index": "books"
                }
            ],
            "type": "resource_already_exists_exception",
            "reason": "index [books/VgC_XMVAQmedaiBNSgO2-w] already exists",	# books索引已经存在
            "index_uuid": "VgC_XMVAQmedaiBNSgO2-w",
            "index": "book"
        },
        "status": 400
    }
    ```

- 查询索引

    ```CMD
    GET请求		http://localhost:9200/books
    ```

    查询索引得到索引相关信息，如下

    ```json
    {
        "book": {
            "aliases": {},
            "mappings": {},
            "settings": {
                "index": {
                    "routing": {
                        "allocation": {
                            "include": {
                                "_tier_preference": "data_content"
                            }
                        }
                    },
                    "number_of_shards": "1",
                    "provided_name": "books",
                    "creation_date": "1645768584849",
                    "number_of_replicas": "1",
                    "uuid": "VgC_XMVAQmedaiBNSgO2-w",
                    "version": {
                        "created": "7160299"
                    }
                }
            }
        }
    }
    ```

    如果查询了不存在的索引，会返回错误信息，例如查询名称为 book 的索引后信息如下

    ```json
    {
        "error": {
            "root_cause": [
                {
                    "type": "index_not_found_exception",
                    "reason": "no such index [book]",
                    "resource.type": "index_or_alias",
                    "resource.id": "book",
                    "index_uuid": "_na_",
                    "index": "book"
                }
            ],
            "type": "index_not_found_exception",
            "reason": "no such index [book]",		# 没有book索引
            "resource.type": "index_or_alias",
            "resource.id": "book",
            "index_uuid": "_na_",
            "index": "book"
        },
        "status": 404
    }
    ```

- 删除索引

    ```CMD
    DELETE请求	http://localhost:9200/books
    ```

    删除所有后，给出删除结果

    ```json
    {
        "acknowledged": true
    }
    ```

    如果重复删除，会给出错误信息，同样在 reason 属性中描述具体的错误原因

    ```JSON
    {
        "error": {
            "root_cause": [
                {
                    "type": "index_not_found_exception",
                    "reason": "no such index [books]",
                    "resource.type": "index_or_alias",
                    "resource.id": "book",
                    "index_uuid": "_na_",
                    "index": "book"
                }
            ],
            "type": "index_not_found_exception",
            "reason": "no such index [books]",		# 没有books索引
            "resource.type": "index_or_alias",
            "resource.id": "book",
            "index_uuid": "_na_",
            "index": "book"
        },
        "status": 404
    }
    ```

- 创建索引并指定分词器

    前面创建的索引是未指定分词器的，可以在创建索引时添加请求参数，设置分词器。目前国内较为流行的分词器是 IK 分词器，使用前先在下对应的分词器，然后使用。IK 分词器下载地址：https://github.com/medcl/elasticsearch-analysis-ik/releases

    分词器下载后解压到 ES 安装目录的 plugins 目录中即可，安装分词器后需要重新启动 ES 服务器。使用 IK 分词器创建索引格式：

    ```json
    PUT请求		http://localhost:9200/books
    
    请求参数如下（注意是json格式的参数）
    {
        "mappings":{							#定义mappings属性，替换创建索引时对应的mappings属性		
            "properties":{						#定义索引中包含的属性设置
                "id":{							#设置索引中包含id属性
                    "type":"keyword"			#当前属性可以被直接搜索
                },
                "name":{						#设置索引中包含name属性
                    "type":"text",              #当前属性是文本信息，参与分词  
                    "analyzer":"ik_max_word",   #使用IK分词器进行分词             
                    "copy_to":"all"				#分词结果拷贝到all属性中
                },
                "type":{
                    "type":"keyword"
                },
                "description":{
                    "type":"text",	                
                    "analyzer":"ik_max_word",                
                    "copy_to":"all"
                },
                "all":{							#定义属性，用来描述多个字段的分词结果集合，当前属性可以参与查询
                    "type":"text",	                
                    "analyzer":"ik_max_word"
                }
            }
        }
    }
    ```

    创建完毕后返回结果和不使用分词器创建索引的结果是一样的，此时可以通过查看索引信息观察到添加的请求参数 mappings 已经进入到了索引属性中

    ```json
    {
        "books": {
            "aliases": {},
            "mappings": {						#mappings属性已经被替换
                "properties": {
                    "all": {
                        "type": "text",
                        "analyzer": "ik_max_word"
                    },
                    "description": {
                        "type": "text",
                        "copy_to": [
                            "all"
                        ],
                        "analyzer": "ik_max_word"
                    },
                    "id": {
                        "type": "keyword"
                    },
                    "name": {
                        "type": "text",
                        "copy_to": [
                            "all"
                        ],
                        "analyzer": "ik_max_word"
                    },
                    "type": {
                        "type": "keyword"
                    }
                }
            },
            "settings": {
                "index": {
                    "routing": {
                        "allocation": {
                            "include": {
                                "_tier_preference": "data_content"
                            }
                        }
                    },
                    "number_of_shards": "1",
                    "provided_name": "books",
                    "creation_date": "1645769809521",
                    "number_of_replicas": "1",
                    "uuid": "DohYKvr_SZO4KRGmbZYmTQ",
                    "version": {
                        "created": "7160299"
                    }
                }
            }
        }
    }
    ```

目前我们已经有了索引了，但是索引中还没有数据，所以要先添加数据，ES 中称数据为文档，下面进行文档操作。

- 添加文档，有三种方式

    ```json
    POST请求	http://localhost:9200/books/_doc		#使用系统生成id
    POST请求	http://localhost:9200/books/_create/1	#使用指定id
    POST请求	http://localhost:9200/books/_doc/1		#使用指定id，不存在创建，存在更新（版本递增）
    
    文档通过请求参数传递，数据格式json
    {
        "name":"Spring Boot",
        "type":"Spring Boot",
        "description":"Spring Boot"
    }  
    ```

- 查询文档

    ```json
    GET请求	http://localhost:9200/books/_doc/1		 #查询单个文档 		
    GET请求	http://localhost:9200/books/_search		 #查询全部文档
    ```

- 条件查询

    ```json
    GET请求	http://localhost:9200/books/_search?q=name:Springboot	# q=查询属性名:查询属性值
    ```

- 删除文档

    ```json
    DELETE请求	http://localhost:9200/books/_doc/1
    ```

- 修改文档（全量更新）

    ```json
    PUT请求	http://localhost:9200/books/_doc/1
    
    文档通过请求参数传递，数据格式json
    {
        "name":"Spring Boot",
        "type":"Spring Boot",
        "description":"Spring Boot"
    }
    ```

- 修改文档（部分更新）

    ```json
    POST请求	http://localhost:9200/books/_update/1
    
    文档通过请求参数传递，数据格式json
    {			
        "doc":{						#部分更新并不是对原始文档进行更新，而是对原始文档对象中的doc属性中的指定属性更新
            "name":"Spring Boot"		#仅更新提供的属性值，未提供的属性值不参与更新操作
        }
    }
    ```

##### 整合

<b>步骤①</b>：导入 Spring Boot 整合 ES 的 starter 坐标

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-data-elasticsearch</artifactId>
</dependency>
```

<b>步骤②</b>：进行基础配置

```yaml
Spring:
  elasticsearch:
    rest:
      uris: http://localhost:9200
```

配置 ES 服务器地址，端口 9200

<b>步骤③</b>：使用 Spring Boot 整合 ES 的专用客户端接口 ElasticsearchRestTemplate 来进行操作

```java
@SpringBootTest
class Spring Boot18EsApplicationTests {
    @Autowired
    private ElasticsearchRestTemplate template;
}
```

上述操作形式是 ES 早期的操作方式，使用的客户端被称为 Low Level Client，这种客户端操作方式性能方面略显不足，于是 ES 开发了全新的客户端操作方式，称为 High Level Client。高级别客户端与 ES 版本同步更新，但是 Spring Boot 最初整合 ES 的时候使用的是低级别客户端，所以企业开发需要更换成高级别的客户端模式。

下面使用高级别客户端方式进行 Spring Boot 整合 ES，操作步骤如下：

<b>步骤①</b>：导入 Spring Boot 整合 ES 高级别客户端的坐标，此种形式目前没有对应的 starter

```xml
<dependency>
    <groupId>org.elasticsearch.client</groupId>
    <artifactId>elasticsearch-rest-high-level-client</artifactId>
</dependency>
```

<b>步骤②</b>：使用编程的形式设置连接的 ES 服务器，并获取客户端对象

```java
@SpringBootTest
class Spring Boot18EsApplicationTests {
    private RestHighLevelClient client;
      @Test
      void testCreateClient() throws IOException {
          HttpHost host = HttpHost.create("http://localhost:9200");
          RestClientBuilder builder = RestClient.builder(host);
          client = new RestHighLevelClient(builder);
  
          client.close();
      }
}
```

配置 ES 服务器地址与端口 9200，记得客户端使用完毕需要手工关闭。由于当前客户端是手工维护的，因此不能通过自动装配的形式加载对象。

<b>步骤③</b>：使用客户端对象操作 ES，例如创建索引

```java
@SpringBootTest
class Spring Boot18EsApplicationTests {
    private RestHighLevelClient client;
      @Test
      void testCreateIndex() throws IOException {
          HttpHost host = HttpHost.create("http://localhost:9200");
          RestClientBuilder builder = RestClient.builder(host);
          client = new RestHighLevelClient(builder);
          
          CreateIndexRequest request = new CreateIndexRequest("books");
          client.indices().create(request, RequestOptions.DEFAULT); 
          
          client.close();
      }
}
```

高级别客户端操作是通过发送请求的方式完成所有操作的，ES 针对各种不同的操作，设定了各式各样的请求对象，上例中创建索引的对象是 CreateIndexRequest，其他操作也会有自己专用的 Request 对象。

当前操作我们发现，无论进行 ES 何种操作，第一步永远是获取 RestHighLevelClient 对象，最后一步永远是关闭该对象的连接。在测试中可以使用测试类的特性去帮助开发者一次性的完成上述操作，但是在业务书写时，还需要自行管理。将上述代码格式转换成使用测试类的初始化方法和销毁方法进行客户端对象的维护。

```JAVA
@SpringBootTest
class Spring Boot18EsApplicationTests {
    @BeforeEach		//在测试类中每个操作运行前运行的方法
    void setUp() {
        HttpHost host = HttpHost.create("http://localhost:9200");
        RestClientBuilder builder = RestClient.builder(host);
        client = new RestHighLevelClient(builder);
    }

    @AfterEach		//在测试类中每个操作运行后运行的方法
    void tearDown() throws IOException {
        client.close();
    }

    private RestHighLevelClient client;

    @Test
    void testCreateIndex() throws IOException {
        CreateIndexRequest request = new CreateIndexRequest("books");
        client.indices().create(request, RequestOptions.DEFAULT);
    }
}
```

现在的书写简化了很多，也更合理。下面使用上述模式将所有的 ES 操作执行一遍，测试结果

<b>创建索引（IK分词器）</b>：

```java
@Test
void testCreateIndexByIK() throws IOException {
    CreateIndexRequest request = new CreateIndexRequest("books");
    String json = "{\n" +
            "    \"mappings\":{\n" +
            "        \"properties\":{\n" +
            "            \"id\":{\n" +
            "                \"type\":\"keyword\"\n" +
            "            },\n" +
            "            \"name\":{\n" +
            "                \"type\":\"text\",\n" +
            "                \"analyzer\":\"ik_max_word\",\n" +
            "                \"copy_to\":\"all\"\n" +
            "            },\n" +
            "            \"type\":{\n" +
            "                \"type\":\"keyword\"\n" +
            "            },\n" +
            "            \"description\":{\n" +
            "                \"type\":\"text\",\n" +
            "                \"analyzer\":\"ik_max_word\",\n" +
            "                \"copy_to\":\"all\"\n" +
            "            },\n" +
            "            \"all\":{\n" +
            "                \"type\":\"text\",\n" +
            "                \"analyzer\":\"ik_max_word\"\n" +
            "            }\n" +
            "        }\n" +
            "    }\n" +
            "}";
    //设置请求中的参数
    request.source(json, XContentType.JSON);
    client.indices().create(request, RequestOptions.DEFAULT);
}
```

IK 分词器是通过请求参数的形式进行设置的，设置请求参数使用 request 对象中的 source 方法进行设置，至于参数是什么，取决于你的操作种类。当请求中需要参数时，均可使用当前形式进行参数设置。	

<b>添加文档</b>：

```java
@Test
//添加文档
void testCreateDoc() throws IOException {
    Book book = bookDao.selectById(1);
    IndexRequest request = new IndexRequest("books").id(book.getId().toString());
    String json = JSON.toJSONString(book);
    request.source(json,XContentType.JSON);
    client.index(request,RequestOptions.DEFAULT);
}
```

添加文档使用的请求对象是 IndexRequest，与创建索引使用的请求对象不同。	

<b>批量添加文档</b>：

```java
@Test
//批量添加文档
void testCreateDocAll() throws IOException {
    List<Book> bookList = bookDao.selectList(null);
    BulkRequest bulk = new BulkRequest();
    for (Book book : bookList) {
        IndexRequest request = new IndexRequest("books").id(book.getId().toString());
        String json = JSON.toJSONString(book);
        request.source(json,XContentType.JSON);
        bulk.add(request);
    }
    client.bulk(bulk,RequestOptions.DEFAULT);
}
```

批量做时，先创建一个 BulkRequest 的对象，可以将该对象理解为是一个保存 request 对象的容器，将所有的请求都初始化好后，添加到 BulkRequest 对象中，再使用 BulkRequest 对象的 bulk 方法，一次性执行完毕。

<b>按id查询文档</b>：

```java
@Test
//按id查询
void testGet() throws IOException {
    GetRequest request = new GetRequest("books","1");
    GetResponse response = client.get(request, RequestOptions.DEFAULT);
    String json = response.getSourceAsString();
    System.out.println(json);
}
```

根据 id 查询文档使用的请求对象是 GetRequest。

<b>按条件查询文档</b>：

```java
@Test
//按条件查询
void testSearch() throws IOException {
    SearchRequest request = new SearchRequest("books");

    SearchSourceBuilder builder = new SearchSourceBuilder();
    builder.query(QueryBuilders.termQuery("all","Spring"));
    request.source(builder);

    SearchResponse response = client.search(request, RequestOptions.DEFAULT);
    SearchHits hits = response.getHits();
    for (SearchHit hit : hits) {
        String source = hit.getSourceAsString();
        //System.out.println(source);
        Book book = JSON.parseObject(source, Book.class);
        System.out.println(book);
    }
}
```

按条件查询文档使用的请求对象是 SearchRequest，查询时调用 SearchRequest 对象的 termQuery 方法，需要给出查询属性名，此处支持使用合并字段，也就是前面定义索引属性时添加的all属性。

Spring Boot 整合 ES 的操作到这里就说完了，与前期进行 Spring Boot 整合 redis 和 mongodb 的差别还是蛮大的，主要原始就是我们没有使用 Spring Boot 整合 ES 的客户端对象。至于操作，由于 ES 操作种类过多，所以显得操作略微有点复杂。有关 Spring Boot 整合 ES 就先学习到这里吧。

<b>总结</b>

1. Spring Boot 整合 ES 步骤
    1. 导入Spring Boot 整合 ES 的 High Level Client 坐标
    2. 手工管理客户端对象，包括初始化和关闭操作
    3. 使用 High Level Client 根据操作的种类不同，选择不同的 Request 对象完成对应操作

## 整合第三方技术

通过第四章的学习，我们领略到了 Spring Boot 在整合第三方技术时强大的一致性，在第五章中我们要使用 Spring Boot 继续整合各种各样的第三方技术，通过本章的学习，可以将之前学习的 Spring Boot 整合第三方技术的思想贯彻到底，还是那三板斧。导坐标、做配置、调 API。

Spring Boot 能够整合的技术实在是太多了，可以说是万物皆可整。本章将从企业级开发中常用的一些技术作为出发点，对各种各样的技术进行整合。

### 缓存

企业级应用主要作用是信息处理，当需要读取数据时，由于受限于数据库的访问效率，导致整体系统性能偏低。

<div align="center"><img src="img/boot/image-20220226154148303.png" alt="image-20220226154148303" style="zoom:67%;" /><br>应用程序直接与数据库打交道，访问效率低</div>							          

为了改善上述现象，开发者通常会在应用程序与数据库之间建立一种临时的数据存储机制，该区域中的数据在内存中保存，读写速度较快，可以有效解决数据库访问效率低下的问题。这一块临时存储数据的区域就是缓存。

<div align="center"><img src="img/boot/image-20220226154233010.png" alt="image-20220226154233010" style="zoom:67%;" /><br>使用缓存后，应用程序与缓存打交道，缓存与数据库打交道，数据访问效率提高</div>

缓存是什么？缓存是一种介于数据永久存储介质与应用程序之间的数据临时存储介质，使用缓存可以有效的减少低速数据读取过程的次数（例如磁盘 IO），提高系统性能。此外缓存不仅可以用于提高永久性存储介质的数据读取效率，还可以提供临时的数据存储空间。而 Spring Boot 提供了对市面上几乎所有的缓存技术进行整合的方案，下面就一起开启 Spring Boot 整合缓存之旅。

#### Spring Boot内置缓存解决方案

Spring Boot 技术提供有内置的缓存解决方案，可以帮助开发者快速开启缓存技术，并使用缓存技术进行数据的快速操作，例如读取缓存数据和写入数据到缓存。

<b>步骤①</b>：导入 Spring Boot 提供的缓存技术对应的 starter

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-cache</artifactId>
</dependency>
```

<b>步骤②</b>：启用缓存，在引导类上方标注注解 @EnableCaching 配置 Spring Boot 程序中可以使用缓存

```java
@SpringBootApplication
//开启缓存功能
@EnableCaching
public class Spring Boot19CacheApplication {
    public static void main(String[] args) {
        SpringApplication.run(Spring Boot19CacheApplication.class, args);
    }
}
```

<b>步骤③</b>：设置操作的数据是否使用缓存

```java
@Service
public class BookServiceImpl implements BookService {
    @Autowired
    private BookDao bookDao;

    @Cacheable(value="cacheSpace",key="#id")
    public Book getById(Integer id) {
        return bookDao.selectById(id);
    }
}
```

在业务方法上面使用注解 @Cacheable 声明当前方法的返回值放入缓存中，其中要指定缓存的存储位置，以及缓存中保存当前方法返回值对应的名称。上例中 value 属性描述缓存的存储位置，可以理解为是一个存储空间名，key 属性描述了缓存中保存数据的名称，使用 #id 读取形参中的 id 值作为缓存名称。

使用 @Cacheable 注解后，执行当前操作，如果发现对应名称在缓存中没有数据，就正常读取数据，然后放入缓存；如果对应名称在缓存中有数据，就终止当前业务方法执行，直接返回缓存中的数据。

#### 手机验证码案例

为了便于下面演示各种各样的缓存技术，我们创建一个手机验证码的案例环境，模拟使用缓存保存手机验证码的过程。

手机验证码案例需求如下：

- 输入手机号获取验证码，组织文档以短信形式发送给用户（页面模拟）
- 输入手机号和验证码验证结果

为了描述上述操作，我们制作两个表现层接口，一个用来模拟发送短信的过程，其实就是根据用户提供的手机号生成一个验证码，然后放入缓存，另一个用来模拟验证码校验的过程，其实就是使用传入的手机号和验证码进行匹配，并返回最终匹配结果。下面直接制作本案例的模拟代码，先以上例中 Spring Boot 提供的内置缓存技术来完成当前案例的制作。

<b>步骤①</b>：导入 Spring Boot 提供的缓存技术对应的 starter

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-cache</artifactId>
</dependency>
```

<b>步骤②</b>：启用缓存，在引导类上方标注注解 @EnableCaching 配置 Spring Boot 程序中可以使用缓存

```java
@SpringBootApplication
//开启缓存功能
@EnableCaching
public class Spring Boot19CacheApplication {
    public static void main(String[] args) {
        SpringApplication.run(Spring Boot19CacheApplication.class, args);
    }
}
```

<b>步骤③</b>：定义验证码对应的实体类，封装手机号与验证码两个属性

```java
@Data
public class SMSCode {
    private String tele;
    private String code;
}
```

<b>步骤④</b>：定义验证码功能的业务层接口与实现类

```java
public interface SMSCodeService {
    public String sendCodeToSMS(String tele);
    public boolean checkCode(SMSCode smsCode);
}

@Service
public class SMSCodeServiceImpl implements SMSCodeService {
    @Autowired
    private CodeUtils codeUtils;

    @CachePut(value = "smsCode", key = "#tele")
    public String sendCodeToSMS(String tele) {
        String code = codeUtils.generator(tele);
        return code;
    }

    public boolean checkCode(SMSCode smsCode) {
        //取出内存中的验证码与传递过来的验证码比对，如果相同，返回true
        String code = smsCode.getCode();
        String cacheCode = codeUtils.get(smsCode.getTele());
        return code.equals(cacheCode);
    }
}
```

获取验证码后，当验证码失效时必须重新获取验证码，因此在获取验证码的功能上不能使用 @Cacheable 注解，@Cacheable 注解是缓存中没有值则放入值，缓存中有值则取值。此处的功能仅仅是生成验证码并放入缓存，并不具有从缓存中取值的功能，因此不能使用 @Cacheable 注解，应该使用仅具有向缓存中保存数据的功能，使用 @CachePut 注解即可。

对于校验验证码的功能建议放入工具类中进行。

<b>步骤⑤</b>：定义验证码的生成策略与根据手机号读取验证码的功能

```java
@Component
public class CodeUtils {
    private String [] patch = {"000000","00000","0000","000","00","0",""};

    public String generator(String tele){
        int hash = tele.hashCode();
        int encryption = 20206666;
        long result = hash ^ encryption;
        long nowTime = System.currentTimeMillis();
        result = result ^ nowTime;
        long code = result % 1000000;
        code = code < 0 ? -code : code;
        String codeStr = code + "";
        int len = codeStr.length();
        return patch[len] + codeStr;
    }

    @Cacheable(value = "smsCode",key="#tele")
    public String get(String tele){
        return null;
    }
}
```

<b>步骤⑥</b>：定义验证码功能的 web 层接口，一个方法用于提供手机号获取验证码，一个方法用于提供手机号和验证码进行校验

```java
@RestController
@RequestMapping("/sms")
public class SMSCodeController {
    @Autowired
    private SMSCodeService smsCodeService;
    
    @GetMapping
    public String getCode(String tele){
        String code = smsCodeService.sendCodeToSMS(tele);
        return code;
    }
    
    @PostMapping
    public boolean checkCode(SMSCode smsCode){
        return smsCodeService.checkCode(smsCode);
    }
}
```

#### Spring Boot整合Ehcache缓存

手机验证码的案例已经完成了，下面就开始 Spring Boot 整合各种各样的缓存技术，第一个整合 Ehcache 技术。Ehcache 是一种缓存技术，使用 Spring Boot 整合 Ehcache 其实就是变更一下缓存技术的实现方式。

<b>步骤①</b>：导入 Ehcache 的坐标

```xml
<dependency>
    <groupId>net.sf.ehcache</groupId>
    <artifactId>ehcache</artifactId>
</dependency>
```

此处为什么不是导入 Ehcache 的 starter，而是导入技术坐标呢？其实 Spring Boot 整合缓存技术做的是通用格式，不管你整合哪种缓存技术，只是实现变化了，操作方式一样。这也体现出 Spring Boot 技术的优点，统一同类技术的整合方式。

<b>步骤②</b>：配置缓存技术实现使用Ehcache

```yaml
Spring:
  cache:
    type: ehcache
    ehcache:
      config: ehcache.xml
```

配置缓存的类型 type 为 ehcache，此处需要说明一下，当前 Spring Boot 可以整合的缓存技术中包含有 ehcach，所以可以这样书写。其实这个 type 不可以随便写的，不是随便写一个名称就可以整合的。

由于 ehcache 的配置有独立的配置文件格式，因此还需要指定 ehcache 的配置文件，以便于读取相应配置

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ehcache xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:noNamespaceSchemaLocation="http://ehcache.org/ehcache.xsd"
         updateCheck="false">
    <diskStore path="D:\ehcache" />

    <!--默认缓存策略 -->
    <!-- external：是否永久存在，设置为true则不会被清除，此时与timeout冲突，通常设置为false-->
    <!-- diskPersistent：是否启用磁盘持久化-->
    <!-- maxElementsInMemory：最大缓存数量-->
    <!-- overflowToDisk：超过最大缓存数量是否持久化到磁盘-->
    <!-- timeToIdleSeconds：最大不活动间隔，设置过长缓存容易溢出，设置过短无效果，可用于记录时效性数据，例如验证码-->
    <!-- timeToLiveSeconds：最大存活时间-->
    <!-- memoryStoreEvictionPolicy：缓存清除策略-->
    <defaultCache
        eternal="false"
        diskPersistent="false"
        maxElementsInMemory="1000"
        overflowToDisk="false"
        timeToIdleSeconds="60"
        timeToLiveSeconds="60"
        memoryStoreEvictionPolicy="LRU" />

    <cache
        name="smsCode"
        eternal="false"
        diskPersistent="false"
        maxElementsInMemory="1000"
        overflowToDisk="false"
        timeToIdleSeconds="10"
        timeToLiveSeconds="10"
        memoryStoreEvictionPolicy="LRU" />
</ehcache>
```

注意前面的案例中，设置了数据保存的位置是 smsCode

```java
@CachePut(value = "smsCode", key = "#tele")
public String sendCodeToSMS(String tele) {
    String code = codeUtils.generator(tele);
    return code;
}	
```

这个设定需要保障 ehcache 中有一个缓存空间名称叫做 smsCode 的配置，前后要统一。在企业开发过程中，通过设置不同名称的 cache 来设定不同的缓存策略，应用于不同的缓存数据。

到这里 Spring Boot 整合 Ehcache 就做完了，可以发现一点，原始代码没有任何修改，仅仅是加了一组配置就可以变更缓存供应商了，这也是 Spring Boot 提供了统一的缓存操作接口的优势，变更实现并不影响原始代码的书写。

<b>总结</b>

1. Spring Boot 使用 Ehcache 作为缓存实现需要导入 Ehcache 的坐标
2. 修改设置，配置缓存供应商为 ehcache，并提供对应的缓存配置文件

#### Spring Boot整合Redis缓存

上节使用 Ehcache 替换了 Spring Boot 内置的缓存技术，其实 Spring Boot 支持的缓存技术还很多，下面使用 redis 技术作为缓存解决方案来实现手机验证码案例。

比对使用 Ehcache 的过程，加坐标，改缓存实现类型为 ehcache，做 Ehcache 的配置。如果还成 redis 做缓存呢？一模一样，加坐标，改缓存实现类型为 redis，做 redis 的配置。差别之处只有一点，redis 的配置可以在 yml 文件中直接进行配置，无需制作独立的配置文件。

<b>步骤①</b>：导入redis的坐标

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-data-redis</artifactId>
</dependency>
```

<b>步骤②</b>：配置缓存技术实现使用 redis

```yaml
Spring:
  redis:
    host: localhost
    port: 6379
  cache:
    type: redis
```

如果需要对 redis 作为缓存进行配置，注意不是对原始的 redis 进行配置，而是配置 redis 作为缓存使用相关的配置，隶属于 Spring.cache.redis 节点下，注意不要写错位置了。

```yaml
Spring:
  redis:
    host: localhost
    port: 6379
  cache:
    type: redis
    redis:
      use-key-prefix: false
      key-prefix: sms_
      cache-null-values: false
      time-to-live: 10s
```

<b>总结</b>

1. Spring Boot 使用 redis 作为缓存实现需要导入redis的坐标
2. 修改设置，配置缓存供应商为 redis，并提供对应的缓存配置

#### Spring Boot整合Memcached缓存

目前我们已经掌握了 3 种缓存解决方案的配置形式，分别是 Spring Boot 内置缓存，ehcache 和 redis，本节研究一下国内比较流行的一款缓存 memcached。

按照之前的套路，其实变更缓存并不繁琐，但是 Spring Boot 并没有支持使用 memcached 作为其缓存解决方案，也就是说在 type 属性中没有 memcached 的配置选项，这里就需要更变一下处理方式了。在整合之前先安装 memcached。

<b>安装</b>

windows 版安装包下载地址：https://www.runoob.com/memcached/window-install-memcached.html

下载的安装包是解压缩就能使用的 zip 文件，解压缩完毕后会得到如下文件

<div align="center"><img src="img/boot/image-20220226174957040.png"/></div>

可执行文件只有一个 memcached.exe，使用该文件可以将 memcached 作为系统服务启动，执行此文件时会出现报错信息，如下：

<div align="center"><img src="img/boot/image-20220226175141986.png" alt="image-20220226175141986" style="zoom:80%;" /></div>

此处出现问题的原因是注册系统服务时需要使用管理员权限，当前账号权限不足导致安装服务失败，切换管理员账号权限启动命令行

<div align="center"><img src="img/boot/image-20220226175302903.png" alt="image-20220226175302903" style="zoom:80%;" /></div>

然后再次执行安装服务的命令即可，如下：

```CMD
memcached.exe -d install
```

服务安装完毕后可以使用命令启动和停止服务，如下：

```cmd
memcached.exe -d start		# 启动服务
memcached.exe -d stop		# 停止服务
```

也可以在任务管理器中进行服务状态的切换

<div align="center"><img src="img/boot/image-20220226175441675.png" alt="image-20220226175441675" style="zoom:67%;" /></div>

<b>变更缓存为Memcached</b>

由于 memcached 未被 Spring Boot 收录为缓存解决方案，因此使用 memcached 需要通过手工硬编码的方式来使用，于是前面的套路都不适用了，需要自己写了。

memcached 目前提供有三种客户端技术，分别是 Memcached Client for Java、SpyMemcached 和 Xmemcached，其中性能指标各方面最好的客户端是 Xmemcached，本次整合就使用这个作为客户端实现技术了。下面开始使用 Xmemcached

<b>步骤①</b>：导入 xmemcached 的坐标

```xml
<dependency>
    <groupId>com.googlecode.xmemcached</groupId>
    <artifactId>xmemcached</artifactId>
    <version>2.4.7</version>
</dependency>
```

<b>步骤②</b>：配置 memcached，制作 memcached 的配置类

```java
@Configuration
public class XMemcachedConfig {
    @Bean
    public MemcachedClient getMemcachedClient() throws IOException {
        MemcachedClientBuilder memcachedClientBuilder = new XMemcachedClientBuilder("localhost:11211");
        MemcachedClient memcachedClient = memcachedClientBuilder.build();
        return memcachedClient;
    }
}
```

memcached 默认对外服务端口 11211。

<b>步骤③</b>：使用 xmemcached 客户端操作缓存，注入 MemcachedClient 对象

```java
@Service
public class SMSCodeServiceImpl implements SMSCodeService {
    @Autowired
    private CodeUtils codeUtils;
    @Autowired
    private MemcachedClient memcachedClient;

    public String sendCodeToSMS(String tele) {
        String code = codeUtils.generator(tele);
        try {
            memcachedClient.set(tele,10,code);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return code;
    }

    public boolean checkCode(SMSCode smsCode) {
        String code = null;
        try {
            code = memcachedClient.get(smsCode.getTele()).toString();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return smsCode.getCode().equals(code);
    }
}
```

设置值到缓存中使用 set 操作，取值使用 get 操作，其实更符合我们开发者的习惯。

上述代码中对于服务器的配置使用硬编码写死到了代码中，将此数据提取出来，做成独立的配置属性。

<b>定义配置属性</b>

以下过程采用前期学习的属性配置方式进行，当前操作有助于理解原理篇中的很多知识。

- 定义配置类，加载必要的配置属性，读取配置文件中 memcached 节点信息

    ```java
    @Component
    @ConfigurationProperties(prefix = "memcached")
    @Data
    public class XMemcachedProperties {
        private String servers;
        private int poolSize;
        private long opTimeout;
    }
    ```

- 定义 memcached 节点信息

    ```yaml
    memcached:
      servers: localhost:11211
      poolSize: 10
      opTimeout: 3000
    ```

- 在 memcached 配置类中加载信息

```java
@Configuration
public class XMemcachedConfig {
    @Autowired
    private XMemcachedProperties props;
    @Bean
    public MemcachedClient getMemcachedClient() throws IOException {
        MemcachedClientBuilder memcachedClientBuilder = new XMemcachedClientBuilder(props.getServers());
        memcachedClientBuilder.setConnectionPoolSize(props.getPoolSize());
        memcachedClientBuilder.setOpTimeout(props.getOpTimeout());
        MemcachedClient memcachedClient = memcachedClientBuilder.build();
        return memcachedClient;
    }
}
```

<b>总结</b>

1. memcached 安装后需要启动对应服务才可以对外提供缓存功能，安装 memcached 服务需要基于 windows 系统管理员权限
2. 由于 Spring Boot 没有提供对 memcached 的缓存整合方案，需要采用手工编码的形式创建 xmemcached 客户端操作缓存
3. 导入 xmemcached 坐标后，创建 memcached 配置类，注册 MemcachedClient 对应的 bean，用于操作缓存
4. 初始化 MemcachedClient 对象所需要使用的属性可以通过自定义配置属性类的形式加载

<b>思考</b>

到这里已经完成了三种缓存的整合，其中 redis 和 mongodb 需要安装独立的服务器，连接时需要输入对应的服务器地址，这种是远程缓存，Ehcache 是一个典型的内存级缓存，因为它什么也不用安装，启动后导入 jar 包就有缓存功能了。这个时候就要问了，能不能这两种缓存一起用呢？

#### Spring Boot整合jetcache缓存

目前我们使用的缓存都是要么 A 要么 B，能不能 AB 一起用呢？这一节就解决这个问题。Spring Boot 针对缓存的整合仅仅停留在用缓存上面，如果缓存自身不支持同时支持 AB 一起用，Spring Boot 也没办法，所以要想解决 AB 缓存一起用的问题，就必须找一款缓存能够支持 AB 两种缓存一起用，有这种缓存吗？还真有，阿里出品，jetcache。

jetcache 严格意义上来说，并不是一个缓存解决方案，只能说他算是一个缓存框架，然后把别的缓存放到 jetcache 中管理，这样就可以支持 AB 缓存一起用了。并且 jetcache 参考了 Spring Boot 整合缓存的思想，整体技术使用方式和 Spring Boot 的缓存解决方案思想非常类似。下面咱们就先把 jetcache 用起来，然后再说它里面的一些小的功能。

做之前要先明确一下，jetcache 并不是随便拿两个缓存都能拼到一起去的。目前 jetcache 支持的缓存方案本地缓存支持两种，远程缓存支持两种，分别如下：

- 本地缓存（Local）
    - LinkedHashMap
    - Caffeine
- 远程缓存（Remote）
    - Redis
    - Tair

为什么 jetcache 只支持 2+2 这么 4 款缓存呢？阿里研发这个技术其实主要是为了满足自身的使用需要。最初肯定只有 1+1 种，逐步变化成 2+2 种。下面就以 LinkedHashMap+Redis 的方案实现本地与远程缓存方案同时使用。

##### 纯远程方案

<b>步骤①</b>：导入 Spring Boot 整合 jetcache 对应的坐标 starter，当前坐标默认使用的远程方案是 redis

```xml
<dependency>
    <groupId>com.alicp.jetcache</groupId>
    <artifactId>jetcache-starter-redis</artifactId>
    <version>2.6.2</version>
</dependency>
```

<b>步骤②</b>：远程方案基本配置

```yaml
jetcache:
  remote:
    default:
      type: redis
      host: localhost
      port: 6379
      poolConfig:
        maxTotal: 50
```

其中 poolConfig 是必配项，否则会报错

<b>步骤③</b>：启用缓存，在引导类上方标注注解 @EnableCreateCacheAnnotation 配置 Spring Boot 程序中可以使用注解的形式创建缓存

```java
@SpringBootApplication
//jetcache启用缓存的主开关
@EnableCreateCacheAnnotation
public class Spring Boot20JetCacheApplication {
    public static void main(String[] args) {
        SpringApplication.run(Spring Boot20JetCacheApplication.class, args);
    }
}
```

<b>步骤④</b>：创建缓存对象 Cache，并使用注解 @CreateCache 标记当前缓存的信息，然后使用 Cache 对象的 API 操作缓存，put 写缓存，get 读缓存。

```java
@Service
public class SMSCodeServiceImpl implements SMSCodeService {
    @Autowired
    private CodeUtils codeUtils;
    
    @CreateCache(name="jetCache_",expire = 10,timeUnit = TimeUnit.SECONDS)
    private Cache<String ,String> jetCache;

    public String sendCodeToSMS(String tele) {
        String code = codeUtils.generator(tele);
        jetCache.put(tele,code);
        return code;
    }

    public boolean checkCode(SMSCode smsCode) {
        String code = jetCache.get(smsCode.getTele());
        return smsCode.getCode().equals(code);
    }
}
```

通过上述 jetcache 使用远程方案连接 redis 可以看出，jetcache 操作缓存时的接口操作更符合开发者习惯，使用缓存就先获取缓存对象 Cache，放数据进去就是 put，取数据出来就是 get，更加简单易懂。并且 jetcache 操作缓存时，可以为某个缓存对象设置过期时间，将同类型的数据放入缓存中，方便有效周期的管理。

上述方案中使用的是配置中定义的 default 缓存，其实这个 default 是个名字，可以随便写，也可以随便加。例如再添加一种缓存解决方案，参照如下配置进行：

```yaml
jetcache:
  remote:
    default:
      type: redis
      host: localhost
      port: 6379
      poolConfig:
        maxTotal: 50
    sms:
      type: redis
      host: localhost
      port: 6379
      poolConfig:
        maxTotal: 50
```

如果想使用名称是 sms 的缓存，需要再创建缓存时指定参数 area，声明使用对应缓存即可

```JAVA
@Service
public class SMSCodeServiceImpl implements SMSCodeService {
    @Autowired
    private CodeUtils codeUtils;
    
    @CreateCache(area="sms",name="jetCache_",expire = 10,timeUnit = TimeUnit.SECONDS)
    private Cache<String ,String> jetCache;

    public String sendCodeToSMS(String tele) {
        String code = codeUtils.generator(tele);
        jetCache.put(tele,code);
        return code;
    }

    public boolean checkCode(SMSCode smsCode) {
        String code = jetCache.get(smsCode.getTele());
        return smsCode.getCode().equals(code);
    }
}
```

##### 纯本地方案

远程方案中，配置中使用 remote 表示远程，换成 local 就是本地，只不过类型不一样而已。

<b>步骤①</b>：导入 Spring Boot 整合 jetcache 对应的坐标 starter

```xml
<dependency>
    <groupId>com.alicp.jetcache</groupId>
    <artifactId>jetcache-starter-redis</artifactId>
    <version>2.6.2</version>
</dependency>
```

<b>步骤②</b>：本地缓存基本配置

```yaml
jetcache:
  local:
    default:
      type: linkedhashmap
      keyConvertor: fastjson
```

为了加速数据获取时 key 的匹配速度，jetcache 要求指定 key 的类型转换器。简单说就是，如果你给了一个 Object 作为 key 的话，我先用 key 的类型转换器给转换成字符串，然后再保存。等到获取数据时，仍然是先使用给定的 Object 转换成字符串，然后根据字符串匹配。由于 jetcache 是阿里的技术，这里推荐 key 的类型转换器使用阿里的 fastjson。

<b>步骤③</b>：启用缓存

```java
@SpringBootApplication
//jetcache启用缓存的主开关
@EnableCreateCacheAnnotation
public class Spring Boot20JetCacheApplication {
    public static void main(String[] args) {
        SpringApplication.run(Spring Boot20JetCacheApplication.class, args);
    }
}
```

<b>步骤④</b>：创建缓存对象 Cache 时，标注当前使用本地缓存

```java
@Service
public class SMSCodeServiceImpl implements SMSCodeService {
    @CreateCache(name="jetCache_",expire = 1000,timeUnit = TimeUnit.SECONDS,cacheType = CacheType.LOCAL)
    private Cache<String ,String> jetCache;

    public String sendCodeToSMS(String tele) {
        String code = codeUtils.generator(tele);
        jetCache.put(tele,code);
        return code;
    }

    public boolean checkCode(SMSCode smsCode) {
        String code = jetCache.get(smsCode.getTele());
        return smsCode.getCode().equals(code);
    }
}
```

cacheType 控制当前缓存使用本地缓存还是远程缓存，配置 cacheType=CacheType.LOCAL 即使用本地缓存。

##### 		本地+远程方案

本地和远程方法都有了，两种方案一起使用如何配置呢？其实就是将两种配置合并到一起就可以了。

```YAML
jetcache:
  local:
    default:
      type: linkedhashmap
      keyConvertor: fastjson
  remote:
    default:
      type: redis
      host: localhost
      port: 6379
      poolConfig:
        maxTotal: 50
    sms:
      type: redis
      host: localhost
      port: 6379
      poolConfig:
        maxTotal: 50
```

在创建缓存的时候，配置 cacheType 为 BOTH 即则本地缓存与远程缓存同时使用。

```java
@Service
public class SMSCodeServiceImpl implements SMSCodeService {
    @CreateCache(name="jetCache_",expire = 1000,timeUnit = TimeUnit.SECONDS,cacheType = CacheType.BOTH)
    private Cache<String ,String> jetCache;
}
```

cacheType 如果不进行配置，默认值是 REMOTE，即仅使用远程缓存方案。关于 jetcache 的配置，参考以下信息

| 属性                                                      | 默认值 | 说明                                                         |
| --------------------------------------------------------- | ------ | ------------------------------------------------------------ |
| jetcache.statIntervalMinutes                              | 0      | 统计间隔，0 表示不统计                                       |
| jetcache.hiddenPackages                                   | 无     | 自动生成 name 时，隐藏指定的包名前缀                         |
| jetcache.[local\|remote].${area}.type                     | 无     | 缓存类型，本地支持 linkedhashmap、caffeine，远程支持 redis、tair |
| jetcache.[local\|remote].${area}.keyConvertor             | 无     | key 转换器，当前仅支持 fastjson                              |
| jetcache.[local\|remote].${area}.valueEncoder             | java   | 仅 remote 类型的缓存需要指定，可选 java 和 kryo              |
| jetcache.[local\|remote].${area}.valueDecoder             | java   | 仅 remote 类型的缓存需要指定，可选 java 和 kryo              |
| jetcache.[local\|remote].${area}.limit                    | 100    | 仅 local 类型的缓存需要指定，缓存实例最大元素数              |
| jetcache.[local\|remote].${area}.expireAfterWriteInMillis | 无穷大 | 默认过期时间，毫秒单位                                       |
| jetcache.local.${area}.expireAfterAccessInMillis          | 0      | 仅 local 类型的缓存有效，毫秒单位，最大不活动间隔            |

以上方案仅支持手工控制缓存，但是 Springcache 方案中的方法缓存特别好用，给一个方法添加一个注解，方法就会自动使用缓存。jetcache 也提供了对应的功能，即方法缓存。

<b>方法缓存</b>

jetcache 提供了方法缓存方案，只不过名称变更了而已。在对应的操作接口上方使用注解 @Cached 即可

<b>步骤①</b>：导入 Spring Boot 整合 jetcache 对应的坐标 starter

```xml
<dependency>
    <groupId>com.alicp.jetcache</groupId>
    <artifactId>jetcache-starter-redis</artifactId>
    <version>2.6.2</version>
</dependency>
```

<b>步骤②</b>：配置缓存

```yaml
jetcache:
  local:
    default:
      type: linkedhashmap
      keyConvertor: fastjson
  remote:
    default:
      type: redis
      host: localhost
      port: 6379
      keyConvertor: fastjson
      valueEncode: java
      valueDecode: java
      poolConfig:
        maxTotal: 50
    sms:
      type: redis
      host: localhost
      port: 6379
      poolConfig:
        maxTotal: 50
```

由于 redis 缓存中不支持保存对象，因此需要对 redis 设置当 Object 类型数据进入到 redis 中时如何进行类型转换。需要配置 keyConvertor 表示 key 的类型转换方式，同时标注 value 的转换类型方式，值进入 redis 时是 java 类型，标注 valueEncode 为 java，值从  redis中读取时转换成 java，标注 valueDecode 为 java。

注意，为了实现 Object 类型的值进出redis，需要保障进出redis的Object类型的数据必须实现序列化接口。

```JAVA
@Data
public class Book implements Serializable {
    private Integer id;
    private String type;
    private String name;
    private String description;
}
```

<b>步骤③</b>：启用缓存时开启方法缓存功能，并配置 basePackages，说明在哪些包中开启方法缓存

```java
@SpringBootApplication
//jetcache启用缓存的主开关
@EnableCreateCacheAnnotation
//开启方法注解缓存
@EnableMethodCache(basePackages = "com.itheima")
public class Spring Boot20JetCacheApplication {
    public static void main(String[] args) {
        SpringApplication.run(Spring Boot20JetCacheApplication.class, args);
    }
}
```

<b>步骤④</b>：使用注解 @Cached 标注当前方法使用缓存

```java
@Service
public class BookServiceImpl implements BookService {
    @Autowired
    private BookDao bookDao;
    
    @Override
    @Cached(name="book_",key="#id",expire = 3600,cacheType = CacheType.REMOTE)
    public Book getById(Integer id) {
        return bookDao.selectById(id);
    }
}
```

##### 远程方案的数据同步

由于远程方案中 redis 保存的数据可以被多个客户端共享，这就存在了数据同步问题。jetcache 提供了 3 个注解解决此问题，分别在更新、删除操作时同步缓存数据，和读取缓存时定时刷新数据

<b>更新缓存</b>

```JAVA
@CacheUpdate(name="book_",key="#book.id",value="#book")
public boolean update(Book book) {
    return bookDao.updateById(book) > 0;
}
```

<b>删除缓存</b>

```JAVA
@CacheInvalidate(name="book_",key = "#id")
public boolean delete(Integer id) {
    return bookDao.deleteById(id) > 0;
}
```

<b>定时刷新缓存</b>

```JAVA
@Cached(name="book_",key="#id",expire = 3600,cacheType = CacheType.REMOTE)
@CacheRefresh(refresh = 5)
public Book getById(Integer id) {
    return bookDao.selectById(id);
}
```

##### 数据报表

jetcache 还提供有简单的数据报表功能，帮助开发者快速查看缓存命中信息，只需要添加一个配置即可

```yaml
jetcache:
  statIntervalMinutes: 1
```

设置后，每 1 分钟在控制台输出缓存数据命中信息

```CMD
[DefaultExecutor] c.alicp.jetcache.support.StatInfoLogger  : jetcache stat from 2022-02-28 09:32:15,892 to 2022-02-28 09:33:00,003
cache    |    qps|   rate|   get|    hit|   fail|   expire|   avgLoadTime|   maxLoadTime
---------+-------+-------+------+-------+-------+---------+--------------+--------------
book_    |   0.66| 75.86%|    29|     22|      0|        0|          28.0|           188
---------+-------+-------+------+-------+-------+---------+--------------+--------------
```

<b>总结</b>

1. jetcache 是一个类似于 Springcache 的缓存解决方案，自身不具有缓存功能，它提供有本地缓存与远程缓存多级共同使用的缓存解决方案
2. jetcache 提供的缓存解决方案受限于目前支持的方案，本地缓存支持两种，远程缓存支持两种
3. 注意数据进入远程缓存时的类型转换问题
4. jetcache 提供方法缓存，并提供了对应的缓存更新与刷新功能
5. jetcache 提供有简单的缓存信息命中报表方便开发者即时监控缓存数据命中情况

<b>思考</b>

jetcache 解决了前期使用缓存方案单一的问题，但是仍然不能灵活的选择缓存进行搭配使用，是否存在一种技术可以灵活的搭配各种各样的缓存使用呢？

#### Spring Boot整合j2cache缓存

jetcache 可以在限定范围内构建多级缓存，但是灵活性不足，不能随意搭配缓存，本节介绍一种可以随意搭配缓存解决方案的缓存整合框架，j2cache。下面就来讲解如何使用这种缓存框架，以 Ehcache 与 redis 整合为例：

<b>步骤①</b>：导入j2cache、redis、ehcache坐标

```xml
<dependency>
    <groupId>net.oschina.j2cache</groupId>
    <artifactId>j2cache-core</artifactId>
    <version>2.8.4-release</version>
</dependency>
<dependency>
    <groupId>net.oschina.j2cache</groupId>
    <artifactId>j2cache-Spring-boot2-starter</artifactId>
    <version>2.8.0-release</version>
</dependency>
<dependency>
    <groupId>net.sf.ehcache</groupId>
    <artifactId>ehcache</artifactId>
</dependency>
```

j2cache 的 starter 中默认包含了 redis 坐标，官方推荐使用 redis 作为二级缓存，因此此处无需导入 redis 坐标

<b>步骤②</b>：配置一级与二级缓存，并配置一二级缓存间数据传递方式，配置书写在名称为 j2cache.properties 的文件中。如果使用 ehcache 还需要单独添加 ehcache 的配置文件

```yaml
# 1级缓存
j2cache.L1.provider_class = ehcache
ehcache.configXml = ehcache.xml

# 2级缓存
j2cache.L2.provider_class = net.oschina.j2cache.cache.support.redis.SpringRedisProvider
j2cache.L2.config_section = redis
redis.hosts = localhost:6379

# 1级缓存中的数据如何到达二级缓存
j2cache.broadcast = net.oschina.j2cache.cache.support.redis.SpringRedisPubSubPolicy
```

此处配置不能乱配置，需要参照官方给出的配置说明进行。例如 1 级供应商选择 ehcache，供应商名称仅仅是一个 ehcache，但是 2 级供应商选择 redis 时要写专用的 Spring 整合 Redis 的供应商类名 SpringRedisProvider，而且这个名称并不是所有的 redis 包中能提供的，也不是 Spring 包中提供的。因此配置 j2cache 必须参照官方文档配置，而且还要去找专用的整合包，导入对应坐标才可以使用。

一级与二级缓存最重要的一个配置就是两者之间的数据沟通方式，此类配置也不是随意配置的，并且不同的缓存解决方案提供的数据沟通方式差异化很大，需要查询官方文档进行设置。

<b>步骤③</b>：使用缓存

```java
@Service
public class SMSCodeServiceImpl implements SMSCodeService {
    @Autowired
    private CodeUtils codeUtils;

    @Autowired
    private CacheChannel cacheChannel;

    public String sendCodeToSMS(String tele) {
        String code = codeUtils.generator(tele);
        cacheChannel.set("sms",tele,code);
        return code;
    }

    public boolean checkCode(SMSCode smsCode) {
        String code = cacheChannel.get("sms",smsCode.getTele()).asString();
        return smsCode.getCode().equals(code);
    }
}
```

j2cache 的使用和 jetcache 比较类似，但是无需开启使用的开关，直接定义缓存对象即可使用，缓存对象名 CacheChannel。

j2cache 的使用不复杂，配置是 j2cache 的核心，毕竟是一个整合型的缓存框架。缓存相关的配置过多，可以查阅 j2cache-core 核心包中的 j2cache.properties 文件中的说明。如下：

```properties
#J2Cache configuration
#########################################
# Cache Broadcast Method
# values:
# jgroups -> use jgroups's multicast
# redis -> use redis publish/subscribe mechanism (using jedis)
# lettuce -> use redis publish/subscribe mechanism (using lettuce, Recommend)
# rabbitmq -> use RabbitMQ publisher/consumer mechanism
# rocketmq -> use RocketMQ publisher/consumer mechanism
# none -> don't notify the other nodes in cluster
# xx.xxxx.xxxx.Xxxxx your own cache broadcast policy classname that implement net.oschina.j2cache.cluster.ClusterPolicy
#########################################
j2cache.broadcast = redis

# jgroups properties
jgroups.channel.name = j2cache
jgroups.configXml = /network.xml

# RabbitMQ properties
rabbitmq.exchange = j2cache
rabbitmq.host = localhost
rabbitmq.port = 5672
rabbitmq.username = guest
rabbitmq.password = guest

# RocketMQ properties
rocketmq.name = j2cache
rocketmq.topic = j2cache
# use ; to split multi hosts
rocketmq.hosts = 127.0.0.1:9876

#########################################
# Level 1&2 provider
# values:
# none -> disable this level cache
# ehcache -> use ehcache2 as level 1 cache
# ehcache3 -> use ehcache3 as level 1 cache
# caffeine -> use caffeine as level 1 cache(only in memory)
# redis -> use redis as level 2 cache (using jedis)
# lettuce -> use redis as level 2 cache (using lettuce)
# readonly-redis -> use redis as level 2 cache ,but never write data to it. if use this provider, you must uncomment `j2cache.L2.config_section` to make the redis configurations available.
# memcached -> use memcached as level 2 cache (xmemcached),
# [classname] -> use custom provider
#########################################

j2cache.L1.provider_class = caffeine
j2cache.L2.provider_class = redis

# When L2 provider isn't `redis`, using `L2.config_section = redis` to read redis configurations
# j2cache.L2.config_section = redis

# Enable/Disable ttl in redis cache data (if disabled, the object in redis will never expire, default:true)
# NOTICE: redis hash mode (redis.storage = hash) do not support this feature)
j2cache.sync_ttl_to_redis = true

# Whether to cache null objects by default (default false)
j2cache.default_cache_null_object = true

#########################################
# Cache Serialization Provider
# values:
# fst -> using fast-serialization (recommend)
# kryo -> using kryo serialization
# json -> using fst's json serialization (testing)
# fastjson -> using fastjson serialization (embed non-static class not support)
# java -> java standard
# fse -> using fse serialization
# [classname implements Serializer]
#########################################

j2cache.serialization = json
#json.map.person = net.oschina.j2cache.demo.Person

#########################################
# Ehcache configuration
#########################################

# ehcache.configXml = /ehcache.xml

# ehcache3.configXml = /ehcache3.xml
# ehcache3.defaultHeapSize = 1000

#########################################
# Caffeine configuration
# caffeine.region.[name] = size, xxxx[s|m|h|d]
#
#########################################
caffeine.properties = /caffeine.properties

#########################################
# Redis connection configuration
#########################################

#########################################
# Redis Cluster Mode
#
# single -> single redis server
# sentinel -> master-slaves servers
# cluster -> cluster servers (数据库配置无效，使用 database = 0）
# sharded -> sharded servers  (密码、数据库必须在 hosts 中指定，且连接池配置无效 ; redis://user:password@127.0.0.1:6379/0）
#
#########################################

redis.mode = single

#redis storage mode (generic|hash)
redis.storage = generic

## redis pub/sub channel name
redis.channel = j2cache
## redis pub/sub server (using redis.hosts when empty)
redis.channel.host =

#cluster name just for sharded
redis.cluster_name = j2cache

## redis cache namespace optional, default[empty]
redis.namespace =

## redis command scan parameter count, default[1000]
#redis.scanCount = 1000

## connection
# Separate multiple redis nodes with commas, such as 192.168.0.10:6379,192.168.0.11:6379,192.168.0.12:6379

redis.hosts = 127.0.0.1:6379
redis.timeout = 2000
redis.password =
redis.database = 0
redis.ssl = false

## redis pool properties
redis.maxTotal = 100
redis.maxIdle = 10
redis.maxWaitMillis = 5000
redis.minEvictableIdleTimeMillis = 60000
redis.minIdle = 1
redis.numTestsPerEvictionRun = 10
redis.lifo = false
redis.softMinEvictableIdleTimeMillis = 10
redis.testOnBorrow = true
redis.testOnReturn = false
redis.testWhileIdle = true
redis.timeBetweenEvictionRunsMillis = 300000
redis.blockWhenExhausted = false
redis.jmxEnabled = false

#########################################
# Lettuce scheme
#
# redis -> single redis server
# rediss -> single redis server with ssl
# redis-sentinel -> redis sentinel
# redis-cluster -> cluster servers
#
#########################################

#########################################
# Lettuce Mode
#
# single -> single redis server
# sentinel -> master-slaves servers
# cluster -> cluster servers (数据库配置无效，使用 database = 0）
# sharded -> sharded servers  (密码、数据库必须在 hosts 中指定，且连接池配置无效 ; redis://user:password@127.0.0.1:6379/0）
#
#########################################

## redis command scan parameter count, default[1000]
#lettuce.scanCount = 1000
lettuce.mode = single
lettuce.namespace =
lettuce.storage = hash
lettuce.channel = j2cache
lettuce.scheme = redis
lettuce.hosts = 127.0.0.1:6379
lettuce.password =
lettuce.database = 0
lettuce.sentinelMasterId =
lettuce.maxTotal = 100
lettuce.maxIdle = 10
lettuce.minIdle = 10
# timeout in milliseconds
lettuce.timeout = 10000
# redis cluster topology refresh interval in milliseconds
lettuce.clusterTopologyRefresh = 3000

#########################################
# memcached server configurations
# refer to https://gitee.com/mirrors/XMemcached
#########################################

memcached.servers = 127.0.0.1:11211
memcached.username =
memcached.password =
memcached.connectionPoolSize = 10
memcached.connectTimeout = 1000
memcached.failureMode = false
memcached.healSessionInterval = 1000
memcached.maxQueuedNoReplyOperations = 100
memcached.opTimeout = 100
memcached.sanitizeKeys = false
```

<b>总结</b>

1. j2cache 是一个缓存框架，自身不具有缓存功能，它提供多种缓存整合在一起使用的方案
2. j2cache 需要通过复杂的配置设置各级缓存，以及缓存之间数据交换的方式
3. j2cache 操作接口通过 CacheChannel 实现

### 任务

Spring Boot 整合第三方技术第二部分我们来说说任务系统，其实这里说的任务系统指的是定时任务。定时任务是企业级开发中必不可少的组成部分，诸如长周期业务数据的计算，例如年度报表，诸如系统脏数据的处理，再比如系统性能监控报告，还有抢购类活动的商品上架，这些都离不开定时任务。本节将介绍两种不同的定时任务技术。

#### Quartz

Quartz 技术是一个比较成熟的定时任务框架，怎么说呢？有点繁琐，用过的都知道，配置略微复杂。Spring Boot 对其进行整合后，简化了一系列的配置，将很多配置采用默认设置，这样开发阶段就简化了很多。再学习 Spring Boot 整合 Quartz 前先普及几个 Quartz 的概念。

- 工作（Job）：用于定义具体执行的工作
- 工作明细（JobDetail）：用于描述定时工作相关的信息
- 触发器（Trigger）：描述了工作明细与调度器的对应关系
- 调度器（Scheduler）：用于描述触发工作的执行规则，通常使用cron表达式定义规则

简单说就是你定时干什么事情，这就是工作，工作不可能就是一个简单的方法，还要设置一些明细信息。工作啥时候执行，设置一个调度器，可以简单理解成设置一个工作执行的时间。工作和调度都是独立定义的，它们两个怎么配合到一起呢？用触发器。下面开始Spring Boot整合Quartz。

<b>步骤①</b>：导入 Spring Boot 整合 Quartz的starter

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-quartz</artifactId>
</dependency>
```

<b>步骤②</b>：定义任务 Bean，按照 Quartz 的开发规范制作，继承 QuartzJobBean

```java
public class MyQuartz extends QuartzJobBean {
    @Override
    protected void executeInternal(JobExecutionContext context) throws JobExecutionException {
        System.out.println("quartz task run...");
    }
}
```

<b>步骤③</b>：创建 Quartz 配置类，定义工作明细（JobDetail）与触发器的（Trigger）bean

```java
@Configuration
public class QuartzConfig {
    @Bean
    public JobDetail printJobDetail(){
        //绑定具体的工作
        return JobBuilder.newJob(MyQuartz.class).storeDurably().build();
    }
    @Bean
    public Trigger printJobTrigger(){
        ScheduleBuilder schedBuilder = CronScheduleBuilder.cronSchedule("0/5 * * * * ?");
        //绑定对应的工作明细
        return TriggerBuilder.newTrigger().forJob(printJobDetail()).withSchedule(schedBuilder).build();
    }
}
```

工作明细中要设置对应的具体工作，使用 newJob() 操作传入对应的工作任务类型即可。

触发器需要绑定任务，使用 forJob() 操作传入绑定的工作明细对象。此处可以为工作明细设置名称然后使用名称绑定，也可以直接调用对应方法绑定。触发器中最核心的规则是执行时间，此处使用调度器定义执行时间，执行时间描述方式使用的是 cron 表达式。有关 cron 表达式的规则，而且格式不能乱设置，不是写个格式就能用的，写不好就会出现冲突问题。

<b>总结</b>

1. Spring Boot 整合 Quartz 就是将 Quartz 对应的核心对象交给 Spring 容器管理，包含两个对象，JobDetail 和 Trigger 对象
2. JobDetail 对象描述的是工作的执行信息，需要绑定一个 QuartzJobBean 类型的对象
3. Trigger 对象定义了一个触发器，需要为其指定绑定的 JobDetail 是哪个，同时要设置执行周期调度器

<b>思考</b>

上面的操作看上去不多，但是 Quartz 将其中的对象划分粒度过细，导致开发的时候有点繁琐，Spring 针对上述规则进行了简化，开发了自己的任务管理组件——Task，如何用呢？

#### Task

Spring 根据定时任务的特征，将定时任务的开发简化到了极致。怎么说呢？要做定时任务总要告诉容器有这功能吧，然后定时执行什么任务直接告诉对应的 bean 什么时间执行就行了，就这么简单，一起来看怎么做

<b>步骤①</b>：开启定时任务功能，在引导类上开启定时任务功能的开关，使用注解 @EnableScheduling

```java
@SpringBootApplication
//开启定时任务功能
@EnableScheduling
public class Spring Boot22TaskApplication {
    public static void main(String[] args) {
        SpringApplication.run(Spring Boot22TaskApplication.class, args);
    }
}
```

<b>步骤②</b>：定义 Bean，在对应要定时执行的操作上方，使用注解 @Scheduled 定义执行的时间，执行时间的描述方式还是 cron 表达式

```java
@Component
public class MyBean {
    @Scheduled(cron = "0/1 * * * * ?")
    public void print(){
        System.out.println(Thread.currentThread().getName()+" :Spring task run...");
    }
}
```

完事，这就完成了定时任务的配置。总体感觉其实什么东西都没少，只不过没有将所有的信息都抽取成 bean，而是直接使用注解绑定定时执行任务的事情而已。

如何想对定时任务进行相关配置，可以通过配置文件进行

```yaml
Spring:
  task:
   	scheduling:
      pool:
       	size: 1							# 任务调度线程池大小 默认 1
      thread-name-prefix: ssm_      	# 调度线程名称前缀 默认 scheduling-      
        shutdown:
          await-termination: false		# 线程池关闭时等待所有任务完成
          await-termination-period: 10s	# 调度线程关闭前最大等待时间，确保最后一定关闭
```

<b>总结</b>

1. Spring task 需要使用注解 @EnableScheduling 开启定时任务功能

2. 为定时执行的的任务设置执行周期，描述方式 cron 表达式


### 邮件

Spring Boot 整合第三方技术第三部分我们来说说邮件系统，发邮件是 java 程序的基本操作，Spring Boot 整合 javamail 其实就是简化开发。不熟悉邮件的可以先学习完 javamail 的基础操作，再来看这一部分内容才能感触到 Spring Boot 整合 javamail 究竟简化了哪些操作。

学习邮件发送之前先了解 3 个概念，这些概念规范了邮件操作过程中的标准。

- SMTP（Simple Mail Transfer Protocol）：简单邮件传输协议，用于<b>发送</b>电子邮件的传输协议
- POP3（Post Office Protocol - Version 3）：用于<b>接收</b>电子邮件的标准协议
- IMAP（Internet Mail Access Protocol）：互联网消息协议，是 POP3 的替代协议

简单说就是 SMPT 是发邮件的标准，POP3 是收邮件的标准，IMAP 是对 POP3 的升级。我们制作程序中操作邮件，通常是发邮件，所以 SMTP 是使用的重点，收邮件大部分都是通过邮件客户端完成，所以开发收邮件的代码极少。除非你要读取邮件内容，然后解析，做邮件功能的统一处理。例如 HR 的邮箱收到求职者的简历，可以读取后统一处理。但是为什么不制作独立的投递简历的系统呢？所以说，好奇怪的需求，因为要想收邮件就要规范发邮件的人的书写格式，这个未免有点强人所难，并且极易收到外部攻击，你不可能使用白名单来收邮件。如果能使用白名单来收邮件然后解析邮件，还不如开发个系统给白名单中的人专用呢，更安全，总之就是鸡肋了。下面就开始学习 Spring Boot 如何整合 javamail 发送邮件。

#### 发送简单邮件

<b>步骤①</b>：导入 Spring Boot 整合 javamail 的 starter

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-mail</artifactId>
</dependency>
```

<b>步骤②</b>：配置邮箱的登录信息

```yaml
Spring:
  mail:
    host: smtp.126.com
    username: test@126.com
    password: test
```

java 程序仅用于发送邮件，邮件的功能还是邮件供应商提供的，所以这里是用别人的邮件服务，要配置对应信息。

host 配置的是提供邮件服务的主机协议，当前程序仅用于发送邮件，因此配置的是 smtp 的协议。

password 并不是邮箱账号的登录密码，是邮件供应商提供的一个加密后的密码，也是为了保障系统安全性。不然外部人员通过地址访问下载了配置文件，直接获取到了邮件密码就会有极大的安全隐患。有关该密码的获取每个邮件供应商提供的方式都不一样，此处略过。可以到邮件供应商的设置页面找 POP3 或 IMAP 这些关键词找到对应的获取位置。下例仅供参考：

<div align="center"><img src="img/boot/image-20220228111251036.png"></div>

<b>步骤③</b>：使用 JavaMailSender 接口发送邮件

```java
@Service
public class SendMailServiceImpl implements SendMailService {
    @Autowired
    private JavaMailSender javaMailSender;

    //发送人
    private String from = "test@qq.com";
    //接收人
    private String to = "test@126.com";
    //标题
    private String subject = "测试邮件";
    //正文
    private String context = "测试邮件正文内容";

    @Override
    public void sendMail() {
        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(from+"(小甜甜)");
        message.setTo(to);
        message.setSubject(subject);
        message.setText(context);
        javaMailSender.send(message);
    }
}
```

将发送邮件的必要信息（发件人、收件人、标题、正文）封装到 SimpleMailMessage 对象中，可以根据规则设置发送人昵称等。

#### 发送多组件邮件（附件、复杂正文）

发送简单邮件仅需要提供对应的 4 个基本信息就可以了，如果想发送复杂的邮件，需要更换邮件对象。使用 MimeMessage 可以发送特殊的邮件。

<b>发送网页正文邮件</b>

```JAVA
@Service
public class SendMailServiceImpl2 implements SendMailService {
    @Autowired
    private JavaMailSender javaMailSender;

    //发送人
    private String from = "test@qq.com";
    //接收人
    private String to = "test@126.com";
    //标题
    private String subject = "测试邮件";
    //正文
    private String context = "<div align="center"><img src='ABC.JPG'/><a href='https://www.itcast.cn'>点开有惊喜</a>";

    public void sendMail() {
        try {
            MimeMessage message = javaMailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message);
            helper.setFrom(to+"(小甜甜)");
            helper.setTo(from);
            helper.setSubject(subject);
            helper.setText(context,true);		//此处设置正文支持html解析

            javaMailSender.send(message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

<b>发送带有附件的邮件</b>

```JAVA
@Service
public class SendMailServiceImpl2 implements SendMailService {
    @Autowired
    private JavaMailSender javaMailSender;

    //发送人
    private String from = "test@qq.com";
    //接收人
    private String to = "test@126.com";
    //标题
    private String subject = "测试邮件";
    //正文
    private String context = "测试邮件正文";

    public void sendMail() {
        try {
            MimeMessage message = javaMailSender.createMimeMessage();
            MimeMessageHelper helper = new MimeMessageHelper(message,true);		//此处设置支持附件
            helper.setFrom(to+"(小甜甜)");
            helper.setTo(from);
            helper.setSubject(subject);
            helper.setText(context);

            //添加附件
            File f1 = new File("Springboot_23_mail-0.0.1-SNAPSHOT.jar");
            File f2 = new File("resources\\logo.png");

            helper.addAttachment(f1.getName(),f1);
            helper.addAttachment("最靠谱的培训结构.png",f2);

            javaMailSender.send(message);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

<b>总结</b>

1. Spring Boot 整合 javamail 其实就是简化了发送邮件的客户端对象 JavaMailSender 的初始化过程，通过配置的形式加载信息简化开发过程


### 消息

Spring Boot 整合第三方技术最后一部分我们来说说消息中间件，首先先介绍一下消息的应用。

#### 消息的概念

从广义角度来说，消息其实就是信息，但是和信息又有所不同。信息通常被定义为一组数据，而消息除了具有数据的特征之外，还有消息的来源与接收的概念。通常发送消息的一方称为消息的生产者，接收消息的一方称为消息的消费者。这样比较后，发现其实消息和信息差别还是很大的。

为什么要设置生产者和消费者呢？这就是要说到消息的意义了。信息通常就是一组数据，但是消息由于有了生产者和消费者，就出现了消息中所包含的信息可以被二次解读，生产者发送消息，可以理解为生产者发送了一个信息，也可以理解为生产者发送了一个命令；消费者接收消息，可以理解为消费者得到了一个信息，也可以理解为消费者得到了一个命令。对比一下我们会发现信息是一个基本数据，而命令则可以关联下一个行为动作，这样就可以理解为基于接收的消息相当于得到了一个行为动作，使用这些行为动作就可以组织成一个业务逻辑，进行进一步的操作。总的来说，消息其实也是一组信息，只是为其赋予了全新的含义，因为有了消息的流动，并且是有方向性的流动，带来了基于流动的行为产生的全新解读。开发者就可以基于消息的这种特殊解，将其换成代码中的指令。

对于消息的理解，初学者总认为消息内部的数据非常复杂，这是一个误区。比如我发送了一个消息，要求接受者翻译发送过去的内容。初学者会认为消息中会包含被翻译的文字，已经本次操作要执行翻译操作而不是打印操作。其实这种现象有点过度解读了，发送的消息中仅仅包含被翻译的文字，但是可以通过控制不同的人接收此消息来确认要做的事情。例如发送被翻译的文字仅到A程序，而A程序只能进行翻译操作，这样就可以发送简单的信息完成复杂的业务了，是通过接收消息的主体不同，进而执行不同的操作，而不会在消息内部定义数据的操作行为，当然如果开发者希望消息中包含操作种类信息也是可以的，只是提出消息的内容可以更简单，更单一。

对于消息的生产者与消费者的工作模式，还可以将消息划分成两种模式，同步消费与异步消息。

所谓同步消息就是生产者发送完消息，等待消费者处理，消费者处理完将结果告知生产者，然后生产者继续向下执行业务。这种模式过于卡生产者的业务执行连续性，在现在的企业级开发中，上述这种业务场景通常不会采用消息的形式进行处理。

所谓异步消息就是生产者发送完消息，无需等待消费者处理完毕，生产者继续向下执行其他动作。比如生产者发送了一个日志信息给日志系统，发送过去以后生产者就向下做其他事情了，无需关注日志系统的执行结果。日志系统根据接收到的日志信息继续进行业务执行，是单纯的记录日志，还是记录日志并报警，这些和生产者无关，这样生产者的业务执行效率就会大幅度提升。并且可以通过添加多个消费者来处理同一个生产者发送的消息来提高系统的高并发性，改善系统工作效率，提高用户体验。一旦某一个消费者由于各种问题宕机了，也不会对业务产生影响，提高了系统的高可用性。

以上简单的介绍了一下消息这种工作模式存在的意义，希望对各位学习者有所帮助。

#### Java处理消息的标准规范

目前企业级开发中广泛使用的消息处理技术共三大类，具体如下：

- JMS
- AMQP
- MQTT

为什么是三大类，而不是三个技术呢？因为这些都是规范，就想 JDBC 技术，是个规范，开发针对规范开发，运行还要靠实现类，例如 MySQL 提供了 JDBC 的实现，最终运行靠的还是实现。并且这三类规范都是针对异步消息进行处理的，也符合消息的设计本质，处理异步的业务。对以上三种消息规范做一下普及

##### JMS

JMS（Java Message Service）,这是一个规范，作用等同于 JDBC 规范，提供了与消息服务相关的 API 接口。

<b>JMS消息模型</b>

JMS 规范中规范了消息有两种模型。分别是<b>点对点模型</b>和<b>发布订阅模型</b>。

<b>点对点模型</b>：peer-2-peer，生产者会将消息发送到一个保存消息的容器中，通常使用队列模型，使用队列保存消息。一个队列的消息只能被一个消费者消费，或未被及时消费导致超时。这种模型下，生产者和消费者是一对一绑定的。

<b>发布订阅模型</b>：publish-subscribe，生产者将消息发送到一个保存消息的容器中，也是使用队列模型来保存。但是消息可以被多个消费者消费，生产者和消费者完全独立，相互不需要感知对方的存在。

以上这种分类是从消息的生产和消费过程来进行区分，针对消息所包含的信息不同，还可以进行不同类别的划分。

<b>JMS消息种类</b>

根据消息中包含的数据种类划分，可以将消息划分成 6 种消息。

- TextMessage
- MapMessage
- BytesMessage
- StreamMessage
- ObjectMessage
- Message （只有消息头和属性）

JMS 主张不同种类的消息，消费方式不同，可以根据使用需要选择不同种类的消息。但是这一点也成为其诟病之处，后面再说。整体上来说，JMS 就是典型的保守派，什么都按照 J2EE 的规范来，做一套规范，定义若干个标准，每个标准下又提供一大批 API。目前对 JMS 规范实现的消息中间件技术还是挺多的，例如 ActiveMQ、Redis、HornetMQ。但是也有一些不太规范的实现，参考 JMS 的标准设计，但是又不完全满足其规范，例如：RabbitMQ、RocketMQ。

##### AMQP

JMS 的问世为消息中间件提供了很强大的规范性支撑，但是使用的过程中就开始被人诟病，比如 JMS 设置的极其复杂的多种类消息处理机制。本来分门别类处理挺好的，为什么会被诟病呢？原因就在于 JMS 的设计是 J2EE 规范，站在 Java 开发的角度思考问题。但是现实往往是复杂度很高的。比如我有一个 .NET 开发的系统 A，有一个 Java 开发的系统 B，现在要从 A 系统给 B 系统发业务消息，结果两边数据格式不统一，没法操作。JMS 不是可以统一数据格式吗？提供了 6 种数据种类，总有一款适合你啊。NO，一个都不能用。因为 A 系统的底层语言不是 Java 语言开发的，根本不支持那些对象。这就意味着如果想使用现有的业务系统A继续开发已经不可能了，必须推翻重新做使用 Java 语言开发的 A 系统。

这时候有人就提出说，你搞那么复杂，整那么多种类干什么？找一种大家都支持的消息数据类型不就解决这个跨平台的问题了吗？大家一想，对啊，于是 AMQP 孕育而生。

单从上面的说明中其实可以明确感知到，AMQP 的出现解决的是消息传递时使用的消息种类的问题，化繁为简，但是其并没有完全推翻 JMS 的操作 API，所以说 AMQP 仅仅是一种协议，规范了数据传输的格式而已。

AMQP（advanced message queuing protocol）：一种协议（高级消息队列协议，也是消息代理规范），规范了网络交换的数据格式，兼容 JMS 操作。
<b>优点</b>

具有跨平台性，服务器供应商，生产者，消费者可以使用不同的语言来实现

<b>JMS消息种类</b>

AMQP 消息种类：byte[]

AMQP 在 JMS 的消息模型基础上又进行了进一步的扩展，除了点对点和发布订阅的模型，开发了几种全新的消息模型，适应各种各样的消息发送。

<b>AMQP消息模型</b>

- direct exchange
- fanout exchange
- topic exchange
- headers exchange
- system exchange

目前实现了 AMQP 协议的消息中间件技术也很多，而且都是较为流行的技术，例如：RabbitMQ、StormMQ、RocketMQ

##### MQTT

MQTT（Message Queueing Telemetry Transport）消息队列遥测传输，专为小设备设计，是物联网（IOT）生态系统中主要成分之一。由于与 JavaEE 企业级开发没有交集，此处不作过多的说明。

除了上述 3 种 J2EE 企业级应用中广泛使用的三种异步消息传递技术，还有一种技术也不能忽略，Kafka。

##### KafKa

Kafka，一种高吞吐量的分布式发布订阅消息系统，提供实时消息功能。Kafka 技术并不是作为消息中间件为主要功能的产品，但是其拥有发布订阅的工作模式，也可以充当消息中间件来使用，而且目前企业级开发中其身影也不少见。

#### 购物订单发送手机短信案例

为了便于下面演示各种各样的消息中间件技术，我们创建一个购物过程生成订单时为用户发送短信的案例环境，模拟使用消息中间件实现发送手机短信的过程。

手机验证码案例需求如下：

- 执行下单业务时（模拟此过程），调用消息服务，将要发送短信的订单id传递给消息中间件
- 消息处理服务接收到要发送的订单id后输出订单 id（模拟发短信）

由于不涉及数据读写，仅开发业务层与表现层，其中短信处理的业务代码独立开发，代码如下：

<b>订单业务</b>

<b>业务层接口</b>

```JAVA
public interface OrderService {
    void order(String id);
}
```

模拟传入订单 id，执行下订单业务，参数为虚拟设定，实际应为订单对应的实体类

<b>业务层实现</b>

```JAVA
@Service
public class OrderServiceImpl implements OrderService {
    @Autowired
    private MessageService messageService;
    
    @Override
    public void order(String id) {
        //一系列操作，包含各种服务调用，处理各种业务
        System.out.println("订单处理开始");
        //短信消息处理
        messageService.sendMessage(id);
        System.out.println("订单处理结束");
        System.out.println();
    }
}
```

业务层转调短信处理的服务 MessageService

<b>表现层服务</b>

```JAVA
@RestController
@RequestMapping("/orders")
public class OrderController {

    @Autowired
    private OrderService orderService;

    @PostMapping("{id}")
    public void order(@PathVariable String id){
        orderService.order(id);
    }
}
```

表现层对外开发接口，传入订单 id 即可（模拟）

<b>短信处理业务</b>

<b>业务层接口</b>

```JAVA
public interface MessageService {
    void sendMessage(String id);
    String doMessage();
}
```

短信处理业务层接口提供两个操作，发送要处理的订单id到消息中间件，另一个操作目前暂且设计成处理消息，实际消息的处理过程不应该是手动执行，应该是自动执行，到具体实现时再进行设计

<b>业务层实现</b>

```JAVA
@Service
public class MessageServiceImpl implements MessageService {
    private ArrayList<String> msgList = new ArrayList<String>();

    @Override
    public void sendMessage(String id) {
        System.out.println("待发送短信的订单已纳入处理队列，id："+id);
        msgList.add(id);
    }

    @Override
    public String doMessage() {
        String id = msgList.remove(0);
        System.out.println("已完成短信发送业务，id："+id);
        return id;
    }
}
```

短信处理业务层实现中使用集合先模拟消息队列，观察效果

<b>表现层服务</b>

```JAVA
@RestController
@RequestMapping("/msgs")
public class MessageController {

    @Autowired
    private MessageService messageService;

    @GetMapping
    public String doMessage(){
        String id = messageService.doMessage();
        return id;
    }
}
```

短信处理表现层接口暂且开发出一个处理消息的入口，但是此业务是对应业务层中设计的模拟接口，实际业务不需要设计此接口。

下面开启 Spring Boot 整合各种各样的消息中间件，从严格满足 JMS 规范的 ActiveMQ 开始

#### 整合ActiveMQ

ActiveMQ 是 MQ 产品中的元老级产品，早期标准 MQ 产品之一，在 AMQP 协议没有出现之前，占据了消息中间件市场的绝大部分份额，后期因为 AMQP 系列产品的出现，迅速走弱，目前仅在一些线上运行的产品中出现，新产品开发较少采用。

##### 安装

windows 版安装包下载地址：[https://activemq.apache.org/components/classic/download](https://activemq.apache.org/components/classic/download/)[/](https://activemq.apache.org/components/classic/download/)

下载的安装包是解压缩就能使用的 zip 文件，解压缩完毕后会得到如下文件

<div align="center"><img src="img/boot/image-20220228160001620.png"></div>

<b>启动服务器</b>

```CMD
activemq.bat
```

运行 bin 目录下的 win32 或 win64 目录下的 activemq.bat 命令即可，根据自己的操作系统选择即可，默认对外服务端口 61616。

<b>访问web管理服务</b>

ActiveMQ 启动后会启动一个 Web 控制台服务，可以通过该服务管理 ActiveMQ。

```CMD
http://127.0.0.1:8161/
```

web 管理服务默认端口 8161，访问后可以打开 ActiveMQ 的管理界面，如下：

<div align="center"><img src="img/boot/image-20220228160844972.png" alt="image-20220228160844972" style="zoom:67%;" /></div>

​		首先输入访问用户名和密码，初始化用户名和密码相同，均为：admin，成功登录后进入管理后台界面，如下：

<div align="center"><img src="img/boot/image-20220228161010401.png"></div>

看到上述界面视为启动 ActiveMQ 服务成功。

<b>启动失败</b>

在 ActiveMQ 启动时要占用多个端口，以下为正常启动信息：

```CMD
wrapper  | --> Wrapper Started as Console
wrapper  | Launching a JVM...
jvm 1    | Wrapper (Version 3.2.3) http://wrapper.tanukisoftware.org
jvm 1    |   Copyright 1999-2006 Tanuki Software, Inc.  All Rights Reserved.
jvm 1    |
jvm 1    | Java Runtime: Oracle Corporation 1.8.0_172 D:\soft\jdk1.8.0_172\jre
jvm 1    |   Heap sizes: current=249344k  free=235037k  max=932352k
jvm 1    |     JVM args: -Dactivemq.home=../.. -Dactivemq.base=../.. -Djavax.net.ssl.keyStorePassword=password -Djavax.net.ssl.trustStorePassword=password -Djavax.net.ssl.keyStore=../../conf/broker.ks -Djavax.net.ssl.trustStore=../../conf/broker.ts -Dcom.sun.management.jmxremote -Dorg.apache.activemq.UseDedicatedTaskRunner=true -Djava.util.logging.config.file=logging.properties -Dactivemq.conf=../../conf -Dactivemq.data=../../data -Djava.security.auth.login.config=../../conf/login.config -Xmx1024m -Djava.library.path=../../bin/win64 -Dwrapper.key=7ySrCD75XhLCpLjd -Dwrapper.port=32000 -Dwrapper.jvm.port.min=31000 -Dwrapper.jvm.port.max=31999 -Dwrapper.pid=9364 -Dwrapper.version=3.2.3 -Dwrapper.native_library=wrapper -Dwrapper.cpu.timeout=10 -Dwrapper.jvmid=1
jvm 1    | Extensions classpath:
jvm 1    |   [..\..\lib,..\..\lib\camel,..\..\lib\optional,..\..\lib\web,..\..\lib\extra]
jvm 1    | ACTIVEMQ_HOME: ..\..
jvm 1    | ACTIVEMQ_BASE: ..\..
jvm 1    | ACTIVEMQ_CONF: ..\..\conf
jvm 1    | ACTIVEMQ_DATA: ..\..\data
jvm 1    | Loading message broker from: xbean:activemq.xml
jvm 1    |  INFO | Refreshing org.apache.activemq.xbean.XBeanBrokerFactory$1@5f3ebfe0: startup date [Mon Feb 28 16:07:48 CST 2022]; root of context hierarchy
jvm 1    |  INFO | Using Persistence Adapter: KahaDBPersistenceAdapter[D:\soft\activemq\bin\win64\..\..\data\kahadb]
jvm 1    |  INFO | KahaDB is version 7
jvm 1    |  INFO | PListStore:[D:\soft\activemq\bin\win64\..\..\data\localhost\tmp_storage] started
jvm 1    |  INFO | Apache ActiveMQ 5.16.3 (localhost, ID:CZBK-20210302VL-10434-1646035669595-0:1) is starting
jvm 1    |  INFO | Listening for connections at: tcp://CZBK-20210302VL:61616?maximumConnections=1000&wireFormat.maxFrameSize=104857600
jvm 1    |  INFO | Connector openwire started
jvm 1    |  INFO | Listening for connections at: amqp://CZBK-20210302VL:5672?maximumConnections=1000&wireFormat.maxFrameSize=104857600
jvm 1    |  INFO | Connector amqp started
jvm 1    |  INFO | Listening for connections at: stomp://CZBK-20210302VL:61613?maximumConnections=1000&wireFormat.maxFrameSize=104857600
jvm 1    |  INFO | Connector stomp started
jvm 1    |  INFO | Listening for connections at: mqtt://CZBK-20210302VL:1883?maximumConnections=1000&wireFormat.maxFrameSize=104857600
jvm 1    |  INFO | Connector mqtt started
jvm 1    |  INFO | Starting Jetty server
jvm 1    |  INFO | Creating Jetty connector
jvm 1    |  WARN | ServletContext@o.e.j.s.ServletContextHandler@7350746f{/,null,STARTING} has uncovered http methods for path: /
jvm 1    |  INFO | Listening for connections at ws://CZBK-20210302VL:61614?maximumConnections=1000&wireFormat.maxFrameSize=104857600
jvm 1    |  INFO | Connector ws started
jvm 1    |  INFO | Apache ActiveMQ 5.16.3 (localhost, ID:CZBK-20210302VL-10434-1646035669595-0:1) started
jvm 1    |  INFO | For help or more information please see: http://activemq.apache.org
jvm 1    |  WARN | Store limit is 102400 mb (current store usage is 0 mb). The data directory: D:\soft\activemq\bin\win64\..\..\data\kahadb only has 68936 mb of usable space. - resetting to maximum available disk space: 68936 mb
jvm 1    |  INFO | ActiveMQ WebConsole available at http://127.0.0.1:8161/
jvm 1    |  INFO | ActiveMQ Jolokia REST API available at http://127.0.0.1:8161/api/jolokia/
```

其中占用的端口有：61616、5672、61613、1883、61614，如果启动失败，请先管理对应端口即可。以下就是某个端口占用的报错信息，可以从抛出异常的位置看出，启动 5672 端口时端口被占用，显示 java.net.BindException: Address already in use: JVM_Bind。Windows 系统中终止端口运行的操作参看[【命令行启动常见问题及解决方案】](#命令行启动常见问题及解决方案)

```CMD
wrapper  | --> Wrapper Started as Console
wrapper  | Launching a JVM...
jvm 1    | Wrapper (Version 3.2.3) http://wrapper.tanukisoftware.org
jvm 1    |   Copyright 1999-2006 Tanuki Software, Inc.  All Rights Reserved.
jvm 1    |
jvm 1    | Java Runtime: Oracle Corporation 1.8.0_172 D:\soft\jdk1.8.0_172\jre
jvm 1    |   Heap sizes: current=249344k  free=235038k  max=932352k
jvm 1    |     JVM args: -Dactivemq.home=../.. -Dactivemq.base=../.. -Djavax.net.ssl.keyStorePassword=password -Djavax.net.ssl.trustStorePassword=password -Djavax.net.ssl.keyStore=../../conf/broker.ks -Djavax.net.ssl.trustStore=../../conf/broker.ts -Dcom.sun.management.jmxremote -Dorg.apache.activemq.UseDedicatedTaskRunner=true -Djava.util.logging.config.file=logging.properties -Dactivemq.conf=../../conf -Dactivemq.data=../../data -Djava.security.auth.login.config=../../conf/login.config -Xmx1024m -Djava.library.path=../../bin/win64 -Dwrapper.key=QPJoy9ZoXeWmmwTS -Dwrapper.port=32000 -Dwrapper.jvm.port.min=31000 -Dwrapper.jvm.port.max=31999 -Dwrapper.pid=14836 -Dwrapper.version=3.2.3 -Dwrapper.native_library=wrapper -Dwrapper.cpu.timeout=10 -Dwrapper.jvmid=1
jvm 1    | Extensions classpath:
jvm 1    |   [..\..\lib,..\..\lib\camel,..\..\lib\optional,..\..\lib\web,..\..\lib\extra]
jvm 1    | ACTIVEMQ_HOME: ..\..
jvm 1    | ACTIVEMQ_BASE: ..\..
jvm 1    | ACTIVEMQ_CONF: ..\..\conf
jvm 1    | ACTIVEMQ_DATA: ..\..\data
jvm 1    | Loading message broker from: xbean:activemq.xml
jvm 1    |  INFO | Refreshing org.apache.activemq.xbean.XBeanBrokerFactory$1@2c9392f5: startup date [Mon Feb 28 16:06:16 CST 2022]; root of context hierarchy
jvm 1    |  INFO | Using Persistence Adapter: KahaDBPersistenceAdapter[D:\soft\activemq\bin\win64\..\..\data\kahadb]
jvm 1    |  INFO | KahaDB is version 7
jvm 1    |  INFO | PListStore:[D:\soft\activemq\bin\win64\..\..\data\localhost\tmp_storage] started
jvm 1    |  INFO | Apache ActiveMQ 5.16.3 (localhost, ID:CZBK-20210302VL-10257-1646035577620-0:1) is starting
jvm 1    |  INFO | Listening for connections at: tcp://CZBK-20210302VL:61616?maximumConnections=1000&wireFormat.maxFrameSize=104857600
jvm 1    |  INFO | Connector openwire started
jvm 1    | ERROR | Failed to start Apache ActiveMQ (localhost, ID:CZBK-20210302VL-10257-1646035577620-0:1)
jvm 1    | java.io.IOException: Transport Connector could not be registered in JMX: java.io.IOException: Failed to bind to server socket: amqp://0.0.0.0:5672?maximumConnections=1000&wireFormat.maxFrameSize=104857600 due to: java.net.BindException: Address already in use: JVM_Bind
jvm 1    |      at org.apache.activemq.util.IOExceptionSupport.create(IOExceptionSupport.java:28)
jvm 1    |      at org.apache.activemq.broker.BrokerService.registerConnectorMBean(BrokerService.java:2288)
jvm 1    |      at org.apache.activemq.broker.BrokerService.startTransportConnector(BrokerService.java:2769)
jvm 1    |      at org.apache.activemq.broker.BrokerService.startAllConnectors(BrokerService.java:2665)
jvm 1    |      at org.apache.activemq.broker.BrokerService.doStartBroker(BrokerService.java:780)
jvm 1    |      at org.apache.activemq.broker.BrokerService.startBroker(BrokerService.java:742)
jvm 1    |      at org.apache.activemq.broker.BrokerService.start(BrokerService.java:645)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerService.afterPropertiesSet(XBeanBrokerService.java:73)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
jvm 1    |      at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
jvm 1    |      at java.lang.reflect.Method.invoke(Method.java:498)
jvm 1    |      at org.Springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.invokeCustomInitMethod(AbstractAutowireCapableBeanFactory.java:1748)
jvm 1    |      at org.Springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.invokeInitMethods(AbstractAutowireCapableBeanFactory.java:1685)
jvm 1    |      at org.Springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.initializeBean(AbstractAutowireCapableBeanFactory.java:1615)
jvm 1    |      at org.Springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.doCreateBean(AbstractAutowireCapableBeanFactory.java:553)
jvm 1    |      at org.Springframework.beans.factory.support.AbstractAutowireCapableBeanFactory.createBean(AbstractAutowireCapableBeanFactory.java:481)
jvm 1    |      at org.Springframework.beans.factory.support.AbstractBeanFactory$1.getObject(AbstractBeanFactory.java:312)
jvm 1    |      at org.Springframework.beans.factory.support.DefaultSingletonBeanRegistry.getSingleton(DefaultSingletonBeanRegistry.java:230)
jvm 1    |      at org.Springframework.beans.factory.support.AbstractBeanFactory.doGetBean(AbstractBeanFactory.java:308)
jvm 1    |      at org.Springframework.beans.factory.support.AbstractBeanFactory.getBean(AbstractBeanFactory.java:197)
jvm 1    |      at org.Springframework.beans.factory.support.DefaultListableBeanFactory.preInstantiateSingletons(DefaultListableBeanFactory.java:756)
jvm 1    |      at org.Springframework.context.support.AbstractApplicationContext.finishBeanFactoryInitialization(AbstractApplicationContext.java:867)
jvm 1    |      at org.Springframework.context.support.AbstractApplicationContext.refresh(AbstractApplicationContext.java:542)
jvm 1    |      at org.apache.xbean.Spring.context.ResourceXmlApplicationContext.<init>(ResourceXmlApplicationContext.java:64)
jvm 1    |      at org.apache.xbean.Spring.context.ResourceXmlApplicationContext.<init>(ResourceXmlApplicationContext.java:52)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory$1.<init>(XBeanBrokerFactory.java:104)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory.createApplicationContext(XBeanBrokerFactory.java:104)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory.createBroker(XBeanBrokerFactory.java:67)
jvm 1    |      at org.apache.activemq.broker.BrokerFactory.createBroker(BrokerFactory.java:71)
jvm 1    |      at org.apache.activemq.broker.BrokerFactory.createBroker(BrokerFactory.java:54)
jvm 1    |      at org.apache.activemq.console.command.StartCommand.runTask(StartCommand.java:87)
jvm 1    |      at org.apache.activemq.console.command.AbstractCommand.execute(AbstractCommand.java:63)
jvm 1    |      at org.apache.activemq.console.command.ShellCommand.runTask(ShellCommand.java:154)
jvm 1    |      at org.apache.activemq.console.command.AbstractCommand.execute(AbstractCommand.java:63)
jvm 1    |      at org.apache.activemq.console.command.ShellCommand.main(ShellCommand.java:104)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
jvm 1    |      at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
jvm 1    |      at java.lang.reflect.Method.invoke(Method.java:498)
jvm 1    |      at org.apache.activemq.console.Main.runTaskClass(Main.java:262)
jvm 1    |      at org.apache.activemq.console.Main.main(Main.java:115)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
jvm 1    |      at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
jvm 1    |      at java.lang.reflect.Method.invoke(Method.java:498)
jvm 1    |      at org.tanukisoftware.wrapper.WrapperSimpleApp.run(WrapperSimpleApp.java:240)
jvm 1    |      at java.lang.Thread.run(Thread.java:748)
jvm 1    | Caused by: java.io.IOException: Failed to bind to server socket: amqp://0.0.0.0:5672?maximumConnections=1000&wireFormat.maxFrameSize=104857600 due to: java.net.BindException: Address already in use: JVM_Bind
jvm 1    |      at org.apache.activemq.util.IOExceptionSupport.create(IOExceptionSupport.java:34)
jvm 1    |      at org.apache.activemq.transport.tcp.TcpTransportServer.bind(TcpTransportServer.java:146)
jvm 1    |      at org.apache.activemq.transport.tcp.TcpTransportFactory.doBind(TcpTransportFactory.java:62)
jvm 1    |      at org.apache.activemq.transport.TransportFactorySupport.bind(TransportFactorySupport.java:40)
jvm 1    |      at org.apache.activemq.broker.TransportConnector.createTransportServer(TransportConnector.java:335)
jvm 1    |      at org.apache.activemq.broker.TransportConnector.getServer(TransportConnector.java:145)
jvm 1    |      at org.apache.activemq.broker.TransportConnector.asManagedConnector(TransportConnector.java:110)
jvm 1    |      at org.apache.activemq.broker.BrokerService.registerConnectorMBean(BrokerService.java:2283)
jvm 1    |      ... 46 more
jvm 1    | Caused by: java.net.BindException: Address already in use: JVM_Bind
jvm 1    |      at java.net.DualStackPlainSocketImpl.bind0(Native Method)
jvm 1    |      at java.net.DualStackPlainSocketImpl.socketBind(DualStackPlainSocketImpl.java:106)
jvm 1    |      at java.net.AbstractPlainSocketImpl.bind(AbstractPlainSocketImpl.java:387)
jvm 1    |      at java.net.PlainSocketImpl.bind(PlainSocketImpl.java:190)
jvm 1    |      at java.net.ServerSocket.bind(ServerSocket.java:375)
jvm 1    |      at java.net.ServerSocket.<init>(ServerSocket.java:237)
jvm 1    |      at javax.net.DefaultServerSocketFactory.createServerSocket(ServerSocketFactory.java:231)
jvm 1    |      at org.apache.activemq.transport.tcp.TcpTransportServer.bind(TcpTransportServer.java:143)
jvm 1    |      ... 52 more
jvm 1    |  INFO | Apache ActiveMQ 5.16.3 (localhost, ID:CZBK-20210302VL-10257-1646035577620-0:1) is shutting down
jvm 1    |  INFO | socketQueue interrupted - stopping
jvm 1    |  INFO | Connector openwire stopped
jvm 1    |  INFO | Could not accept connection during shutdown  : null (null)
jvm 1    |  INFO | Connector amqp stopped
jvm 1    |  INFO | Connector stomp stopped
jvm 1    |  INFO | Connector mqtt stopped
jvm 1    |  INFO | Connector ws stopped
jvm 1    |  INFO | PListStore:[D:\soft\activemq\bin\win64\..\..\data\localhost\tmp_storage] stopped
jvm 1    |  INFO | Stopping async queue tasks
jvm 1    |  INFO | Stopping async topic tasks
jvm 1    |  INFO | Stopped KahaDB
jvm 1    |  INFO | Apache ActiveMQ 5.16.3 (localhost, ID:CZBK-20210302VL-10257-1646035577620-0:1) uptime 0.426 seconds
jvm 1    |  INFO | Apache ActiveMQ 5.16.3 (localhost, ID:CZBK-20210302VL-10257-1646035577620-0:1) is shutdown
jvm 1    |  INFO | Closing org.apache.activemq.xbean.XBeanBrokerFactory$1@2c9392f5: startup date [Mon Feb 28 16:06:16 CST 2022]; root of context hierarchy
jvm 1    |  WARN | Exception encountered during context initialization - cancelling refresh attempt: org.Springframework.beans.factory.BeanCreationException: Error creating bean with name 'org.apache.activemq.xbean.XBeanBrokerService#0' defined in class path resource [activemq.xml]: Invocation of init method failed; nested exception is java.io.IOException: Transport Connector could not be registered in JMX: java.io.IOException: Failed to bind to server socket: amqp://0.0.0.0:5672?maximumConnections=1000&wireFormat.maxFrameSize=104857600 due to: java.net.BindException: Address already in use: JVM_Bind
jvm 1    | ERROR: java.lang.RuntimeException: Failed to execute start task. Reason: java.lang.IllegalStateException: BeanFactory not initialized or already closed - call 'refresh' before accessing beans via the ApplicationContext
jvm 1    | java.lang.RuntimeException: Failed to execute start task. Reason: java.lang.IllegalStateException: BeanFactory not initialized or already closed - call 'refresh' before accessing beans via the ApplicationContext
jvm 1    |      at org.apache.activemq.console.command.StartCommand.runTask(StartCommand.java:91)
jvm 1    |      at org.apache.activemq.console.command.AbstractCommand.execute(AbstractCommand.java:63)
jvm 1    |      at org.apache.activemq.console.command.ShellCommand.runTask(ShellCommand.java:154)
jvm 1    |      at org.apache.activemq.console.command.AbstractCommand.execute(AbstractCommand.java:63)
jvm 1    |      at org.apache.activemq.console.command.ShellCommand.main(ShellCommand.java:104)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
jvm 1    |      at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
jvm 1    |      at java.lang.reflect.Method.invoke(Method.java:498)
jvm 1    |      at org.apache.activemq.console.Main.runTaskClass(Main.java:262)
jvm 1    |      at org.apache.activemq.console.Main.main(Main.java:115)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
jvm 1    |      at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
jvm 1    |      at java.lang.reflect.Method.invoke(Method.java:498)
jvm 1    |      at org.tanukisoftware.wrapper.WrapperSimpleApp.run(WrapperSimpleApp.java:240)
jvm 1    |      at java.lang.Thread.run(Thread.java:748)
jvm 1    | Caused by: java.lang.IllegalStateException: BeanFactory not initialized or already closed - call 'refresh' before accessing beans via the ApplicationContext
jvm 1    |      at org.Springframework.context.support.AbstractRefreshableApplicationContext.getBeanFactory(AbstractRefreshableApplicationContext.java:164)
jvm 1    |      at org.Springframework.context.support.AbstractApplicationContext.destroyBeans(AbstractApplicationContext.java:1034)
jvm 1    |      at org.Springframework.context.support.AbstractApplicationContext.refresh(AbstractApplicationContext.java:555)
jvm 1    |      at org.apache.xbean.Spring.context.ResourceXmlApplicationContext.<init>(ResourceXmlApplicationContext.java:64)
jvm 1    |      at org.apache.xbean.Spring.context.ResourceXmlApplicationContext.<init>(ResourceXmlApplicationContext.java:52)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory$1.<init>(XBeanBrokerFactory.java:104)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory.createApplicationContext(XBeanBrokerFactory.java:104)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory.createBroker(XBeanBrokerFactory.java:67)
jvm 1    |      at org.apache.activemq.broker.BrokerFactory.createBroker(BrokerFactory.java:71)
jvm 1    |      at org.apache.activemq.broker.BrokerFactory.createBroker(BrokerFactory.java:54)
jvm 1    |      at org.apache.activemq.console.command.StartCommand.runTask(StartCommand.java:87)
jvm 1    |      ... 16 more
jvm 1    | ERROR: java.lang.IllegalStateException: BeanFactory not initialized or already closed - call 'refresh' before accessing beans via the ApplicationContext
jvm 1    | java.lang.IllegalStateException: BeanFactory not initialized or already closed - call 'refresh' before accessing beans via the ApplicationContext
jvm 1    |      at org.Springframework.context.support.AbstractRefreshableApplicationContext.getBeanFactory(AbstractRefreshableApplicationContext.java:164)
jvm 1    |      at org.Springframework.context.support.AbstractApplicationContext.destroyBeans(AbstractApplicationContext.java:1034)
jvm 1    |      at org.Springframework.context.support.AbstractApplicationContext.refresh(AbstractApplicationContext.java:555)
jvm 1    |      at org.apache.xbean.Spring.context.ResourceXmlApplicationContext.<init>(ResourceXmlApplicationContext.java:64)
jvm 1    |      at org.apache.xbean.Spring.context.ResourceXmlApplicationContext.<init>(ResourceXmlApplicationContext.java:52)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory$1.<init>(XBeanBrokerFactory.java:104)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory.createApplicationContext(XBeanBrokerFactory.java:104)
jvm 1    |      at org.apache.activemq.xbean.XBeanBrokerFactory.createBroker(XBeanBrokerFactory.java:67)
jvm 1    |      at org.apache.activemq.broker.BrokerFactory.createBroker(BrokerFactory.java:71)
jvm 1    |      at org.apache.activemq.broker.BrokerFactory.createBroker(BrokerFactory.java:54)
jvm 1    |      at org.apache.activemq.console.command.StartCommand.runTask(StartCommand.java:87)
jvm 1    |      at org.apache.activemq.console.command.AbstractCommand.execute(AbstractCommand.java:63)
jvm 1    |      at org.apache.activemq.console.command.ShellCommand.runTask(ShellCommand.java:154)
jvm 1    |      at org.apache.activemq.console.command.AbstractCommand.execute(AbstractCommand.java:63)
jvm 1    |      at org.apache.activemq.console.command.ShellCommand.main(ShellCommand.java:104)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
jvm 1    |      at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
jvm 1    |      at java.lang.reflect.Method.invoke(Method.java:498)
jvm 1    |      at org.apache.activemq.console.Main.runTaskClass(Main.java:262)
jvm 1    |      at org.apache.activemq.console.Main.main(Main.java:115)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
jvm 1    |      at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
jvm 1    |      at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
jvm 1    |      at java.lang.reflect.Method.invoke(Method.java:498)
jvm 1    |      at org.tanukisoftware.wrapper.WrapperSimpleApp.run(WrapperSimpleApp.java:240)
jvm 1    |      at java.lang.Thread.run(Thread.java:748)
wrapper  | <-- Wrapper Stopped
请按任意键继续. . .
```

##### 整合

做了这么多 Spring Boot 整合第三方技术，已经摸到门路了，加坐标，做配置，调接口，直接开工

<b>步骤①</b>：导入 Spring Boot 整合 ActiveMQ 的 starter

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-activemq</artifactId>
</dependency>
```

<b>步骤②</b>：配置 ActiveMQ 的服务器地址

```yaml
Spring:
  activemq:
    broker-url: tcp://localhost:61616
```

<b>步骤③</b>：使用 JmsMessagingTemplate 操作 ActiveMQ

```java
@Service
public class MessageServiceActivemqImpl implements MessageService {
    @Autowired
    private JmsMessagingTemplate messagingTemplate;

    @Override
    public void sendMessage(String id) {
        System.out.println("待发送短信的订单已纳入处理队列，id："+id);
        messagingTemplate.convertAndSend("order.queue.id",id);
    }

    @Override
    public String doMessage() {
        String id = messagingTemplate.receiveAndConvert("order.queue.id",String.class);
        System.out.println("已完成短信发送业务，id："+id);
        return id;
    }
}
```

发送消息需要先将消息的类型转换成字符串，然后再发送，所以是 convertAndSend，定义消息发送的位置，和具体的消息内容，此处使用 id 作为消息内容。

接收消息需要先将消息接收到，然后再转换成指定的数据类型，所以是 receiveAndConvert，接收消息除了提供读取的位置，还要给出转换后的数据的具体类型。

<b>步骤④</b>：使用消息监听器在服务器启动后，监听指定位置，当消息出现后，立即消费消息

```JAVA
@Component
public class MessageListener {
    @JmsListener(destination = "order.queue.id")
    @SendTo("order.other.queue.id")
    public String receive(String id){
        System.out.println("已完成短信发送业务，id："+id);
        return "new:"+id;
    }
}
```

使用注解 @JmsListener 定义当前方法监听 ActiveMQ 中指定名称的消息队列。

如果当前消息队列处理完还需要继续向下传递当前消息到另一个队列中使用注解 @SendTo 即可，这样即可构造连续执行的顺序消息队列。

<b>步骤⑤</b>：切换消息模型由点对点模型到发布订阅模型，修改 jms 配置即可

```yaml
Spring:
  activemq:
    broker-url: tcp://localhost:61616
  jms:
    pub-sub-domain: true
```

pub-sub-domain 默认值为 false，即点对点模型，修改为 true 后就是发布订阅模型。

<b>总结</b>

1. Spring Boot 整合 ActiveMQ 提供了 JmsMessagingTemplate 对象作为客户端操作消息队列
2. 操作 ActiveMQ 需要配置 ActiveMQ 服务器地址，默认端口 61616
3. 企业开发时通常使用监听器来处理消息队列中的消息，设置监听器使用注解 @JmsListener
4. 配置 jms 的 pub-sub-domain 属性可以在点对点模型和发布订阅模型间切换消息模型

#### 整合RabbitMQ

RabbitMQ 是 MQ 产品中的目前较为流行的产品之一，它遵从 AMQP 协议。RabbitMQ 的底层实现语言使用的是 Erlang，所以安装 RabbitMQ 需要先安装 Erlang。

<b>Erlang安装</b>

windows 版安装包下载地址：[https](https://www.erlang.org/downloads)[://www.erlang.org/downloads](https://www.erlang.org/downloads)

下载完毕后得到 exe 安装文件，一键傻瓜式安装，安装完毕需要重启，需要重启，需要重启。

安装的过程中可能会出现依赖 Windows 组件的提示，根据提示下载安装即可，都是自动执行的，如下：

<div align="center"><img src="img/boot/image-20220228164851551.png"></div>

Erlang 安装后需要配置环境变量，否则 RabbitMQ 将无法找到安装的 Erlang。需要配置项如下，作用等同 JDK 配置环境变量的作用。

- ERLANG_HOME
- PATH

##### 安装

windows 版安装包下载地址：[https://](https://rabbitmq.com/install-windows.html)[rabbitmq.com/install-windows.html](https://rabbitmq.com/install-windows.html)

下载完毕后得到 exe 安装文件，一键傻瓜式安装，安装完毕后会得到如下文件

<div align="center"><img src="img/boot/image-20220228165151524.png" alt="image-20220228165151524" style="zoom:67%;" /></div>

<b>启动服务器</b>

```CMD
rabbitmq-service.bat start		# 启动服务
rabbitmq-service.bat stop		# 停止服务
rabbitmqctl status				# 查看服务状态
```

运行 sbin 目录下的 rabbitmq-service.bat 命令即可，start 参数表示启动，stop 参数表示退出，默认对外服务端口 5672。

注意：启动 rabbitmq 的过程实际上是开启 rabbitmq 对应的系统服务，需要管理员权限方可执行。

说明：有没有感觉 5672 的服务端口很熟悉？activemq 与 rabbitmq 有一个端口冲突问题，学习阶段无论操作哪一个？请确保另一个处于关闭状态。

说明：不喜欢命令行的小伙伴可以使用任务管理器中的服务页，找到 RabbitMQ 服务，使用鼠标右键菜单控制服务的启停。

<div align="center"><img src="img/boot/image-20220228170147193.png" alt="image-20220228170147193" style="zoom:67%;" /></div>

<b>访问web管理服务</b>

RabbitMQ 也提供有 web 控制台服务，但是此功能是一个插件，需要先启用才可以使用。

```CMD
rabbitmq-plugins.bat list							# 查看当前所有插件的运行状态
rabbitmq-plugins.bat enable rabbitmq_management		# 启动rabbitmq_management插件
```

启动插件后可以在插件运行状态中查看是否运行，运行后通过浏览器即可打开服务后台管理界面

```CMD
http://localhost:15672
```

web 管理服务默认端口 15672，访问后可以打开 RabbitMQ 的管理界面，如下：

<div align="center"><img src="img/boot/image-20220228170504793.png"></div>

首先输入访问用户名和密码，初始化用户名和密码相同，均为：guest，成功登录后进入管理后台界面，如下：

<div align="center"><img src="img/boot/image-20220228170535261.png"></div>

##### 整合(direct模型)

RabbitMQ 满足 AMQP 协议，因此不同的消息模型对应的制作不同，先使用最简单的 direct 模型开发。

<b>步骤①</b>：导入 Spring Boot 整合 amqp 的 starter，amqp 协议默认实现为 rabbitmq 方案

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-amqp</artifactId>
</dependency>
```

<b>步骤②</b>：配置 RabbitMQ 的服务器地址

```yaml
Spring:
  rabbitmq:
    host: localhost
    port: 5672
```

<b>步骤③</b>：初始化直连模式系统设置

由于 RabbitMQ 不同模型要使用不同的交换机，因此需要先初始化 RabbitMQ 相关的对象，例如队列，交换机等

```JAVA
@Configuration
public class RabbitConfigDirect {
    @Bean
    public Queue directQueue(){
        return new Queue("direct_queue");
    }
    @Bean
    public Queue directQueue2(){
        return new Queue("direct_queue2");
    }
    @Bean
    public DirectExchange directExchange(){
        return new DirectExchange("directExchange");
    }
    @Bean
    public Binding bindingDirect(){
        return BindingBuilder.bind(directQueue()).to(directExchange()).with("direct");
    }
    @Bean
    public Binding bindingDirect2(){
        return BindingBuilder.bind(directQueue2()).to(directExchange()).with("direct2");
    }
}
```

队列 Queue 与直连交换机 DirectExchange 创建后，还需要绑定他们之间的关系 Binding，这样就可以通过交换机操作对应队列。

<b>步骤④</b>：使用 AmqpTemplate 操作 RabbitMQ

```java
@Service
public class MessageServiceRabbitmqDirectImpl implements MessageService {
    @Autowired
    private AmqpTemplate amqpTemplate;

    @Override
    public void sendMessage(String id) {
        System.out.println("待发送短信的订单已纳入处理队列（rabbitmq direct），id："+id);
        amqpTemplate.convertAndSend("directExchange","direct",id);
    }
}
```

amqp 协议中的操作 API 接口名称看上去和 jms 规范的操作 API 接口很相似，但是传递参数差异很大。

<b>步骤⑤</b>：使用消息监听器在服务器启动后，监听指定位置，当消息出现后，立即消费消息

```JAVA
@Component
public class MessageListener {
    @RabbitListener(queues = "direct_queue")
    public void receive(String id){
        System.out.println("已完成短信发送业务(rabbitmq direct)，id："+id);
    }
}
```

使用注解 @RabbitListener 定义当前方法监听 RabbitMQ 中指定名称的消息队列。

##### 整合(topic模型)

<b>步骤①</b>：同上

<b>步骤②</b>：同上

<b>步骤③</b>：初始化主题模式系统设置

```JAVA
@Configuration
public class RabbitConfigTopic {
    @Bean
    public Queue topicQueue(){
        return new Queue("topic_queue");
    }
    @Bean
    public Queue topicQueue2(){
        return new Queue("topic_queue2");
    }
    @Bean
    public TopicExchange topicExchange(){
        return new TopicExchange("topicExchange");
    }
    @Bean
    public Binding bindingTopic(){
        return BindingBuilder.bind(topicQueue()).to(topicExchange()).with("topic.*.id");
    }
    @Bean
    public Binding bindingTopic2(){
        return BindingBuilder.bind(topicQueue2()).to(topicExchange()).with("topic.orders.*");
    }
}
```

主题模式支持 routingKey 匹配模式，* 表示匹配一个单词，# 表示匹配任意内容，这样就可以通过主题交换机将消息分发到不同的队列中，详细内容请参看 RabbitMQ 系列课程。	

| 匹配键        | topic.\*.* | topic.# |
| ----------------- | --------------- | ----------- |
| topic.order.id    | true            | true        |
| order.topic.id    | false           | false       |
| topic.sm.order.id | false           | true        |
| topic.sm.id       | false           | true        |
| topic.id.order    | true            | true        |
| topic.id          | false           | true        |
| topic.order       | false           | true        |

<b>步骤④</b>：使用 AmqpTemplate 操作 RabbitMQ

```java
@Service
public class MessageServiceRabbitmqTopicImpl implements MessageService {
    @Autowired
    private AmqpTemplate amqpTemplate;

    @Override
    public void sendMessage(String id) {
        System.out.println("待发送短信的订单已纳入处理队列（rabbitmq topic），id："+id);
        amqpTemplate.convertAndSend("topicExchange","topic.orders.id",id);
    }
}
```

发送消息后，根据当前提供的 routingKey 与绑定交换机时设定的 routingKey 进行匹配，规则匹配成功消息才会进入到对应的队列中。

<b>步骤⑤</b>：使用消息监听器在服务器启动后，监听指定队列

```JAVA
@Component
public class MessageListener {
    @RabbitListener(queues = "topic_queue")
    public void receive(String id){
        System.out.println("已完成短信发送业务(rabbitmq topic 1)，id："+id);
    }
    @RabbitListener(queues = "topic_queue2")
    public void receive2(String id){
        System.out.println("已完成短信发送业务(rabbitmq topic 22222222)，id："+id);
    }
}
```

使用注解 @RabbitListener 定义当前方法监听 RabbitMQ 中指定名称的消息队列。

<b>总结</b>

1. Spring Boot 整合 RabbitMQ 提供了 AmqpTemplate 对象作为客户端操作消息队列
2. 操作 ActiveMQ 需要配置ActiveMQ服务器地址，默认端口 5672
3. 企业开发时通常使用监听器来处理消息队列中的消息，设置监听器使用注解 @RabbitListener
4. RabbitMQ 有 5 种消息模型，使用的队列相同，但是交换机不同。交换机不同，对应的消息进入的策略也不同

#### 整合RocketMQ

RocketMQ 由阿里研发，后捐赠给 apache 基金会，目前是 apache 基金会顶级项目之一，也是目前市面上的 MQ 产品中较为流行的产品之一，它遵从 AMQP 协议。

##### 安装

windows 版安装包下载地址：[https://rocketmq.apache.org](https://rocketmq.apache.org/)[/](https://rocketmq.apache.org/)

下载完毕后得到 zip 压缩文件，解压缩即可使用，解压后得到如下文件

<div align="center"><img src="img/boot/image-20220228174453471.png"></div>

RocketMQ 安装后需要配置环境变量，具体如下：

- ROCKETMQ_HOME
- PATH
- NAMESRV_ADDR （建议）： 127.0.0.1:9876

关于 NAMESRV_ADDR 对于初学者来说建议配置此项，也可以通过命令设置对应值，操作略显繁琐，建议配置。系统学习 RocketMQ 知识后即可灵活控制该项。

<b>RocketMQ工作模式</b>

在 RocketMQ 中，处理业务的服务器称为 broker，生产者与消费者不是直接与 broker 联系的，而是通过命名服务器进行通信。broker 启动后会通知命名服务器自己已经上线，这样命名服务器中就保存有所有的 broker 信息。当生产者与消费者需要连接 broker 时，通过命名服务器找到对应的处理业务的 broker，因此命名服务器在整套结构中起到一个信息中心的作用。并且 broker 启动前必须保障命名服务器先启动。

<div align="center"><img src="img/boot/image-20220228175123790.png" alt="image-20220228175123790" style="zoom:80%;" /></div>

<b>启动服务器</b>

```CMD
mqnamesrv		# 启动命名服务器
mqbroker		# 启动broker
```

运行 bin 目录下的 mqnamesrv 命令即可启动命名服务器，默认对外服务端口 9876。

运行 bin 目录下的 mqbroker 命令即可启动 broker 服务器，如果环境变量中没有设置 NAMESRV_ADDR 则需要在运行 mqbroker 指令前通过set指令设置 NAMESRV_ADDR 的值，并且每次开启均需要设置此项。

<b>测试服务器启动状态</b>

RocketMQ 提供有一套测试服务器功能的测试程序，运行 bin 目录下的 tools 命令即可使用。

```CMD
tools org.apache.rocketmq.example.quickstart.Producer		# 生产消息
tools org.apache.rocketmq.example.quickstart.Consumer		# 消费消息
```

##### 整合（异步消息）

<b>步骤①</b>：导入 Spring Boot 整合 RocketMQ 的 starter，此坐标不由 Spring Boot 维护版本

```xml
<dependency>
    <groupId>org.apache.rocketmq</groupId>
    <artifactId>rocketmq-Spring-boot-starter</artifactId>
    <version>2.2.1</version>
</dependency>
```

<b>步骤②</b>：配置 RocketMQ 的服务器地址

```yaml
rocketmq:
  name-server: localhost:9876
  producer:
    group: group_rocketmq
```

设置默认的生产者消费者所属组 group。

<b>步骤③</b>：使用 RocketMQTemplate 操作 RocketMQ

```java
@Service
public class MessageServiceRocketmqImpl implements MessageService {
    @Autowired
    private RocketMQTemplate rocketMQTemplate;

    @Override
    public void sendMessage(String id) {
        System.out.println("待发送短信的订单已纳入处理队列（rocketmq），id："+id);
        SendCallback callback = new SendCallback() {
            @Override
            public void onSuccess(SendResult sendResult) {
                System.out.println("消息发送成功");
            }
            @Override
            public void onException(Throwable e) {
                System.out.println("消息发送失败！！！！！");
            }
        };
        rocketMQTemplate.asyncSend("order_id",id,callback);
    }
}

```

使用 asyncSend 方法发送异步消息。

<b>步骤④</b>：使用消息监听器在服务器启动后，监听指定位置，当消息出现后，立即消费消息

```JAVA
@Component
@RocketMQMessageListener(topic = "order_id",consumerGroup = "group_rocketmq")
public class MessageListener implements RocketMQListener<String> {
    @Override
    public void onMessage(String id) {
        System.out.println("已完成短信发送业务(rocketmq)，id："+id);
    }
}
```

RocketMQ 的监听器必须按照标准格式开发，实现 RocketMQListener 接口，泛型为消息类型。

使用注解 @RocketMQMessageListener 定义当前类监听 RabbitMQ 中指定组、指定名称的消息队列。

<b>总结</b>

1. Spring Boot 整合 RocketMQ 使用 RocketMQTemplate 对象作为客户端操作消息队列
2. 操作 RocketMQ 需要配置 RocketMQ 服务器地址，默认端口 9876
3. 企业开发时通常使用监听器来处理消息队列中的消息，设置监听器使用注解 @RocketMQMessageListener

#### 整合Kafka

##### 安装

windows 版安装包下载地址：[https://](https://kafka.apache.org/downloads)[kafka.apache.org/downloads](https://kafka.apache.org/downloads)

下载完毕后得到 tgz 压缩文件，使用解压缩软件解压缩即可使用，解压后得到如下文件

<div align="center"><img src="img/boot/image-20220228181442155.png"></div>

建议使用 windows 版 2.8.1 版本。

<b>启动服务器</b>

kafka 服务器的功能相当于 RocketMQ 中的 broker，kafka 运行还需要一个类似于命名服务器的服务。在 kafka 安装目录中自带一个类似于命名服务器的工具，叫做 zookeeper，它的作用是注册中心，相关知识请到对应课程中学习。

```CMD
zookeeper-server-start.bat ..\..\config\zookeeper.properties		# 启动zookeeper
kafka-server-start.bat ..\..\config\server.properties				# 启动kafka
```

运行 bin 目录下的 windows 目录下的 zookeeper-server-start 命令即可启动注册中心，默认对外服务端口 2181。

运行 bin 目录下的 windows 目录下的 kafka-server-start 命令即可启动 kafka 服务器，默认对外服务端口 9092。

<b>创建主题</b>

和之前操作其他 MQ 产品相似，kakfa 也是基于主题操作，操作之前需要先初始化 topic。

```CMD
# 创建topic
kafka-topics.bat --create --zookeeper localhost:2181 --replication-factor 1 --partitions 1 --topic itheima
# 查询topic
kafka-topics.bat --zookeeper 127.0.0.1:2181 --list					
# 删除topic
kafka-topics.bat --delete --zookeeper localhost:2181 --topic itheima
```

<b>测试服务器启动状态</b>

Kafka 提供有一套测试服务器功能的测试程序，运行 bin 目录下的 windows 目录下的命令即可使用。

```CMD
kafka-console-producer.bat --broker-list localhost:9092 --topic itheima							# 测试生产消息
kafka-console-consumer.bat --bootstrap-server localhost:9092 --topic itheima --from-beginning	# 测试消息消费
```

##### 整合

<b>步骤①</b>：导入 Spring Boot 整合 Kafka 的 starter，此坐标由 Spring Boot 维护版本

```xml
<dependency>
    <groupId>org.Springframework.kafka</groupId>
    <artifactId>Spring-kafka</artifactId>
</dependency>
```

<b>步骤②</b>：配置 Kafka 的服务器地址

```yaml
Spring:
  kafka:
    bootstrap-servers: localhost:9092
    consumer:
      group-id: order
```

设置默认的生产者消费者所属组 id。

<b>步骤③</b>：使用 KafkaTemplate 操作 Kafka

```java
@Service
public class MessageServiceKafkaImpl implements MessageService {
    @Autowired
    private KafkaTemplate<String,String> kafkaTemplate;

    @Override
    public void sendMessage(String id) {
        System.out.println("待发送短信的订单已纳入处理队列（kafka），id："+id);
        kafkaTemplate.send("itheima2022",id);
    }
}
```

使用 send 方法发送消息，需要传入 topic 名称。

<b>步骤④</b>：使用消息监听器在服务器启动后，监听指定位置，当消息出现后，立即消费消息

```JAVA
@Component
public class MessageListener {
    @KafkaListener(topics = "itheima2022")
    public void onMessage(ConsumerRecord<String,String> record){
        System.out.println("已完成短信发送业务(kafka)，id："+record.value());
    }
}
```

使用注解 @KafkaListener 定义当前方法监听 Kafka 中指定 topic 的消息，接收到的消息封装在对象 ConsumerRecord 中，获取数据从 ConsumerRecord 对象中获取即可。

<b>总结</b>

1. Spring Boot 整合 Kafka 使用 KafkaTemplate 对象作为客户端操作消息队列

2. 操作 Kafka 需要配置 Kafka 服务器地址，默认端口 9092

3. 企业开发时通常使用监听器来处理消息队列中的消息，设置监听器使用注解 @KafkaListener。接收消息保存在形参 ConsumerRecord 对象中


## 监控

在说监控之前，需要回顾一下软件业的发展史。最早的软件完成一些非常简单的功能，代码不多，错误也少。随着软件功能的逐步完善，软件的功能变得越来越复杂，功能不能得到有效的保障，这个阶段出现了针对软件功能的检测，也就是软件测试。伴随着计算机操作系统的逐步升级，软件的运行状态也变得开始让人捉摸不透，出现了不稳定的状况。伴随着计算机网络的发展，程序也从单机状态切换成基于计算机网络的程序，应用于网络的程序开始出现，由于网络的不稳定性，程序的运行状态让使用者更加堪忧。互联网的出现彻底打破了软件的思维模式，随之而来的互联网软件就更加凸显出应对各种各样复杂的网络情况之下的弱小。计算机软件的运行状况已经成为了软件运行的一个大话题，针对软件的运行状况就出现了全新的思维，建立起了初代的软件运行状态监控。

什么是监控？就是通过软件的方式展示另一个软件的运行情况，运行的情况则通过各种各样的指标数据反馈给监控人员。例如网络是否顺畅、服务器是否在运行、程序的功能是否能够整百分百运行成功，内存是否够用，等等等等。

本章要讲解的监控就是对软件的运行情况进行监督，但是 Spring Boot 程序与非 Spring Boot 程序的差异还是很大的，为了方便监控软件的开发，Spring Boot 提供了一套功能接口，为开发者加速开发过程。

### 监控的意义

对于现代的互联网程序来说，规模越来越大，功能越来越复杂，还要追求更好的客户体验，因此要监控的信息量也就比较大了。由于现在的互联网程序大部分都是基于微服务的程序，一个程序的运行需要若干个服务来保障，因此第一个要监控的指标就是服务是否正常运行，也就是<b>监控服务状态是否处理宕机状态</b>。一旦发现某个服务宕机了，必须马上给出对应的解决方案，避免整体应用功能受影响。其次，由于互联网程序服务的客户量是巨大的，当客户的请求在短时间内集中达到服务器后，就会出现各种程序运行指标的波动。比如内存占用严重，请求无法及时响应处理等，这就是第二个要监控的重要指标，<b>监控服务运行指标</b>。虽然软件是对外提供用户的访问需求，完成对应功能的，但是后台的运行是否平稳，是否出现了不影响客户使用的功能隐患，这些也是要密切监控的，此时就需要在不停机的情况下，监控系统运行情况，日志是一个不错的手段。如果在众多日志中找到开发者或运维人员所关注的日志信息，简单快速有效的过滤出要看的日志也是监控系统需要考虑的问题，这就是第三个要监控的指标，<b>监控程序运行日志</b>。虽然我们期望程序一直平稳运行，但是由于突发情况的出现，例如服务器被攻击、服务器内存溢出等情况造成了服务器宕机，此时当前服务不能满足使用需要，就要将其重启甚至关闭，如果快速控制服务器的启停也是程序运行过程中不可回避的问题，这就是第四个监控项，<b>管理服务状态</b>。以上这些仅仅是从大的方面来思考监控这个问题，还有很多的细节点，例如上线了一个新功能，定时提醒用户续费，这种功能不是上线后马上就运行的，但是当前功能是否真的启动，如果快速的查询到这个功能已经开启，这也是监控中要解决的问题，等等。看来监控真的是一项非常重要的工作。

通过上述描述，可以看出监控很重要。那具体的监控要如何开展呢？还要从实际的程序运行角度出发。比如现在有 3 个服务支撑着一个程序的运行，每个服务都有自己的运行状态。

<div align="center"><img src="img/boot/image-20220301093704396.png" alt="image-20220301093704396" style="zoom:50%;" /></div>

此时被监控的信息就要在三个不同的程序中去查询并展示，但是三个服务是服务于一个程序的运行的，如果不能合并到一个平台上展示，监控工作量巨大，而且信息对称性差，要不停的在三个监控端查看数据。如果将业务放大成 30 个，300 个，3000 个呢？看来必须有一个单独的平台，将多个被监控的服务对应的监控指标信息汇总在一起，这样更利于监控工作的开展。

<div align="center"><img src="img/boot/image-20220301094001896.png" alt="image-20220301094001896" style="zoom:50%;" /></div>

新的程序专门用来监控，新的问题就出现了，是被监控程序主动上报信息还是监控程序主动获取信息？如果监控程序不能主动获取信息，这就意味着监控程序有可能看到的是很久之前被监控程序上报的信息，万一被监控程序宕机了，监控程序就无法区分究竟是好久没法信息了，还是已经下线了。所以监控程序必须具有主动发起请求获取被监控服务信息的能力。

<div align="center"><img src="img/boot/image-20220301094259844.png" alt="image-20220301094259844" style="zoom:50%;" /></div>

如果监控程序要监控服务时，主动获取对方的信息。那监控程序如何知道哪些程序被自己监控呢？不可能在监控程序中设置我监控谁，这样互联网上的所有程序岂不是都可以被监控到，这样的话信息安全将无法得到保障。合理的做法只能是在被监控程序启动时上报监控程序，告诉监控程序你可以监控我了。看来需要在被监控程序端做主动上报的操作，这就要求被监控程序中配置对应的监控程序是谁。

<div align="center"><img src="img/boot/image-20220301094547748.png" alt="image-20220301094547748" style="zoom:50%;" /></div>

被监控程序可以提供各种各样的指标数据给监控程序看，但是每一个指标都代表着公司的机密信息，并不是所有的指标都可以给任何人看的，乃至运维人员，所以对被监控指标的是否开放出来给监控系统看，也需要做详细的设定。

以上描述的整个过程就是一个监控系统的基本流程。

<b>总结</b>

1. 监控是一个非常重要的工作，是保障程序正常运行的基础手段
2. 监控的过程通过一个监控程序进行，它汇总所有被监控的程序的信息集中统一展示
3. 被监控程序需要主动上报自己被监控，同时要设置哪些指标被监控

<b>思考</b>

下面就要开始做监控了，新的问题就来了，监控程序怎么做呢？难道要自己写吗？肯定是不现实的，如何进行监控？

### 可视化监控平台

Spring Boot 抽取了大部分监控系统的常用指标，提出了监控的总思想。然后就有好心的同志根据监控的总思想，制作了一个通用性很强的监控系统，因为是基于 Spring Boot 监控的核心思想制作的，所以这个程序被命名为 <b>Spring Boot Admin</b>。

Spring Boot Admin，这是一个开源社区项目，用于管理和监控 Spring Boot 应用程序。这个项目中包含有客户端和服务端两部分，而监控平台指的就是服务端。我们做的程序如果需要被监控，将我们做的程序制作成客户端，然后配置服务端地址后，服务端就可以通过 HTTP 请求的方式从客户端获取对应的信息，并通过UI界面展示对应信息。

下面就来开发这套监控程序，先制作服务端，其实服务端可以理解为是一个web程序，收到一些信息后展示这些信息。

<b>服务端开发</b>

<b>步骤①</b>：导入 Spring Boot admin 对应的 starter，版本与当前使用的 Spring Boot 版本保持一致，并将其配置成 web 工程

```xml
<dependency>
    <groupId>de.codecentric</groupId>
    <artifactId>Spring-boot-admin-starter-server</artifactId>
    <version>2.5.4</version>
</dependency>

<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-web</artifactId>
</dependency>
```

上述过程可以通过创建项目时使用勾选的形式完成。

<div align="center"><img src="img/boot/image-20220301102432817.png" alt="image-20220301102432817" style="zoom:50%;" /></div>

<b>步骤②</b>：在引导类上添加注解 @EnableAdminServer，声明当前应用启动后作为 SpringBootAdmin 的服务器使用

```java
@SpringBootApplication
@EnableAdminServer
public class Spring Boot25AdminServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(Spring Boot25AdminServerApplication.class, args);
    }
}
```

做到这里，这个服务器就开发好了，启动后就可以访问当前程序了，界面如下。

<div align="center"><img src="img/boot/image-20220301103028468.png" alt="image-20220301103028468" style="zoom: 50%;" /></div>

由于目前没有启动任何被监控的程序，所以里面什么信息都没有。下面制作一个被监控的客户端程序。

<b>客户端开发</b>

客户端程序开发其实和服务端开发思路基本相似，多了一些配置而已。

<b>步骤①</b>：导入 Spring Boot admin 对应的 starter，版本与当前使用的 Spring Boot 版本保持一致，并将其配置成 web 工程

```xml
<dependency>
    <groupId>de.codecentric</groupId>
    <artifactId>Spring-boot-admin-starter-client</artifactId>
    <version>2.5.4</version>
</dependency>

<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-starter-web</artifactId>
</dependency>
```

上述过程也可以通过创建项目时使用勾选的形式完成，不过一定要小心，端口配置成不一样的，否则会冲突。

<b>步骤②</b>：设置当前客户端将信息上传到哪个服务器上，通过 yml 文件配置

```yaml
Spring:
  boot:
    admin:
      client:
        url: http://localhost:8080
```

做到这里，这个客户端就可以启动了。启动后再次访问服务端程序，界面如下。

<div align="center"><img src="img/boot/image-20220301103838079.png" alt="image-20220301103838079" style="zoom: 50%;" /></div>

可以看到，当前监控了 1 个程序，点击进去查看详细信息。

<div align="center"><img src="img/boot/image-20220301103936386.png" alt="image-20220301103936386" style="zoom: 50%;" /></div>

由于当前没有设置开放哪些信息给监控服务器，所以目前看不到什么有效的信息。下面需要做两组配置就可以看到信息了。

1. 开放指定信息给服务器看
2. 允许服务器以 HTTP 请求的方式获取对应的信息

配置如下：

```yaml
server:
  port: 80
Spring:
  boot:
    admin:
      client:
        url: http://localhost:8080
management:
  endpoint:
    health:
      show-details: always
  endpoints:
    web:
      exposure:
        include: "*"
```

上述配置对于初学者来说比较容易混淆。简单解释一下，到下一节再做具体的讲解。Springbootadmin 的客户端默认开放了 13 组信息给服务器，但是这些信息除了一个之外，其他的信息都不让通过 HTTP 请求查看。所以你看到的信息基本上就没什么内容了，只能看到一个内容，就是下面的健康信息。

<div align="center"><img src="img/boot/image-20220301104742563.png" alt="image-20220301104742563" style="zoom: 50%;" /></div>

但是即便如此我们看到健康信息中也没什么内容，原因在于健康信息中有一些信息描述了你当前应用使用了什么技术等信息，如果无脑的对外暴露功能会有安全隐患。通过配置就可以开放所有的健康信息明细查看了。

```yaml
management:
  endpoint:
    health:
      show-details: always
```

健康明细信息如下：

<div align="center"><img src="img/boot/image-20220301105116554.png" alt="image-20220301105116554" style="zoom: 50%;" /></div>

目前除了健康信息，其他信息都查阅不了。原因在于其他 12 种信息是默认不提供给服务器通过 HTTP 请求查阅的，所以需要开启查阅的内容项，使用 * 表示查阅全部。记得带引号。

```yaml
endpoints:
  web:
    exposure:
      include: "*"
```

配置后再刷新服务器页面，就可以看到所有的信息了。

<div align="center"><img src="img/boot/image-20220301105554494.png" alt="image-20220301105554494" style="zoom: 50%;" /></div>

以上界面中展示的信息量就非常大了，包含了 13 组信息，有性能指标监控，加载的 bean 列表，加载的系统属性，日志的显示控制等等。

<b>配置多个客户端</b>

可以通过配置客户端的方式在其他的 Spring Boot 程序中添加客户端坐标，这样当前服务器就可以监控多个客户端程序了。每个客户端展示不同的监控信息。

<div align="center"><img src="img/boot/image-20220301110352170.png" alt="image-20220301110352170" style="zoom: 50%;" /></div>

进入监控面板，如果你加载的应用具有功能，在监控面板中可以看到3组信息展示的与之前加载的空工程不一样。

- 类加载面板中可以查阅到开发者自定义的类，如左图

<div align="center"><img src="img/boot/image-20220301161246835.png" alt="image-20220301161246835" style="zoom:33%;" /><img src="img/boot/image-20220301161949431.png" alt="image-20220301161949431" style="zoom:33%;" /></div>

- 映射中可以查阅到当前应用配置的所有请求

<div align="center"><img src="img/boot/image-20220301161418791.png" alt="image-20220301161418791" style="zoom: 33%;" /><img src="img/boot/image-20220301162008737.png" alt="image-20220301162008737" style="zoom:33%;" /></div>

- 性能指标中可以查阅当前应用独有的请求路径统计数据

<div align="center"><img src="img/boot/image-20220301161906949.png"  style="zoom: 33%;" /><img src="img/boot/image-20220301162040670.png" style="zoom: 33%;" /></div>

<b>总结</b>

1. 开发监控服务端需要导入坐标，然后在引导类上添加注解@EnableAdminServer，并将其配置成web程序即可
2. 开发被监控的客户端需要导入坐标，然后配置服务端服务器地址，并做开放指标的设定即可
3. 在监控平台中可以查阅到各种各样被监控的指标，前提是客户端开放了被监控的指标

<b>思考</b>

之前说过，服务端要想监控客户端，需要主动的获取到对应信息并展示出来。但是目前我们并没有在客户端开发任何新的功能，但是服务端确可以获取监控信息，谁帮我们做的这些功能呢？

### 监控原理

通过查阅监控中的映射指标，可以看到当前系统中可以运行的所有请求路径，其中大部分路径以 /actuator 开头

<div align="center"><img src="img/boot/image-20220301170214076.png" alt="image-20220301170214076" style="zoom: 50%;" /></div>

首先这些请求路径不是开发者自己编写的，其次这个路径代表什么含义呢？既然这个路径可以访问，就可以通过浏览器发送该请求看看究竟可以得到什么信息。

<div align="center"><img src="img/boot/image-20220301170723057.png"></div>

通过发送请求，可以得到一组json信息，如下

```json
{
    "_links": {
        "self": {
            "href": "http://localhost:81/actuator",
            "templated": false
        },
        "beans": {
            "href": "http://localhost:81/actuator/beans",
            "templated": false
        },
        "caches-cache": {
            "href": "http://localhost:81/actuator/caches/{cache}",
            "templated": true
        },
        "caches": {
            "href": "http://localhost:81/actuator/caches",
            "templated": false
        },
        "health": {
            "href": "http://localhost:81/actuator/health",
            "templated": false
        },
        "health-path": {
            "href": "http://localhost:81/actuator/health/{*path}",
            "templated": true
        },
        "info": {
            "href": "http://localhost:81/actuator/info",
            "templated": false
        },
        "conditions": {
            "href": "http://localhost:81/actuator/conditions",
            "templated": false
        },
        "shutdown": {
            "href": "http://localhost:81/actuator/shutdown",
            "templated": false
        },
        "configprops": {
            "href": "http://localhost:81/actuator/configprops",
            "templated": false
        },
        "configprops-prefix": {
            "href": "http://localhost:81/actuator/configprops/{prefix}",
            "templated": true
        },
        "env": {
            "href": "http://localhost:81/actuator/env",
            "templated": false
        },
        "env-toMatch": {
            "href": "http://localhost:81/actuator/env/{toMatch}",
            "templated": true
        },
        "loggers": {
            "href": "http://localhost:81/actuator/loggers",
            "templated": false
        },
        "loggers-name": {
            "href": "http://localhost:81/actuator/loggers/{name}",
            "templated": true
        },
        "heapdump": {
            "href": "http://localhost:81/actuator/heapdump",
            "templated": false
        },
        "threaddump": {
            "href": "http://localhost:81/actuator/threaddump",
            "templated": false
        },
        "metrics-requiredMetricName": {
            "href": "http://localhost:81/actuator/metrics/{requiredMetricName}",
            "templated": true
        },
        "metrics": {
            "href": "http://localhost:81/actuator/metrics",
            "templated": false
        },
        "scheduledtasks": {
            "href": "http://localhost:81/actuator/scheduledtasks",
            "templated": false
        },
        "mappings": {
            "href": "http://localhost:81/actuator/mappings",
            "templated": false
        }
    }
}
```

其中每一组数据都有一个请求路径，而在这里请求路径中有之前看到过的 health，发送此请求又得到了一组信息

```JSON
{
    "status": "UP",
    "components": {
        "diskSpace": {
            "status": "UP",
            "details": {
                "total": 297042808832,
                "free": 72284409856,
                "threshold": 10485760,
                "exists": true
            }
        },
        "ping": {
            "status": "UP"
        }
    }
}
```

当前信息与监控面板中的数据存在着对应关系

<div align="center"><img src="img/boot/image-20220301171025615.png" alt="image-20220301171025615" style="zoom:50%;" /></div>

原来监控中显示的信息实际上是通过发送请求后得到 json 数据，然后展示出来。按照上述操作，可以发送更多的以 /actuator 开头的链接地址，获取更多的数据，这些数据汇总到一起组成了监控平台显示的所有数据。

到这里我们得到了一个核心信息，监控平台中显示的信息实际上是通过对被监控的应用发送请求得到的。那这些请求谁开发的呢？打开被监控应用的 pom 文件，其中导入了 Spring Boot admin 的对应的 client，在这个资源中导入了一个名称叫做 actuator 的包。被监控的应用之所以可以对外提供上述请求路径，就是因为添加了这个包。

<div align="center"><img src="img/boot/image-20220301171437817.png"></div>

这个 actuator 是什么呢？这就是本节要讲的核心内容，监控的端点。

Actuator，可以称为端点，描述了一组监控信息，SpringBootAdmin 提供了多个内置端点，通过访问端点就可以获取对应的监控信息，也可以根据需要自定义端点信息。通过发送请求路劲 <b>/actuator</b> 可以访问应用所有端点信息，如果端点中还有明细信息可以发送请求 <b>/actuator/端点名称</b> 来获取详细信息。以下列出了所有端点信息说明：

| ID               | 描述                                                         | 默认启用 |
| ---------------- | ------------------------------------------------------------ | -------- |
| auditevents      | 暴露当前应用程序的审计事件信息。                             | 是       |
| beans            | 显示应用程序中所有 Spring bean 的完整列表。                  | 是       |
| caches           | 暴露可用的缓存。                                             | 是       |
| conditions       | 显示在配置和自动配置类上评估的条件以及它们匹配或不匹配的原因。 | 是       |
| configprops      | 显示所有 @ConfigurationProperties 的校对清单。               | 是       |
| env              | 暴露 Spring ConfigurableEnvironment 中的属性。               | 是       |
| flyway           | 显示已应用的 Flyway 数据库迁移。                             | 是       |
| health           | 显示应用程序健康信息                                         | 是       |
| httptrace        | 显示 HTTP 追踪信息（默认情况下，最后 100 个 HTTP 请求/响应交换）。 | 是       |
| info             | 显示应用程序信息。                                           | 是       |
| integrationgraph | 显示 Spring Integration 图。                                 | 是       |
| loggers          | 显示和修改应用程序中日志记录器的配置。                       | 是       |
| liquibase        | 显示已应用的 Liquibase 数据库迁移。                          | 是       |
| metrics          | 显示当前应用程序的指标度量信息。                             | 是       |
| mappings         | 显示所有 @RequestMapping 路径的整理清单。                    | 是       |
| scheduledtasks   | 显示应用程序中的调度任务。                                   | 是       |
| sessions         | 允许从 Spring Session 支持的会话存储中检索和删除用户会话。当使用 Spring Session 的响应式 Web 应用程序支持时不可用。 | 是       |
| shutdown         | 正常关闭应用程序。                                           | 否       |
| threaddump       | 执行线程 dump。                                              | 是       |
| heapdump         | 返回一个 hprof 堆 dump 文件。                                | 是       |
| jolokia          | 通过 HTTP 暴露 JMX bean（当  Jolokia 在 classpath 上时，不适用于 WebFlux）。 | 是       |
| logfile          | 返回日志文件的内容（如果已设置 logging.file 或 logging.path 属性）。支持使用 HTTP Range 头来检索部分日志文件的内容。 | 是       |
| prometheus       | 以可以由 Prometheus 服务器抓取的格式暴露指标。               | 是       |

上述端点每一项代表被监控的指标，如果对外开放则监控平台可以查询到对应的端点信息，如果未开放则无法查询对应的端点信息。通过配置可以设置端点是否对外开放功能。使用 enable 属性控制端点是否对外开放。其中 health 端点为默认端点，不能关闭。

```yaml
management:
  endpoint:
    health:						# 端点名称
      show-details: always
    info:						# 端点名称
      enabled: true				# 是否开放
```

为了方便开发者快速配置端点，Spring Boot admin 设置了 13 个较为常用的端点作为默认开放的端点，如果需要控制默认开放的端点的开放状态，可以通过配置设置，如下：

```YAML
management:
  endpoints:
    enabled-by-default: true	# 是否开启默认端点，默认值true
```

上述端点开启后，就可以通过端点对应的路径查看对应的信息了。但是此时还不能通过 HTTP 请求查询此信息，还需要开启通过 HTTP 请求查询的端点名称，使用“*”可以简化配置成开放所有端点的 WEB 端 HTTP 请求权限。

```YAML
management:
  endpoints:
    web:
      exposure:
        include: "*"
```

整体上来说，对于端点的配置有两组信息，一组是 endpoints 开头的，对所有端点进行配置，一组是 endpoint 开头的，对具体端点进行配置。

```YAML
management:
  endpoint:		# 具体端点的配置
    health:
      show-details: always
    info:
      enabled: true
  endpoints:	# 全部端点的配置
    web:
      exposure:
        include: "*"
    enabled-by-default: true
```

<b>总结</b>

1. 被监控客户端通过添加 actuator 的坐标可以对外提供被访问的端点功能

2. 端点功能的开放与关闭可以通过配置进行控制

3. web 端默认无法获取所有端点信息，通过配置开放端点功能


### 自定义监控指标

端点描述了被监控的信息，除了系统默认的指标，还可以自行添加显示的指标，下面就通过3种不同的端点的指标自定义方式来学习端点信息的二次开发。

<b>INFO端点</b>

info 端点描述了当前应用的基本信息，可以通过两种形式快速配置 info 端点的信息

- 配置形式

    在 yml 文件中通过设置 info 节点的信息就可以快速配置端点信息

    ```yaml
    info:
      appName: @project.artifactId@
      version: @project.version@
      company: 传智教育
      author: itheima
    ```

    配置完毕后，对应信息显示在监控平台上

    <div align="center"><img src="img/boot/image-20220301174133248.png" alt="image-20220301174133248" style="zoom:50%;" /></div>

    也可以通过请求端点信息路径获取对应 json 信息

    <div align="center"><img src="img/boot/image-20220301174241310.png" alt="image-20220301174241310" style="zoom:50%;" /></div>

- 编程形式

    通过配置的形式只能添加固定的数据，如果需要动态数据还可以通过配置 bean 的方式为 info 端点添加信息，此信息与配置信息共存

    ```JAVA
    @Component
    public class InfoConfig implements InfoContributor {
        @Override
        public void contribute(Info.Builder builder) {
            builder.withDetail("runTime",System.currentTimeMillis());		//添加单个信息
            Map infoMap = new HashMap();		
            infoMap.put("buildTime","2006");
            builder.withDetails(infoMap);									//添加一组信息
        }
    }
    ```

<b>Health端点</b>

health 端点描述当前应用的运行健康指标，即应用的运行是否成功。通过编程的形式可以扩展指标信息。

```JAVA
@Component
public class HealthConfig extends AbstractHealthIndicator {
    @Override
    protected void doHealthCheck(Health.Builder builder) throws Exception {
        boolean condition = true;
        if(condition) {
            builder.status(Status.UP);					//设置运行状态为启动状态
            builder.withDetail("runTime", System.currentTimeMillis());
            Map infoMap = new HashMap();
            infoMap.put("buildTime", "2006");
            builder.withDetails(infoMap);
        }else{
            builder.status(Status.OUT_OF_SERVICE);		//设置运行状态为不在服务状态
            builder.withDetail("上线了吗？","你做梦");
        }
    }
}
```

当任意一个组件状态不为 UP 时，整体应用对外服务状态为非 UP 状态。

<div align="center"><img src="img/boot/image-20220301174751845.png" alt="image-20220301174751845" style="zoom:50%;" /></div>

<b>Metrics端点</b>

metrics 端点描述了性能指标，除了系统自带的监控性能指标，还可以自定义性能指标。

```JAVA
@Service
public class BookServiceImpl extends ServiceImpl<BookDao, Book> implements IBookService {
    @Autowired
    private BookDao bookDao;

    private Counter counter;

    public BookServiceImpl(MeterRegistry meterRegistry){
        counter = meterRegistry.counter("用户付费操作次数：");
    }

    @Override
    public boolean delete(Integer id) {
        //每次执行删除业务等同于执行了付费业务
        counter.increment();
        return bookDao.deleteById(id) > 0;
    }
}
```

在性能指标中就出现了自定义的性能指标监控项

<div align="center"><img src="img/boot/image-20220301175101812.png" alt="image-20220301175101812" style="zoom:50%;" /></div>

<b>自定义端点</b>

可以根据业务需要自定义端点，方便业务监控

```JAVA
@Component
@Endpoint(id="pay",enableByDefault = true)
public class PayEndpoint {
    @ReadOperation
    public Object getPay(){
        Map payMap = new HashMap();
        payMap.put("level 1","300");
        payMap.put("level 2","291");
        payMap.put("level 3","666");
        return payMap;
    }
}
```

由于此端点数据 Spirng boot admin 无法预知该如何展示，所以通过界面无法看到此数据，通过 HTTP 请求路径可以获取到当前端点的信息，但是需要先开启当前端点对外功能，或者设置当前端点为默认开发的端点。

<div align="center"><img src="img/boot/image-20220301175355482.png" style="zoom:50%;" /></div>

<b>总结</b>

1. 端点的指标可以自定义，但是每种不同的指标根据其功能不同，自定义方式不同
2. info 端点通过配置和编程的方式都可以添加端点指标
3. health 端点通过编程的方式添加端点指标，需要注意要为对应指标添加启动状态的逻辑设定
4. metrics 指标通过在业务中添加监控操作设置指标
5. 可以自定义端点添加更多的指标

# 原理篇

个人观点，不建议过早的去研究技术的原理。原因有二：

1️⃣<b>先应用熟练，培养技术应用的条件反射，然后再学原理</b>。大把的学习者天天还纠结于这里少写一个这，那里少写一个那，程序都跑不下去，要啥原理，要啥自行车。这里要说一句啊，懂不懂啥意思那不叫原理，原理是抽象到顶层设计层面的东西。知道为什么写这句话，知道错误的原因和懂原理是两码事。

2️⃣ <b>原理不是看源码，源码只能称作原理的落地实现方式</b>，当好的落地实现方式出现后，就会有新旧版本的迭代，底层实现方式也会伴随着更新升级。但是原理不变，只是找到了更好的实现最初目标的路径。<b>原理应该使用最通俗易懂的语言，把设计思想讲出来</b>，至于看源码，只是因为目前的技术原创人员只想到了当前这种最笨的设计方案，还没有更好的。比如 Spirng 程序，写起来很费劲，Spring Boot 出来以后就简单轻松了很多，实现方案变了，原理不变。<b>学习去读懂若干行代码，是选了一条成本最高的路线，看源码仅仅是验证原理，源码仅对应程序流程，不对应原理。原理是思想级的，不是代码级的，原理是原本的道理。</b>

Spring Boot 技术本身就是为了加速 Spirng 程序的开发的，可以大胆的说，Spring Boot 技术没有自己的原理层面的设计，仅仅是实现方案进行了改进。将 Spring Boot 定位成工具，你就不会去想方设法的学习其原理了。就像是将木头分割成若干份，我们可以用斧子，用锯子，用刀，用火烧或者一脚踹断它，这些都是方式方法，而究其本质底层原理是植物纤维的组织方式，研究完这个，你再看前述的各种工具，都是基于这个原理在说如何变更破坏这种植物纤维的方式。所以不要一张嘴说了若干种技术，然后告诉自己，这就是 Spring Boot 的原理。没有的事，Spring Boot 作为一款工具，压根就没有原理。我们下面要学习的其实就是 Spring Boot 程序的工作流程。

下面就开始学习原理篇。原理篇中包含如下内容：

- 自动配置工作流程
- 自定义 starter 开发
- Spring Boot 程序启动流程

下面开启第一部分自动配置工作流程的学习

## 自动配置工作流程

自动配置是 Spring Boot 技术非常好用的核心因素，前面学习了这么多种技术的整合，每一个都离不开自动配置。不过在学习自动配置的时候，需要我们对 Spring 容器如何进行 bean 管理的过程非常熟悉才行，所以这里需要先复习一下有关 Spring 技术中 bean 加载相关的知识。方式方法很多，逐一快速复习一下，查漏补缺。不过这里需要声明一点，这里列出的 bean 的加载方式仅仅应用于后面课程的学习，并不是所有的 Spring 加载 bean 的方式。但这些方案之间有千丝万缕的关系，顺着看完，就明白自动配置是怎么回事了。

### bean的加载方式

关于 bean 的加载方式，Spring 提供了各种各样的形式。因为 Spring 管理 bean 整体上来说就是由 Spring 维护对象的生命周期，所以 bean 的加载可以从大的方面划分成 2 种形式。已知类并交给 Spring 管理，和已知类名并交给 Spring 管理。有什么区别？一个给 .class，一个给类名字符串。内部其实都一样，都是通过 Spring 的 BeanDefinition 对象初始化 Spring 的 bean。如果前面这句话看起来有障碍，可以去复习一下 Spring 的相关知识。

<b style="color:orange">参考 B 站满一航老师录制的 Spring 高级课程，链接地址如下：https://www.bilibili.com/video/BV1P44y1N7QG 这个课真的很牛逼！强烈推荐！满老师的其他课也非常棒！</b>

#### 方式一：配置文件+`<bean/>`标签

最高端的食材往往只需要最简单的烹饪方法。最初级的 bean 的加载方式其实可以直击 Spring 管控 bean 的核心思想，就是提供类名，然后 Spring 就可以管理了。所以第一种方式就是给出 bean 的类名，内部在利用反射机制加载成 class，然后，就没有然后了，拿到了 class 你就可以搞定一切了。

```XML
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.Springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.Springframework.org/schema/beans http://www.Springframework.org/schema/beans/Spring-beans.xsd">
    <!--xml方式声明自己开发的bean-->
    <bean id="cat" class="Cat"/>
    <bean class="Dog"/>

    <!--xml方式声明第三方开发的bean-->
    <bean id="dataSource" class="com.alibaba.druid.pool.DruidDataSource"/>
    <bean class="com.alibaba.druid.pool.DruidDataSource"/>
    <bean class="com.alibaba.druid.pool.DruidDataSource"/>
</beans>
```

#### 方式二：配置文件扫描+注解定义bean

由于方式一种需要将 Spring 管控的 bean 全部写在 xml 文件中，对于程序员来说非常不友好，所以就有了第二种方式。哪一个类要受到 Spring 管控加载成 bean，就在这个类的上面加一个注解，还可以顺带起一个 bean 的名字（id）。这里可以使用的注解有 @Component 以及三个衍生注解 @Service、@Controller、@Repository。

```JAVA
@Component("tom")
public class Cat {}
```

```JAVA
@Service
public class Mouse {}
```

当然，由于我们无法在第三方提供的技术源代码中去添加上述 4 个注解，因此当你需要加载第三方开发的 bean 的时候可以使用下列方式定义注解式的 bean。@Bean 定义在一个方法上方，当前方法的返回值就可以交给 Spring 管控，记得这个方法所在的类一定要定义在 @Component 修饰的类中，有人会说不是 @Configuration 吗？建议把 Spring 注解开发相关课程学习一下，就不会有这个疑问了，<b style="color:orange">@Configuration 其实只是 @Component 的模式注解。</b>

```JAVA
@Component
public class DbConfig {
    @Bean
    public DruidDataSource dataSource(){
        DruidDataSource ds = new DruidDataSource();
        return ds;
    }
}
```

上面提供的仅仅是 bean 的声明，Spring 并没有感知到这些东西，像极了上课积极回答问题的你，手举的非常高，可惜老师都没有往你的方向看上一眼。想让 Spring 感知到这些积极的小伙伴，必须设置 Spring 去检查这些类，看他们是否贴标签。可以通过下列 xml 配置设置 Spring 去检查哪些包，发现定了对应注解，就将对应的类纳入 Spring 管控范围，声明成 bean。

```XML
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.Springframework.org/schema/beans"
       xmlns:context="http://www.Springframework.org/schema/context"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="
       http://www.Springframework.org/schema/beans
       http://www.Springframework.org/schema/beans/Spring-beans.xsd
       http://www.Springframework.org/schema/context
       http://www.Springframework.org/schema/context/Spring-context.xsd
    ">
    <!--指定扫描加载bean的位置-->
    <context:component-scan base-package="com.itheima.bean,com.itheima.config"/>
</beans>
```

方式二声明 bean 的方式是目前企业中较为常见的 bean 的声明方式，但是也有缺点。方式一中，通过一个配置文件，你可以查阅当前 Spring 环境中定义了多少个或者说多少种 bean，但是方式二没有任何一个地方可以查阅整体信息，只有当程序运行起来才能感知到加载了多少个 bean。

#### 方式三：注解方式声明配置类

方式二已经完美的简化了 bean 的声明，以后再也不用写茫茫多的配置信息了。仔细观察 xml 配置文件，会发现这个文件中只剩了扫描包这句话，于是就有人提出，使用 java 类替换掉这种固定格式的配置，所以下面这种格式就出现了。严格意义上讲不能算全新的方式，但是由于此种开发形式是企业级开发中的主流形式，所以单独独立出来做成一种方式。但是方式二和方式三其实差别还是挺大的，番外篇找个时间再聊吧。

定义一个类并使用 @ComponentScan 替代原始 xml 配置中的包扫描这个动作，其实功能基本相同。为什么说基本，还是有差别的。先卖个关子吧，番外篇再聊。

```java
@ComponentScan({"com.itheima.bean","com.itheima.config"})
public class SpringConfig3 {
    @Bean
    public DogFactoryBean dog(){
        return new DogFactoryBean();
    }
}
```

##### 使用FactroyBean接口

补充一个小知识，Spring 提供了一个接口 FactoryBean，也可以用于声明 bean，只不过实现了 FactoryBean 接口的类造出来的对象不是当前类的对象，而是 FactoryBean 接口泛型指定类型的对象。如下列，造出来的 bean 并不是 DogFactoryBean，而是 Dog。有什么用呢？可以在对象初始化前做一些事情，下例中的注释位置就是让你自己去扩展要做的其他事情的。

```JAVA
public class DogFactoryBean implements FactoryBean<Dog> {
    @Override
    public Dog getObject() throws Exception {
        Dog d = new Dog();
        //.........
        return d;
    }
    @Override
    public Class<?> getObjectType() {
        return Dog.class;
    }
    @Override
    public boolean isSingleton() {
        return true;
    }
}
```

有人说，注释中的代码写入 Dog 的构造方法不就行了吗？干嘛这么费劲转一圈，还写个类，还要实现接口，多麻烦啊。还真不一样，你可以理解为 Dog 是一个抽象后剥离的特别干净的模型，但是实际使用的时候必须进行一系列的初始化动作。只不过根据情况不同，初始化动作不同而已。如果写入 Dog，或许初始化动作 A 当前并不能满足你的需要，这个时候你就要做一个 DogB 的方案了。然后，就没有然后了，你就要做两个 Dog 类。当时使用 FactoryBean 接口就可以完美解决这个问题。

通常实现了 FactoryBean 接口的类使用 @Bean 的形式进行加载，当然你也可以使用 @Component 去声明 DogFactoryBean，只要被扫描加载到即可，但是这种格式加载总觉得怪怪的，指向性不是很明确。

```java
@ComponentScan({"com.itheima.bean","com.itheima.config"})
public class SpringConfig3 {
    @Bean
    public DogFactoryBean dog(){
        return new DogFactoryBean();
    }
}
```

##### 注解格式导入XML格式配置的bean

再补充一个小知识，由于早起开发的系统大部分都是采用 xml 的形式配置 bean，现在的企业级开发基本上不用这种模式了。但是如果你特别幸运，需要基于之前的系统进行二次开发，这就尴尬了。新开发的用注解格式，之前开发的是 xml 格式。这个时候可不是让你选择用哪种模式的，而是两种要同时使用。Spring 提供了一个注解可以解决这个问题，@ImportResource，在配置类上直接写上要被融合的 xml 配置文件名即可，算的上一种兼容性解决方案，没啥实际意义。

```JAVA
@Configuration
@ImportResource("applicationContext1.xml")
public class SpringConfig32 {}
```

##### proxyBeanMethods属性

前面的例子中用到了 @Configuration 这个注解，当我们使用 AnnotationConfigApplicationContext 加载配置类的时候，配置类可以不添加这个注解。但是这个注解有一个更加强大的功能，它可以保障配置类中使用方法创建的 bean 的唯一性。为 @Configuration 注解设置 proxyBeanMethods 属性值为 true 即可，由于此属性默认值为 true，所以很少看见明确书写的，除非想放弃此功能。

```JAVA
@Configuration(proxyBeanMethods = true)
public class SpringConfig33 {
    @Bean
    public Cat cat(){
        return new Cat();
    }
}
```

下面通过容器再调用上面的 cat 方法时，得到的就是同一个对象了。注意，必须使用 Spring 容器对象调用此方法才有保持 bean 唯一性的特性。此特性在很多底层源码中有应用，前面讲 MQ 时，也应用了此特性，只不过当前没有解释而已。这里算是填个坑吧。

```JAVA
public class App33 {
    public static void main(String[] args) {
        ApplicationContext ctx = new AnnotationConfigApplicationContext(SpringConfig33.class);
        String[] names = ctx.getBeanDefinitionNames();
        for (String name : names) {
            System.out.println(name);
        }
        System.out.println("-------------------------");
        SpringConfig33 SpringConfig33 = ctx.getBean("SpringConfig33", SpringConfig33.class);
        System.out.println(SpringConfig33.cat());
        System.out.println(SpringConfig33.cat());
        System.out.println(SpringConfig33.cat());
    }
}
```

#### 方式四：使用@Import注解注入bean

使用扫描的方式加载 bean 是企业级开发中常见的 bean 的加载方式，但是由于扫描的时候不仅可以加载到你要的东西，还有可能加载到各种各样的乱七八糟的东西，万一没有控制好得不偿失了。

有人就会奇怪，会有什么问题呢？比如你扫描了 com.itheima.service 包，后来因为业务需要，又扫描了 com.itheima.dao 包，你发现 com.itheima 包下面只有 service 和 dao 这两个包，这就简单了，直接扫描 com.itheima 就行了。但是万万没想到，十天后你加入了一个外部依赖包，里面也有 com.itheima 包，这下就热闹了，该来的不该来的全来了。

所以我们需要一种精准制导的加载方式，使用 @Import 注解就可以解决你的问题。它可以加载所有的一切，只需要在注解的参数中写上加载的类对应的 .class 即可。有人就会觉得，还要自己手写，多麻烦，不如扫描好用。对呀，但是他可以指定加载啊，好的命名规范配合 @ComponentScan 可以解决很多问题，但是 @Import 注解拥有其重要的应用场景。有没有想过假如你要加载的 bean 没有使用 @Component 修饰呢？这下就无解了，而 @Import 就无需考虑这个问题。

```java
@Import({Dog.class,DbConfig.class})
public class SpringConfig4 {}
```

##### 使用@Import注解注入配置类

除了加载 bean，还可以使用 @Import 注解加载配置类。其实本质上是一样的，不解释太多了。

```java
@Import(DogFactoryBean.class)
public class SpringConfig4 {}
```

#### 方式五：编程形式注册bean

前面介绍的加载 bean 的方式都是在容器启动阶段完成 bean 的加载，下面这种方式就比较特殊了，可以在容器初始化完成后手动加载 bean。通过这种方式可以实现编程式控制 bean 的加载。

```java
public class App5 {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext ctx = new AnnotationConfigApplicationContext(SpringConfig.class);
        //上下文容器对象已经初始化完毕后，手工加载bean
        ctx.register(Mouse.class);
    }
}
```

其实这种方式坑还是挺多的，比如容器中已经有了某种类型的 bean，再加载会不会覆盖呢？这都是要思考和关注的问题。新手慎用。

```java
public class App5 {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext ctx = new AnnotationConfigApplicationContext(SpringConfig.class);
        //上下文容器对象已经初始化完毕后，手工加载bean
        ctx.registerBean("tom", Cat.class,0);
        ctx.registerBean("tom", Cat.class,1);
        ctx.registerBean("tom", Cat.class,2);
        System.out.println(ctx.getBean(Cat.class));
    }
}
```

#### 方式六：导入实现了ImportSelector接口的类

在方式五种，我们感受了 bean 的加载可以进行编程化的控制，添加 if 语句就可以实现 bean 的加载控制了。但是毕竟是在容器初始化后实现 bean 的加载控制，那是否可以在容器初始化过程中进行控制呢？答案是必须的。实现 ImportSelector 接口的类可以设置加载的 bean 的全路径类名，记得一点，只要能编程就能判定，能判定意味着可以控制程序的运行走向，进而控制一切。

现在又多了一种控制 bean 加载的方式，或者说是选择 bean 的方式。

```java
public class MyImportSelector implements ImportSelector {
    @Override
    public String[] selectImports(AnnotationMetadata metadata) {
        //各种条件的判定，判定完毕后，决定是否装载指定的bean
        boolean flag = metadata.hasAnnotation("org.Springframework.context.annotation.Configuration");
        if(flag){
            return new String[]{"com.itheima.bean.Dog"};
        }
        return new String[]{"com.itheima.bean.Cat"};
    }
}
```

#### 方式七：导入实现了ImportBeanDefinitionRegistrar接口的类

方式六中提供了给定类全路径类名控制 bean 加载的形式，如果对 Spring 的 bean 的加载原理比较熟悉的小伙伴知道，其实 bean 的加载不是一个简简单单的对象，Spring 中定义了一个叫做 BeanDefinition 的东西，它才是控制 bean 初始化加载的核心。BeanDefinition 接口中给出了若干种方法，可以控制 bean 的相关属性。说个最简单的，创建的对象是单例还是非单例，在 BeanDefinition 中定义了 scope 属性就可以控制这个。如果你感觉方式六没有给你开放出足够的对 bean 的控制操作，那么方式七你值得拥有。我们可以通过定义一个类，然后实现 ImportBeanDefinitionRegistrar 接口的方式定义 bean，并且还可以让你对 bean 的初始化进行更加细粒度的控制，不过对于新手并不是很友好。忽然给你开放了若干个操作，还真不知道如何下手。

```java
public class MyRegistrar implements ImportBeanDefinitionRegistrar {
    @Override
    public void registerBeanDefinitions(AnnotationMetadata metadata, BeanDefinitionRegistry registry) {
        BeanDefinition beanDefinition = 	
            BeanDefinitionBuilder.rootBeanDefinition(BookServiceImpl2.class).getBeanDefinition();
        registry.registerBeanDefinition("bookService",beanDefinition);
    }
}
```

#### 方式八：导入实现了BeanDefinitionRegistryPostProcessor接口的类

上述七种方式都是在容器初始化过程中进行 bean 的加载或者声明，但是这里有一个 bug。这么多种方式，它们之间如果有冲突怎么办？谁能有最终裁定权？这是个好问题，当某种类型的 bean 被接二连三的使用各种方式加载后，在你对所有加载方式的加载顺序没有完全理解清晰之前，你还真不知道最后谁说了算。即便你理清楚了，保不齐和你一起开发的猪队友又添加了一个 bean，得嘞，这下就热闹了。

Spring 挥舞它仲裁者的大刀来了一个致命一击，都别哔哔了，我说了算，BeanDefinitionRegistryPostProcessor，看名字知道，BeanDefinition 意思是 bean 定义，Registry 注册的意思，Post 后置，Processor 处理器，全称 bean 定义后处理器，干啥的？在所有 bean 注册都折腾完后，它把最后一道关，说白了，它说了算，这下消停了，它是最后一个运行的。

```java
public class MyPostProcessor implements BeanDefinitionRegistryPostProcessor {
    @Override
    public void postProcessBeanDefinitionRegistry(BeanDefinitionRegistry registry) throws BeansException {
        BeanDefinition beanDefinition = 
            BeanDefinitionBuilder.rootBeanDefinition(BookServiceImpl4.class).getBeanDefinition();
        registry.registerBeanDefinition("bookService",beanDefinition);
    }
}
```

总体上来说，上面介绍了各种各样的 bean 的注册加载初始化方式，脑子里建立个概念吧，方式很多，Spring 源码中大量运用各种方式。复习的内容就先说到这里。

<b>总结</b>

1. bean 的定义由前期 xml 配置逐步演化成注解配置，本质是一样的，都是通过反射机制加载类名后创建对象，对象就是 Spring 管控的 bean
2. @Import 注解可以指定加载某一个类作为 Spring 管控的 bean，如果被加载的类中还具有 @Bean 相关的定义，会被一同加载
3. Spring 开放出了若干种可编程控制的 bean 的初始化方式，通过分支语句由固定的加载 bean 转成了可以选择 bean 是否加载或者选择加载哪一种 bean

### bean的加载控制

前面复习 bean 的加载时，提出了有关加载控制的方式，其中手工注册 bean，ImportSelector 接口，ImportBeanDefinitionRegistrar 接口，BeanDefinitionRegistryPostProcessor 接口都可以控制 bean 的加载，这一节就来说说这些加载控制。

企业级开发中不可能在 Spring 容器中进行 bean 的饱和式加载的。什么是饱和式加载，就是不管用不用，全部加载。比如 jdk 中有两万个类，那就加载两万个 bean，显然是不合理的，因为你压根就不会使用其中大部分的 bean。那合理的加载方式是什么？肯定是必要性加载，就是用什么加载什么。继续思考，加载哪些 bean 通常受什么影响呢？最容易想的就是你要用什么技术，就加载对应的 bean。用什么技术意味着什么？就是加载对应技术的类。所以在 Spring 容器中，通过判定是否加载了某个类来控制某些 bean 的加载是一种常见操作。下例给出了对应的代码实现，其实思想很简单，先判断一个类的全路径名是否能够成功加载，加载成功说明有这个类，那就干某项具体的工作，否则就干别的工作。

```JAVA
public class MyImportSelector implements ImportSelector {
    @Override
    public String[] selectImports(AnnotationMetadata importingClassMetadata) {
        try {
            Class<?> clazz = Class.forName("com.itheima.bean.Mouse");
            if(clazz != null) {
                return new String[]{"com.itheima.bean.Cat"};
            }
        } catch (ClassNotFoundException e) {
//            e.printStackTrace();
            return new String[0];
        }
        return null;
    }
}
```

通过上述的分析，可以看到此类操作将成为企业级开发中的常见操作，于是 Spring Boot 将把这些常用操作给我们做了一次封装。这种逻辑判定你开发者就别搞了，Spring Boot 信不过这种新手开发者，给你封装一下，做几个注解，你填参数就行。

下例使用 @ConditionalOnClass 注解实现了当虚拟机中加载了 com.itheima.bean.Wolf 类时加载对应的 bean。比较一下上面的代码和下面的代码，有没有感觉很清爽。其实此类注解还有很多。

```java
@Bean
@ConditionalOnClass(name = "com.itheima.bean.Wolf")
public Cat tom(){
    return new Cat();
}
```

@ConditionalOnMissingClass 注解控制虚拟机中没有加载指定的类才加载对应的 bean。

```java
@Bean
@ConditionalOnMissingClass("com.itheima.bean.Dog")
public Cat tom(){
    return new Cat();
}
```

这种条件还可以做并且的逻辑关系，写 2 个就是 2 个条件都成立，写多个就是多个条件都成立。

```java
@Bean
@ConditionalOnClass(name = "com.itheima.bean.Wolf")
@ConditionalOnMissingClass("com.itheima.bean.Mouse")
public Cat tom(){
    return new Cat();
}
```

除了判定是否加载类，还可以对当前容器类型做判定，下例是判定当前容器环境是否是 web 环境。

```java
@Bean
@ConditionalOnWebApplication
public Cat tom(){
    return new Cat();
}
```

下面是判定容器环境是否是非 web 环境。

```java
@Bean
@ConditionalOnNotWebApplication
public Cat tom(){
    return new Cat();
}
```

当然还可以判定是否加载了指定名称的 bean，这种有什么用呢？太有用了。比如当前容器中已经提供了 jdbcTemplate 对应的 bean，你还需要再加载一个全新的 jdbcTemplate 的 bean 吗？没有必要了嘛。Spring 说，如果你自己写的话，我就不帮你操这份心了，如果你没写，我再给你提供。自适应，自适应，明白？没有的话就提供给你，有的话就用你自己的。

```java
@Bean
@ConditionalOnBean(name="jerry")
public Cat tom(){
    return new Cat();
}
```

以下就是判定当前是否加载了 mysql 的驱动类，如果加载了，我就给你搞一个 Druid 的数据源对象出来，完美！

```JAVA
public class SpringConfig {
    @Bean
    @ConditionalOnClass(name="com.mysql.jdbc.Driver")
    public DruidDataSource dataSource(){
        return new DruidDataSource();
    }
}
```

其中 Spring Boot 的 bean 加载控制注解还有很多，这里就不一一列举了，最常用的判定条件就是根据类是否加载来进行控制。

<b>总结</b>

1. Spring Boot 定义了若干种控制 bean 加载的条件设置注解，由 Spring 固定加载 bean 变成了可以根据情况选择性的加载 bean

### bean的依赖属性配置管理

bean 的加载及加载控制已经搞完了，下面研究一下 bean 内部的事情。bean 在运行的时候，实现对应的业务逻辑时有可能需要开发者提供一些设置值，有就是属性了。如果使用构造方法将参数固定，灵活性不足，这个时候就可以使用前期学习的 bean 的属性配置相关的知识进行灵活的配置了。先通过 yml 配置文件，设置 bean 运行需要使用的配置信息。

```yaml
cartoon:
  cat:
    name: "图多盖洛"
    age: 5
  mouse:
    name: "泰菲"
    age: 1
```

然后定义一个封装属性的专用类，加载配置属性，读取对应前缀相关的属性值。

```JAVA
@ConfigurationProperties(prefix = "cartoon")
@Data
public class CartoonProperties {
    private Cat cat;
    private Mouse mouse;
}
```

最后在使用的位置注入对应的配置即可。

```JAVA
@EnableConfigurationProperties(CartoonProperties.class)
public class CartoonCatAndMouse{
    @Autowired
    private CartoonProperties cartoonProperties;
}
```

建议在业务类上使用 @EnableConfigurationProperties 声明 bean，这样在不使用这个类的时候，也不会无故加载专用的属性配置类 CartoonProperties，减少 Spring 管控的资源数量。

<b>总结</b>

1. bean 的运行如果需要外部设置值，建议将设置值封装成专用的属性类* * * * Properties
2. 设置属性类加载指定前缀的配置信息
3. 在需要使用属性类的位置通过注解 @EnableConfigurationProperties 加载 bean，而不要直接在属性配置类上定义 bean，减少资源加载的数量，因需加载而不要饱和式加载。

### 自动配置原理（工作流程）

经过前面的知识复习，下面终于进入到了本章核心内容的学习，自动配置原理。原理谈不上，就是自动配置的工作流程。

啥叫自动配置呢？简单说就是 Spring Boot 根据我们开发者的行为猜测你要做什么事情，然后把你要用的 bean 都给你准备好。听上去是不是很神奇？其实非常简单，前面复习的东西都已经讲完了。Spring Boot 咋做到的呢？就是看你导入了什么类，就知道你想干什么了。然后把你有可能要用的 bean（注意是有可能）都给你加载好，你直接使用就行了，Spring Boot 把所需要的一切工作都做完了。

自动配置的意义就是加速开发效率，将开发者使用某种技术时需要使用的 bean 根据情况提前加载好，实现自动配置的效果。当然，开发者有可能需要提供必要的参数，比如你要用  mysql 技术，导入了 mysql 的坐标，Spring Boot 就知道了你要做数据库操作，一系列的数据库操作相关的bean都给你提前声明好，但是你要告诉 Spring Boot 你到底用哪一个数据库，像什么 IP 地址啊，端口啊，你不告诉 Spring Boot，Spring Boot 就无法帮你把自动配置相关的工作做完。

而这种思想其实就是在日常的开发过程中根据开发者的习惯慢慢抽取得到了。整体过程分为 2 个阶段：

<b>阶段一：准备阶段</b>

1. Spring Boot 的开发人员先大量收集 Spring 开发者的编程习惯，整理开发过程每一个程序经常使用的技术列表，形成一个<b>技术集 A</b>
2. 收集常用技术(<b>技术集 A</b>)的使用参数，不管你用什么常用设置，我用什么常用设置，统统收集起来整理一下，得到开发过程中每一个技术的常用设置，形成每一个技术对应的<b>设置集 B</b>

<b>阶段二：加载阶段</b>

1. Spring Boot 初始化 Spring 容器基础环境，读取用户的配置信息，加载用户自定义的 bean 和导入的其他坐标，形成<b>初始化环境</b>
2. Spring Boot 将 <b>技术集 A</b> 包含的所有技术在 Spring Boot 启动时默认全部加载，这时肯定加载的东西有一些是无效的，没有用的
3. Spring Boot 会对 <b>技术集 A</b> 中每一个技术约定出启动这个技术对应的条件，并设置成按条件加载，由于开发者导入了一些 bean 和其他坐标，也就是与<b>初始化环境</b>，这个时候就可以根据这个<b>初始化环境</b>与 Spring Boot 的 <b>技术集 A</b> 进行比对了，哪个匹配上加载哪个
4. 因为有些技术不做配置就无法工作，所以 Spring Boot 开始对 <b>设置集 B</b> 下手了。它统计出各个国家各个行业的开发者使用某个技术时最常用的设置是什么，然后把这些设置作为默认值直接设置好，并告诉开发者当前设置我已经给你搞了一套，你要用可以直接用，这样可以减少开发者配置参数的工作量
5. 但是默认配置不一定能解决问题，于是 Spring Boot 开放修改**设置集 B** 的接口，可以由开发者根据需要决定是否覆盖默认配置

​		以上这些仅仅是一个思想，落地到代码实现阶段就要好好思考一下怎么实现了。假定我们想自己实现自动配置的功能，都要做哪些工作呢？

- 首先指定一个技术 X，我们打算让技术 X 具备自动配置的功能，这个技术 X 可以是任意功能，这个技术隶属于上面描述的 <b>技术集 A</b>

```JAVA
public class CartoonCatAndMouse{}
```

- 然后找出技术 X 使用过程中的常用配置 Y，这个配置隶属于上面表述的 <b>设置集 B</b>

```yaml
cartoon:
  cat:
    name: "图多盖洛"
    age: 5
  mouse:
    name: "泰菲"
    age: 1
```

- 将常用配置 Y 设计出对应的 yml 配置书写格式，然后定义一个属性类封装对应的配置属性，这个过程其实就是上一节咱们做的 bean 的依赖属性管理，一模一样

```JAVA
@ConfigurationProperties(prefix = "cartoon")
@Data
public class CartoonProperties {
    private Cat cat;
    private Mouse mouse;
}
```

- 最后做一个配置类，当这个类加载的时候就可以初始化对应的功能 bean，并且可以加载到对应的配置

```JAVA
@EnableConfigurationProperties(CartoonProperties.class)
public class CartoonCatAndMouse implements ApplicationContextAware {
    private CartoonProperties cartoonProperties;
}
```

- 当然，你也可以为当前自动配置类设置上激活条件，例如使用 @CondtionOn* * * * 为其设置加载条件

```JAVA
@ConditionalOnClass(name="org.Springframework.data.redis.core.RedisOperations")
@EnableConfigurationProperties(CartoonProperties.class)
public class CartoonCatAndMouse implements ApplicationContextAware {
    private CartoonProperties cartoonProperties;
}
```

做到这里都已经做完了，但是遇到了一个全新的问题，如何让 Spring Boot 启动的时候去加载这个类呢？如果不加载的话，我们做的条件判定，做的属性加载这些全部都失效了。 Spring Boot 为我们开放了一个配置入口，在配置目录中创建 META-INF 目录，并创建 Spring.factories 文件，在其中添加设置，说明哪些类要启动自动配置就可以了。

```CMD
# Auto Configure
org.Springframework.boot.autoconfigure.EnableAutoConfiguration=\
com.itheima.bean.CartoonCatAndMouse
```

其实这个文件就做了一件事，通过这种配置的方式加载了指定的类。转了一圈，就是个普通的 bean 的加载，和最初使用 xml 格式加载 bean 几乎没有区别，格式变了而已。那自动配置的核心究竟是什么呢？自动配置其实是一个小的生态，可以按照如下思想理解：

1. 自动配置从根本上来说就是一个 bean 的加载
2. 通过 bean 加载条件的控制给开发者一种感觉，自动配置是自适应的，可以根据情况自己判定，但实际上就是最普通的分支语句的应用，这是蒙蔽我们双眼的第一层面纱
3. 使用 bean 的时候，如果不设置属性，就有默认值，如果不想用默认值，就可以自己设置，也就是可以修改部分或者全部参数，感觉这个过程好屌，也是一种自适应的形式，其实还是需要使用分支语句来做判断的，这是蒙蔽我们双眼的第二层面纱
4. Spring Boot 技术提前将大量开发者有可能使用的技术提前做好了，条件也写好了，用的时候你导入了一个坐标，对应技术就可以使用了，其实就是提前帮我们把 Spring.factories 文件写好了，这是蒙蔽我们双眼的第三层面纱

你在不知道自动配置这个知识的情况下，经过上面这一二三，你当然觉得自动配置是一种特别牛的技术，但是一窥究竟后发现，也就那么回事。而且现在 Spring Boot 程序启动时，在后台偷偷的做了这么多次检测，这么多种情况判定，不用问了，效率一定是非常低的，毕竟它要检测100余种技术是否在你程序中使用。

以上内容是自动配置的工作流程。

<b>总结</b>

1. Spring Boot 启动时先加载 Spring.factories 文件中的 org.Springframework.boot.autoconfigure.EnableAutoConfiguration 配置项，将其中配置的所有的类都加载成 bean
2. 在加载 bean 的时候，bean 对应的类定义上都设置有加载条件，因此有可能加载成功，也可能条件检测失败不加载 bean
3. 对于可以正常加载成 bean 的类，通常会通过 @EnableConfigurationProperties 注解初始化对应的配置属性类并加载对应的配置
4. 配置属性类上通常会通过 @ConfigurationProperties 加载指定前缀的配置，当然这些配置通常都有默认值。如果没有默认值，就强制你必须配置后使用了

### 变更自动配置

知道了自动配置的执行过程，下面就可以根据这个自动配置的流程做一些高级定制了。例如系统默认会加载 100 多种自动配置的技术，如果我们先手工干预此工程，禁用自动配置是否可行呢？答案一定是可以的。方式还挺多：

<b>方式一：通过 yaml 配置设置排除指定的自动配置类</b>

```YAML
Spring:
  autoconfigure:
    exclude:
      - org.Springframework.boot.autoconfigure.task.TaskExecutionAutoConfiguration
```

<b>方式二：通过注解参数排除自动配置类</b>

```JAVA
@EnableAutoConfiguration(excludeName = "",exclude = {})
```

<b>方式三：排除坐标（应用面较窄）</b>

如果当前自动配置中包含有更多的自动配置功能，也就是一个套娃的效果。此时可以通过检测条件的控制来管理自动配置是否启动。例如 web 程序启动时会自动启动 tomcat 服务器，可以通过排除坐标的方式，让加载 tomcat 服务器的条件失效。不过需要提醒一点，你把 tomcat 排除掉，记得再加一种可以运行的服务器。

```XML
<dependencies>
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-web</artifactId>
        <!--web起步依赖环境中，排除Tomcat起步依赖，匹配自动配置条件-->
        <exclusions>
            <exclusion>
                <groupId>org.Springframework.boot</groupId>
                <artifactId>Spring-boot-starter-tomcat</artifactId>
            </exclusion>
        </exclusions>
    </dependency>
    <!--添加Jetty起步依赖，匹配自动配置条件-->
    <dependency>
        <groupId>org.Springframework.boot</groupId>
        <artifactId>Spring-boot-starter-jetty</artifactId>
    </dependency>
</dependencies>
```

<b>总结</b>

1. Spring Boot 的自动配置并不是必然运行的，可以通过配置的形式干预是否启用对应的自动配置功能

## 自定义starter开发

自动配置学习完后，我们就可以基于自动配置的特性，开发 Spring Boot 技术中最引以为傲的功能了，starter。其实通过前期学习，我们发现用什么技术直接导入对应的 starter，然后就实现了 Spring Boot 整合对应技术，再加上一些简单的配置，就可以直接使用了。这种设计方式对开发者非常友好，本章就通过一个案例的制作，开发自定义 starter 来实现自定义功能的快捷添加。

### 案例：记录系统访客独立IP访问次数

本案例的功能是统计网站独立 IP 访问次数的功能，并将访问信息在后台持续输出。整体功能是在后台每 10 秒输出一次监控信息（格式：IP+访问次数） ，当用户访问网站时，对用户的访问行为进行统计。

例如：张三访问网站功能 15 次，IP 地址：192.168.0.135，李四访问网站功能 20 次，IP 地址：61.129.65.248。那么在网站后台就输出如下监控信息，此信息每 10 秒刷新一次。

```tex
         IP访问监控
+-----ip-address-----+--num--+
|     192.168.0.135  |   15  |
|     61.129.65.248  |   20  |
+--------------------+-------+
```

在进行具体制作之前，先对功能做具体的分析

1. 数据记录在什么位置

    最终记录的数据是一个字符串（IP 地址）对应一个数字（访问次数），此处可以选择的数据存储模型可以使用 java 提供的 map 模型，也就是 key-value 的键值对模型，或者具有 key-value 键值对模型的存储技术，例如 redis 技术。本案例使用 map 作为实现方案。

2. 统计功能运行位置，因为每次 web 请求都需要进行统计，因此使用拦截器会是比较好的方案，本案例使用拦截器来实现。不过在制作初期，先使用调用的形式进行测试，等功能完成了，再改成拦截器的实现方案。

3. 为了提升统计数据展示的灵活度，为统计功能添加配置项。输出频度，输出的数据格式，统计数据的显示模式均可以通过配置实现调整。

    - 输出频度，默认 10 秒
    - 数据特征：累计数据 / 阶段数据，默认累计数据
    - 输出格式：详细模式 / 极简模式 

在下面的制作中，分成若干个步骤实现。先完成最基本的统计功能的制作，然后开发出统计报表，接下来把所有的配置都设置好，最后将拦截器功能实现，整体功能就做完了。

### IP计数业务功能开发（自定义starter）

本功能最终要实现的效果是在现有的项目中导入一个 starter，对应的功能就添加上了，删除掉对应的 starter，功能就消失了，要求功能要与原始项目完全解耦。因此需要开发一个独立的模块，制作对应功能。

<b>步骤一：创建全新的模块，定义业务功能类</b>

功能类的制作并不复杂，定义一个业务类，声明一个 Map 对象，用于记录 ip 访问次数，key 是 ip 地址，value 是访问次数

```JAVA
public class IpCountService {
    private Map<String,Integer> ipCountMap = new HashMap<String,Integer>();
}
```

有些小伙伴可能会有疑问，不设置成静态的，如何在每次请求时进行数据共享呢？记得，当前类加载成 bean 以后是一个单例对象，对象都是单例的，哪里存在多个对象共享变量的问题。

<b>步骤二：制作统计功能</b>

制作统计操作对应的方法，每次访问后对应 ip 的记录次数 +1。需要分情况处理，如果当前没有对应 ip 的数据，新增一条数据，否则就修改对应 key 的值 +1 即可

```JAVA
public class IpCountService {
    private Map<String,Integer> ipCountMap = new HashMap<String,Integer>();
    public void count(){
        //每次调用当前操作，就记录当前访问的IP，然后累加访问次数
        //1.获取当前操作的IP地址
        String ip = null;
        //2.根据IP地址从Map取值，并递增
        Integer count = ipCountMap.get(ip);
        if(count == null){
            ipCountMap.put(ip,1);
        }else{
            ipCountMap.put(ip,count + 1);
        }
    }
}
```

因为当前功能最终导入到其他项目中进行，而导入当前功能的项目是一个 web 项目，可以从容器中直接获取请求对象，因此获取 IP 地址的操作可以通过自动装配得到请求对象，然后获取对应的访问 IP 地址。

```JAVA
public class IpCountService {
    private Map<String,Integer> ipCountMap = new HashMap<String,Integer>();
    @Autowired
    //当前的request对象的注入工作由使用当前starter的工程提供自动装配
    private HttpServletRequest httpServletRequest;
    public void count(){
        //每次调用当前操作，就记录当前访问的IP，然后累加访问次数
        //1.获取当前操作的IP地址
        String ip = httpServletRequest.getRemoteAddr();
        //2.根据IP地址从Map取值，并递增
        Integer count = ipCountMap.get(ip);
        if(count == null){
            ipCountMap.put(ip,1);
        }else{
            ipCountMap.put(ip,count + 1);
        }
    }
}
```

<b>步骤三：定义自动配置类</b>

我们需要做到的效果是导入当前模块即开启此功能，因此使用自动配置实现功能的自动装载，需要开发自动配置类在启动项目时加载当前功能。

```JAVA
public class IpAutoConfiguration {
    @Bean
    public IpCountService ipCountService(){
        return new IpCountService();
    }
}
```

自动配置类需要在 Spring.factories 文件中做配置方可自动运行。

```properties
# Auto Configure
org.Springframework.boot.autoconfigure.EnableAutoConfiguration=cn.itcast.autoconfig.IpAutoConfiguration
```

<b>步骤四：在原始项目中模拟调用，测试功能</b>

原始调用项目中导入当前开发的 starter

```XML
<dependency>
    <groupId>cn.itcast</groupId>
    <artifactId>ip_Spring_boot_starter</artifactId>
    <version>0.0.1-SNAPSHOT</version>
</dependency>
```

推荐选择调用方便的功能做测试，推荐使用分页操作，当然也可以换其他功能位置进行测试。

```JAVA
@RestController
@RequestMapping("/books")
public class BookController {
    @Autowired
    private IpCountService ipCountService;
    @GetMapping("{currentPage}/{pageSize}")
    public R getPage(@PathVariable int currentPage,@PathVariable int pageSize,Book book){
        ipCountService.count();
        IPage<Book> page = bookService.getPage(currentPage, pageSize,book);
        if( currentPage > page.getPages()){
            page = bookService.getPage((int)page.getPages(), pageSize,book);
        }
        return new R(true, page);
    }
}
```

<font color="#f0f"><b>温馨提示</b></font>

由于当前制作的功能需要在对应的调用位置进行坐标导入，因此必须保障仓库中具有当前开发的功能，所以每次原始代码修改后，需要重新编译并安装到仓库中。为防止问题出现，建议每次安装之前先 clean 然后 install，保障资源进行了更新。切记切记！！

**当前效果**

每次调用分页操作后，可以在控制台输出当前访问的 IP 地址，此功能可以在 count 操作中添加日志或者输出语句进行测试。

### 定时任务报表开发

当前已经实现了在业务功能类中记录访问数据，但是还没有输出监控的信息到控制台。由于监控信息需要每 10 秒输出 1 次，因此需要使用定时器功能。可以选取第三方技术 Quartz 实现，也可以选择 Spring 内置的 task 来完成此功能，此处选用 Spring 的 task 作为实现方案。

<b>步骤一：开启定时任务功能</b>

定时任务功能开启需要在当前功能的总配置中设置，结合现有业务设定，比较合理的位置是设置在自动配置类上。加载自动配置类即启用定时任务功能。

```JAVA
@EnableScheduling
public class IpAutoConfiguration {
    @Bean
    public IpCountService ipCountService(){
        return new IpCountService();
    }
}
```

<b>步骤二：制作显示统计数据功能</b>

定义显示统计功能的操作 print()，并设置定时任务，当前设置每 5 秒运行一次统计数据。

```JAVA
public class IpCountService {
    private Map<String,Integer> ipCountMap = new HashMap<String,Integer>();
    @Scheduled(cron = "0/5 * * * * ?")
    public void print(){
        System.out.println("         IP访问监控");
        System.out.println("+-----ip-address-----+--num--+");
        for (Map.Entry<String, Integer> entry : ipCountMap.entrySet()) {
            String key = entry.getKey();
            Integer value = entry.getValue();
            System.out.println(String.format("|%18s  |%5d  |",key,value));
        }
        System.out.println("+--------------------+-------+");
      }
}
```

其中关于统计报表的显示信息拼接可以使用各种形式进行，此处使用 String 类中的格式化字符串操作进行，学习者可以根据自己的喜好调整实现方案。

<font color="#f0f"><b>温馨提示</b></font>
每次运行效果之前先 clean 然后 install，切记切记！！

<b>当前效果</b>

每次调用分页操作后，可以在控制台看到统计数据，到此基础功能已经开发完毕。

### 使用属性配置设置功能参数

由于当前报表显示的信息格式固定，为提高报表信息显示的灵活性，需要通过 yml 文件设置参数，控制报表的显示格式。

<b>步骤一：定义参数格式</b>

设置 3 个属性，分别用来控制显示周期（cycle），阶段数据是否清空（cycleReset），数据显示格式（model）

```YAML
tools:
  ip:
    cycle: 10
    cycleReset: false
    model: "detail"
```

<b>步骤二：定义封装参数的属性类，读取配置参数</b>

为防止项目组定义的参数种类过多，产生冲突，通常设置属性前缀会至少使用两级属性作为前缀进行区分。

日志输出模式是在若干个类别选项中选择某一项，对于此种分类性数据建议制作枚举定义分类数据，当然使用字符串也可以。

```JAVA
@ConfigurationProperties(prefix = "tools.ip")
public class IpProperties {
    /**
     * 日志显示周期
     */
    private Long cycle = 5L;
    /**
     * 是否周期内重置数据
     */
    private Boolean cycleReset = false;
    /**
     * 日志输出模式  detail：详细模式  simple：极简模式
     */
    private String model = LogModel.DETAIL.value;
    public enum LogModel{
        DETAIL("detail"),
        SIMPLE("simple");
        private String value;
        LogModel(String value) {
            this.value = value;
        }
        public String getValue() {
            return value;
        }
    }
}
```

<b>步骤三：加载属性类</b>

```JAVA
@EnableScheduling
@EnableConfigurationProperties(IpProperties.class)
public class IpAutoConfiguration {
    @Bean
    public IpCountService ipCountService(){
        return new IpCountService();
    }
}
```

<b>步骤四：应用配置属性</b>

在应用配置属性的功能类中，使用自动装配加载对应的配置 bean，然后使用配置信息做分支处理。

注意：清除数据的功能一定要在输出后运行，否则每次查阅的数据均为空白数据。

```JAVA
public class IpCountService {
    private Map<String,Integer> ipCountMap = new HashMap<String,Integer>();
    @Autowired
    private IpProperties ipProperties;
    @Scheduled(cron = "0/5 * * * * ?")
    public void print(){
        if(ipProperties.getModel().equals(IpProperties.LogModel.DETAIL.getValue())){
            System.out.println("         IP访问监控");
            System.out.println("+-----ip-address-----+--num--+");
            for (Map.Entry<String, Integer> entry : ipCountMap.entrySet()) {
                String key = entry.getKey();
                Integer value = entry.getValue();
                System.out.println(String.format("|%18s  |%5d  |",key,value));
            }
            System.out.println("+--------------------+-------+");
        }else if(ipProperties.getModel().equals(IpProperties.LogModel.SIMPLE.getValue())){
            System.out.println("     IP访问监控");
            System.out.println("+-----ip-address-----+");
            for (String key: ipCountMap.keySet()) {
                System.out.println(String.format("|%18s  |",key));
            }
            System.out.println("+--------------------+");
        }
        //阶段内统计数据归零
        if(ipProperties.getCycleReset()){
            ipCountMap.clear();
        }
    }
}
```

<font color="#f0f"><b>温馨提示</b></font>

每次运行效果之前先 clean 然后 install，切记切记！！

<b>当前效果</b>

在 web 程序端可以通过控制 yml 文件中的配置参数对统计信息进行格式控制。但是数据显示周期还未进行控制。

### 使用属性配置设置定时器参数

在使用属性配置中的显示周期数据时，遇到了一些问题。由于无法在 @Scheduled 注解上直接使用配置数据，改用曲线救国的方针，放弃使用 @EnableConfigurationProperties 注解对应的功能，改成最原始的 bean 定义格式。

<b>步骤一：@Scheduled 注解使用#{}读取 bean 属性值</b>

此处读取 bean 名称为 ipProperties 的 bean 的 cycle 属性值

```JAVA
@Scheduled(cron = "0/#{ipProperties.cycle} * * * * ?")
public void print(){}
```

<b>步骤二：属性类定义 bean 并指定 bean 的访问名称</b>

如果此处不设置 bean 的访问名称，Spring 会使用自己的命名生成器生成 bean 的长名称，无法实现属性的读取

```JAVA
@Component("ipProperties")
@ConfigurationProperties(prefix = "tools.ip")
public class IpProperties {}
```

<b>步骤三：弃用 @EnableConfigurationProperties 注解对应的功能，改为导入 bean 的形式加载配置属性类</b>

```JAVA
@EnableScheduling
//@EnableConfigurationProperties(IpProperties.class)
@Import(IpProperties.class)
public class IpAutoConfiguration {
    @Bean
    public IpCountService ipCountService(){
        return new IpCountService();
    }
}
```

<font color="#f0f"><b>温馨提示</b></font>

每次运行效果之前先 clean 然后 install，切记切记！！

<b>当前效果</b>

在 web 程序端可以通过控制 yml 文件中的配置参数对统计信息的显示周期进行控制

### 拦截器开发

基础功能基本上已经完成了制作，下面进行拦截器的开发。开发时先在 web 工程中制作，然后将所有功能挪入 starter 模块中

<b>步骤一：开发拦截器</b>

使用自动装配加载统计功能的业务类，并在拦截器中调用对应功能

```JAVA
public class IpCountInterceptor implements HandlerInterceptor {
    @Autowired
    private IpCountService ipCountService;
    @Override
    public boolean preHandle(HttpServletRequest request, 
                             HttpServletResponse response, Object handler) throws Exception {
        ipCountService.count();
        return true;
    }
}
```

<b>步骤二：配置拦截器</b>

配置 mvc 拦截器，设置拦截对应的请求路径。此处拦截所有请求，用户可以根据使用需要设置要拦截的请求。甚至可以在此处加载 IpCountProperties 中的属性，通过配置设置拦截器拦截的请求。

```JAVA
@Configuration
public class SpringMvcConfig implements WebMvcConfigurer {
    @Override
    public void addInterceptors(InterceptorRegistry registry) {
        registry.addInterceptor(ipCountInterceptor()).addPathPatterns("/**");
    }
    @Bean
    public IpCountInterceptor ipCountInterceptor(){
        return new IpCountInterceptor();
    }
}
```

<font color="#f0f"><b>温馨提示</b></font>

每次运行效果之前先 clean 然后 install，切记切记！！

<b>当前效果</b>

在 web 程序端导入对应的 starter 后功能开启，去掉坐标后功能消失，实现自定义 starter 的效果。

到此当前案例全部完成，自定义 stater 的开发其实在第一轮开发中就已经完成了，就是创建独立模块导出独立功能，需要使用的位置导入对应的 starter 即可。如果是在企业中开发，记得不仅需要将开发完成的 starter 模块 install 到自己的本地仓库中，开发完毕后还要 deploy 到私服上，否则别人就无法使用了。

### 功能性完善——开启yml提示功能

我们在使用 Spring Boot 的配置属性时，都可以看到提示，尤其是导入了对应的 starter 后，也会有对应的提示信息出现。但是现在我们的 starter 没有对应的提示功能，这种设定就非常的不友好，本节解决自定义 starter 功能如何开启配置提示的问题。

Spring Boot 提供有专用的工具实现此功能，仅需要导入下列坐标。

```xml
<dependency>
    <groupId>org.Springframework.boot</groupId>
    <artifactId>Spring-boot-configuration-processor</artifactId>
    <optional>true</optional>
</dependency>
```

程序编译后，在 META-INF 目录中会生成对应的提示文件，然后拷贝生成出的文件到自己开发的 META-INF 目录中，并对其进行编辑。打开生成的文件，可以看到如下信息。其中 groups 属性定义了当前配置的提示信息总体描述，当前配置属于哪一个属性封装类，properties 属性描述了当前配置中每一个属性的具体设置，包含名称、类型、描述、默认值等信息。hints 属性默认是空白的，没有进行设置。hints 属性可以参考 Spring Boot 源码中的制作，设置当前属性封装类专用的提示信息，下例中为日志输出模式属性 model 设置了两种可选提示信息。

```JAVA
{
  "groups": [
    {
      "name": "tools.ip",
      "type": "cn.itcast.properties.IpProperties",
      "sourceType": "cn.itcast.properties.IpProperties"
    }
  ],
  "properties": [
    {
      "name": "tools.ip.cycle",
      "type": "java.lang.Long",
      "description": "日志显示周期",
      "sourceType": "cn.itcast.properties.IpProperties",
      "defaultValue": 5
    },
    {
      "name": "tools.ip.cycle-reset",
      "type": "java.lang.Boolean",
      "description": "是否周期内重置数据",
      "sourceType": "cn.itcast.properties.IpProperties",
      "defaultValue": false
    },
    {
      "name": "tools.ip.model",
      "type": "java.lang.String",
      "description": "日志输出模式  detail：详细模式  simple：极简模式",
      "sourceType": "cn.itcast.properties.IpProperties"
    }
  ],
  "hints": [
    {
      "name": "tools.ip.model",
      "values": [
        {
          "value": "detail",
          "description": "详细模式."
        },
        {
          "value": "simple",
          "description": "极简模式."
        }
      ]
    }
  ]
}
```

<b>总结</b>

1. 自定义 starter 其实就是做一个独立的功能模块，核心技术是利用自动配置的效果在加载模块后加载对应的功能
2. 通常会为自定义 starter 的自动配置功能添加足够的条件控制，而不会做成 100% 加载对功能的效果
3. 本例中使用 map 保存数据，如果换用 redis 方案，在 starter 开发模块中就要导入 redis 对应的 starter
4. 对于配置属性务必开启提示功能，否则使用者无法感知配置应该如何书写

## Spring Boot程序启动流程解析

原理篇学习到这里即将结束，最后一章说一下 Spring Boot 程序的启动流程。对于 Spring Boot 技术来说，它用于加速 Spring 程序的开发，核心本质还是 Spring 程序的运行，所以于其说是 Spring Boot 程序的启动流程，不如说是 Spring Boot 对 Spring 程序的启动流程做了哪些更改。

其实不管是 Spring Boot 程序还是 Spring 程序，启动过程本质上都是在做容器的初始化，并将对应的 bean 初始化出来放入容器。在 Spring 环境中，每个 bean 的初始化都要开发者自己添加设置，但是切换成 Spring Boot 程序后，自动配置功能的添加帮助开发者提前预设了很多 bean 的初始化过程，加上各种各样的参数设置，使得整体初始化过程显得略微复杂，但是核心本质还是在做一件事，初始化容器。作为开发者只要搞清楚 Spring Boot 提供了哪些参数设置的环节，同时初始化容器的过程中都做了哪些事情就行了。

Spring Boot 初始化的参数根据参数的提供方，划分成如下 3 个大类，每个大类的参数又被封装了各种各样的对象，具体如下：

- 环境属性（Environment）
- 系统配置（Spring.factories）
- 参数（Arguments、application.properties）

以下通过代码流向介绍了 Spring Boot 程序启动时每一环节做的具体事情。

```JAVA
Spring Boot30StartupApplication【10】->SpringApplication.run(Spring Boot30StartupApplication.class, args);
    SpringApplication【1332】->return run(new Class<?>[] { primarySource }, args);
        SpringApplication【1343】->return new SpringApplication(primarySources).run(args);
            SpringApplication【1343】->SpringApplication(primarySources)
            # 加载各种配置信息，初始化各种配置对象
                SpringApplication【266】->this(null, primarySources);
                    SpringApplication【280】->public SpringApplication(ResourceLoader resourceLoader, Class<?>... primarySources)
                        SpringApplication【281】->this.resourceLoader = resourceLoader;
                        # 初始化资源加载器
                        SpringApplication【283】->this.primarySources = new LinkedHashSet<>(Arrays.asList(primarySources));
                        # 初始化配置类的类名信息（格式转换）
                        SpringApplication【284】->this.webApplicationType = WebApplicationType.deduceFromClasspath();
                        # 确认当前容器加载的类型
                        SpringApplication【285】->this.bootstrapRegistryInitializers = getBootstrapRegistryInitializersFromSpringFactories();
                        # 获取系统配置引导信息
                        SpringApplication【286】->setInitializers((Collection) getSpringFactoriesInstances(ApplicationContextInitializer.class));
                        # 获取ApplicationContextInitializer.class对应的实例
                        SpringApplication【287】->setListeners((Collection) getSpringFactoriesInstances(ApplicationListener.class));
                        # 初始化监听器，对初始化过程及运行过程进行干预
                        SpringApplication【288】->this.mainApplicationClass = deduceMainApplicationClass();
                        # 初始化了引导类类名信息，备用
            SpringApplication【1343】->new SpringApplication(primarySources).run(args)
            # 初始化容器，得到ApplicationContext对象
                SpringApplication【323】->StopWatch stopWatch = new StopWatch();
                # 设置计时器
                SpringApplication【324】->stopWatch.start();
                # 计时开始
                SpringApplication【325】->DefaultBootstrapContext bootstrapContext = createBootstrapContext();
                # 系统引导信息对应的上下文对象
                SpringApplication【327】->configureHeadlessProperty();
                # 模拟输入输出信号，避免出现因缺少外设导致的信号传输失败，进而引发错误（模拟显示器，键盘，鼠标...）
                    java.awt.headless=true
                SpringApplication【328】->SpringApplicationRunListeners listeners = getRunListeners(args);
                # 获取当前注册的所有监听器
                SpringApplication【329】->listeners.starting(bootstrapContext, this.mainApplicationClass);
                # 监听器执行了对应的操作步骤
                SpringApplication【331】->ApplicationArguments applicationArguments = new DefaultApplicationArguments(args);
                # 获取参数
                SpringApplication【333】->ConfigurableEnvironment environment = prepareEnvironment(listeners, bootstrapContext, applicationArguments);
                # 将前期读取的数据加载成了一个环境对象，用来描述信息
                SpringApplication【333】->configureIgnoreBeanInfo(environment);
                # 做了一个配置，备用
                SpringApplication【334】->Banner printedBanner = printBanner(environment);
                # 初始化logo
                SpringApplication【335】->context = createApplicationContext();
                # 创建容器对象，根据前期配置的容器类型进行判定并创建
                SpringApplication【363】->context.setApplicationStartup(this.applicationStartup);
                # 设置启动模式
                SpringApplication【337】->prepareContext(bootstrapContext, context, environment, listeners, applicationArguments, printedBanner);
                # 对容器进行设置，参数来源于前期的设定
                SpringApplication【338】->refreshContext(context);
                # 刷新容器环境
                SpringApplication【339】->afterRefresh(context, applicationArguments);
                # 刷新完毕后做后处理
                SpringApplication【340】->stopWatch.stop();
                # 计时结束
                SpringApplication【341】->if (this.logStartupInfo) {
                # 判定是否记录启动时间的日志
                SpringApplication【342】->    new StartupInfoLogger(this.mainApplicationClass).logStarted(getApplicationLog(), stopWatch);
                # 创建日志对应的对象，输出日志信息，包含启动时间
                SpringApplication【344】->listeners.started(context);
                # 监听器执行了对应的操作步骤
                SpringApplication【345】->callRunners(context, applicationArguments);
                # 调用运行器
                SpringApplication【353】->listeners.running(context);
                # 监听器执行了对应的操作步骤
```

上述过程描述了 Spring Boot 程序启动过程中做的所有的事情。如果想干预 Spring Boot 的启动过程，比如自定义一个数据库环境检测的程序，该如何将这个过程加入 Spring Boot 的启动流程呢？

遇到这样的问题，大部分技术是这样设计的，设计若干个标准接口，对应程序中的所有标准过程。当你想干预某个过程时，实现接口就行了。例如 Spring 技术中 bean 的生命周期管理就是采用标准接口进行的。

```JAVA
public class Abc implements InitializingBean, DisposableBean {
    public void destroy() throws Exception {
        //销毁操作
    }
    public void afterPropertiesSet() throws Exception {
        //初始化操作
    }
}
```

Spring Boot 启动过程由于存在着大量的过程阶段，如果设计接口就要设计十余个标准接口，这样对开发者不友好，同时整体过程管理分散，十余个过程各自为政，管理难度大，过程过于松散。那 Spring Boot 如何解决这个问题呢？它采用了一种最原始的设计模式来解决这个问题，这就是监听器模式，使用监听器来解决这个问题。

Spring Boot 将自身的启动过程比喻成一个大的事件，该事件是由若干个小的事件组成的。例如：

- org.Springframework.boot.context.event.ApplicationStartingEvent
    - 应用启动事件，在应用运行但未进行任何处理时，将发送 ApplicationStartingEvent
- org.Springframework.boot.context.event.ApplicationEnvironmentPreparedEvent
    - 环境准备事件，当 Environment 被使用，且上下文创建之前，将发送 ApplicationEnvironmentPreparedEvent
- org.Springframework.boot.context.event.ApplicationContextInitializedEvent
    - 上下文初始化事件
- org.Springframework.boot.context.event.ApplicationPreparedEvent
    - 应用准备事件，在开始刷新之前，bean 定义被加载之后发送 ApplicationPreparedEvent
- org.Springframework.context.event.ContextRefreshedEvent
    - 上下文刷新事件
- org.Springframework.boot.context.event.ApplicationStartedEvent
    - 应用启动完成事件，在上下文刷新之后且所有的应用和命令行运行器被调用之前发送 ApplicationStartedEvent
- org.Springframework.boot.context.event.ApplicationReadyEvent
    - 应用准备就绪事件，在应用程序和命令行运行器被调用之后，将发出 ApplicationReadyEvent，用于通知应用已经准备处理请求
- org.Springframework.context.event.ContextClosedEvent（上下文关闭事件，对应容器关闭）

上述列出的仅仅是部分事件，当应用启动后走到某一个过程点时，监听器监听到某个事件触发，就会执行对应的事件。除了系统内置的事件处理，用户还可以根据需要自定义开发当前事件触发时要做的其他动作。

```JAVA
//设定监听器，在应用启动开始事件时进行功能追加
public class MyListener implements ApplicationListener<ApplicationStartingEvent> {
    public void onApplicationEvent(ApplicationStartingEvent event) {
		//自定义事件处理逻辑
    }
}
```

按照上述方案处理，用户就可以干预 Spring Boot 启动过程的所有工作节点，设置自己的业务系统中独有的功能点。

<b>总结</b>

1. Spring Boot 启动流程是先初始化容器需要的各种配置，并加载成各种对象，初始化容器时读取这些对象，创建容器
2. 整体流程采用事件监听的机制进行过程控制，开发者可以根据需要自行扩展，添加对应的监听器绑定具体事件，就可以在事件触发位置执行开发者的业务代码

番外篇，さようなら！ 안녕히 계십시오！แล้วเจอกัน！До свидания ！خداحافظ ！
