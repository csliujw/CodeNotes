# 一、Http协议

## 小知识

> **扩展**

HTTP协议是明文传输，不安全，不适合传输安全性要求高的文件，如密码

HTTPS，对HTTP协议不安全的改进，叫安全套接字超文本传输协议，采用了基于SSL(Secure Sockets Layer)进行加密，安全性高！SSL依靠证书来验证服务器的身份，并为浏览器和服务器之间的通信加密。

> HTTP和HTTPS的对比

- https相对于http加入了ssl层，
- 需要到ca申请收费的证书，SSL证书需要钱，功能越强大的证书费用越高，个人网站、小网站没有必要一般不会用。
- 安全但是耗时多，缓存不是很好，HTTPS协议握手阶段比较费时，会使页面的加载时间延长近50%，增加10%到20%的耗电；
- 注意兼容http和https

> **SSL**

​			SSL协议位于[TCP/IP协议](https://baike.baidu.com/item/TCP%2FIP协议)与各种[应用层](https://baike.baidu.com/item/应用层)协议之间，为[数据通讯](https://baike.baidu.com/item/数据通讯)提供安全支持。SSL协议可分为两层： SSL记录协议（SSL Record Protocol）：它建立在可靠的[传输协议](https://baike.baidu.com/item/传输协议)（如TCP）之上，为高层协议提供[数据封装](https://baike.baidu.com/item/数据封装)、压缩、加密等基本功能的支持。 SSL[握手协议](https://baike.baidu.com/item/握手协议)（SSL Handshake Protocol）：它建立在SSL记录协议之上，用于在实际的数据传输开始前，通讯双方进行[身份认证](https://baike.baidu.com/item/身份认证)、协商[加密算法](https://baike.baidu.com/item/加密算法)、交换加密[密钥](https://baike.baidu.com/item/密钥)等。

## 1.1 基本知识

> **概念：Hyper Text Transfer Protocol 超文本传输协议**

> **传输协议：定义了，客户端和服务器端通信时，发送数据的格式**

- **基于TCP/IP的高级协议**
- **默认端口号:80**
  - 常见端口8080 tomcat端口
  - 3306 `MySQL`端口
  - 6379 `Redis`端口
  - 1521 oracle端口
- **基于请求/响应模型的：一次请求对应一次响应**
  - tomcat采用的`NIO`，一次请求对应一个新的线程
- **无状态的：每次请求之间相互独立，不能交互数据**
- **历史版本**
  - 1.0：每一次请求响应都会建立新的连接
  - 1.1：复用连接

## 1.2 请求消息数据格式

### 1.2.1 请求行

- 请求方式 请求url 请求协议/版本
  - GET /login.html	HTTP/1.1

- HTTP协议有7种请求方式，常用的有2种
  - GET：请求参数在请求行中，在url后；请求的url长度有限制的；不太安全。
  - POST：请求参数在请求体中；请求的url长度没有限制的；相对安全。

### 1.2.2 请求头

> **客户端浏览器告诉服务器一些信息**，格式==请求头名称: 请求头值==

- 常见的请求头：
  - User-Agent：浏览器告诉服务器，我访问你使用的浏览器的版本信息；可以在服务器端获取该头的信息，解决浏览器的兼容性问题。
  - Referer：http://localhost/login.html，告诉服务器，我(当前请求)从哪里来？
    - 可以防盗链
    - 可以进行统计工作

### 1.2.3 请求空行

空行，就是用于分割POST请求的请求头，和请求体的。

### 1.2.4 请求体(正文)：

封装POST请求消息的请求参数的

字符格式

```http
请求方式 请求的url      协议版本
POST    /login.html	  HTTP/1.1

// 主机名称
Host: localhost

// 浏览器信息
User-Agent: Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0

// 可接收的文件格式
Accept:text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8

// 可接收的语言
Accept-Language: zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2

Accept-Encoding: gzip, deflate

Referer: http://localhost/login.html

Connection: keep-alive

Upgrade-Insecure-Requests: 1

username=zhangsan	
```

## 1.3 响应消息数据格式

> **响应消息：服务器端发送给客户端的数据**

### 1.3.1 响应消息基础知识

* 数据格式：
* 响应行
	* 组成：协议/版本 响应状态码 状态码描述 如HTTP/1.1 200 Ok
	* 响应状态码：服务器告诉客户端浏览器本次请求和响应的一个状态。
	* 状态码都是3位数字 

- **状态码分类**
  - 1xx：服务器就收客户端消息，但没有接受完成，等待一段时间后，发送1xx多状态码
  - 2xx：成功。代表：200
  - 3xx：重定向。代表：302(重定向)，304(访问缓存)
  - 4xx：客户端错误。
  - 5xx：服务器端错误。

* **典型状态码代表**：
	* 404：请求路径没有对应的资源）
	* 405：请求方式没有对应的doXxx方法
	* 500：服务器内部出现异常
	* 302：重定向
	* 304：访问缓存

----

### 1.3.2 常见响应消息

> 格式：头名称： 值

- **Content-Type：**服务器告诉客户端本次响应体数据格式以及编码格式

- **Content-disposition：**服务器告诉客户端以什么格式打开响应体数据

  - in-line:默认值,在当前页面内打开【值】
  - attachment;filename=xxx：以附件形式打开响应体。文件下载【值】

- 响应空行

- **响应体:**传输的数据

- **响应字符串格式**

  HTTP/1.1 200 OK
  Content-Type: text/html;charset=UTF-8
  Content-Length: 101
  Date: Wed, 06 Jun 2018 07:08:42 GMT

  <html>  

  <body>
    hello , response
   </body>
  </html>

----

# 二、tomcat基础知识

- web相关概念
- web服务器软件：Tomcat
- Servlet入门学习

## 2.1 web相关概念

- **软件架构**
  - **C/S：客户端/服务器端；**一般主要的计算由客户端自行完成，服务器用作数据交互，存储客户端计算出的数据结果。
  - **B/S**：**浏览器/服务器端**：服务器承担主要的计算。
- **资源分类**
  - **静态资源**：所有用户访问后，得到的结果都是一样的，称为静态资源.静态资源可以直接被浏览器解析【如：html，css，js】
  - **动态资源**：每个用户访问相同资源后，得到的结果可能不一样。称为动态资源。动态资源被访问后，需要先转换为静态资源，在返回给浏览器【如：servlet/jsp，php】
- **网络通信三要素**
  - IP：电子设备(计算机)在网络中的唯一标识。
  - 端口：应用程序在计算机中的唯一标识。 0~65536
  - 传输协议：规定了数据传输的规则
    - 基础协议：**TCP**【安全协议，三次握手，速度稍慢】**UDP**：【不安全协议。 速度快】

## 2.2 web服务器软件

- 服务器：安装了服务器软件的计算机
- 服务器软件：接收用户的请求，处理请求，做出响应。
- web服务器软件：接收用户的请求，处理请求，做出响应。
  - 在web服务器软件中，可以部署web项目，让用户通过浏览器来访问这些项目。
- 常见的Java相关的web服务器软件：
  - webLogic：oracle公司，大型的JavaEE服务器，支持所有的JavaEE规范，收费的。
  - webSphere：IBM公司，大型的JavaEE服务器，支持所有的JavaEE规范，收费的。
  - JBOSS：JBOSS公司的，大型的JavaEE服务器，支持所有的JavaEE规范，收费的。
  - Tomcat：Apache基金组织，中小型的JavaEE服务器，仅仅支持少量的JavaEE规范servlet/jsp。开源的，**免费**的。
- JavaEE：Java语言在企业级开发中使用的技术规范的总和，一共规定了13项大的规范

## 2.3 Tomcat的安装启动

### 2.3.1 安装

- 下载：http://tomcat.apache.org/
- 安装：解压压缩包即可。
- 卸载：删除目录就行了
- 启动：
  - bin/startup.bat ,双击运行该文件即可
  - 访问：浏览器输入：http://localhost:8080 回车访问自己
  - http://别人的ip:8080 访问别人

### 2.3.2 可能遇到的问题：

- 黑窗口一闪而过：

  - 原因： 没有正确配置JAVA_HOME环境变量，配置下JAVA_HOME即可。

- 启动报错：

  - 暴力：找到占用的端口号，并且找到对应的进程，杀死该进程【Windows下用 netstat -ano】

  - 温柔：修改自身的端口号

    ```xml
    conf/server.xml
    <Connector port="8888" protocol="HTTP/1.1"
    		               connectionTimeout="20000"
    		               redirectPort="8445" />
    
    ```

    一般会将tomcat的默认端口号修改为80。80端口号是HTTP协议的默认端口号。
    好处：在访问时，就不用输入端口号

### 2.3.3 关闭

- 正常关闭
  - bin/shutdown.bat
  - ctrl+c
- 强制关闭
  - 点击启动窗口的==×==

## 2.4 部署项目

- **部署方式**

  - 1.直接将项目放到webapps目录下即可。
  - 2.配置conf/server.xml文件
  - 3.在conf\Catalina\localhost创建任意名称的xml文件。在文件中编写

- **webapps部署方式**

  - 直接将项目放到webapps目录下
  - /hello：项目的访问路径会被映射为虚拟目录
  - 简化部署：将项目打成一个war包，再将war包放置到webapps目录下。war包会被自动解压。【war包的压缩方式和zip的压缩方式一样】

- **conf/server.xml部署方式**

  - 在<Host>标签体中配置

    <Context docBase="D:\hello" path="/hehe" />

    docBase：项目存放的路径

    path：虚拟目录

- **conf\Catalina\localhost配置方式**

  - 在conf\Catalina\localhost创建任意名称的xml文件。在文件中编写<Context docBase="D:\hello" />
  - D:\hello 是项目的绝对路径
  - 虚拟目录：xml文件的名称。如xml名称为demo，那么虚拟目录就是`localhost:8080/demo`

- **静态项目和动态项目：**

【java动态项目的目录结构】

|-- 项目的根目录

​	|-- WEB-INF目录：WEB-INF下的文件无法通过URL直接访问

​		|-- web.xml：web项目的核心配置文件

​		|-- classes目录：放置字节码文件的目录

​		|-- lib目录：放置依赖的jar包



> **URL和URI**

URL：统一资源定位符，用于定位这个资源在哪里。

URI：统一资源标识符，标识这个资源唯一。

----

# 三、Servlet

> **Servlet：server applet**。运行在服务器端的小程序。

Servlet就是一个接口，定义了Java类被浏览器访问到(tomcat识别)的规则。

Java定制规范，提供接口，其他厂商根据规范和接口进行实际的功能实现。

## 3.1 Servlet的配置

> **XML和注解不能同时配置一个`Servlet`，不过如果配置的`URL`不一样，那么就没事，是可以的！**

```xml
在web.xml中配置：[xml解析，servlet-name一样的进行匹配！]
<!--配置Servlet -->
<servlet>
    <servlet-name>demo1</servlet-name>
    <servlet-class>cn.web.servlet.ServletDemo1</servlet-class>
</servlet>

<servlet-mapping>
    <servlet-name>demo1</servlet-name>
    <url-pattern>/demo1</url-pattern>
</servlet-mapping>
```

## 3.2 `Servlet`执行原理

- 当服务器接受到客户端浏览器的请求后，会解析请求URL路径，获取访问的Servlet的资源路径
- 查找`web.xml`文件，是否有对应的<url-pattern>标签体内容。
- 如果有，则在找到对应的<servlet-class>全类名
- tomcat会将==字节码文件==加载进内存，并且创建其对象
- 调用其方法
  - **servlet** 的service方法是一定会调用的，

## 3.3 Servlet生命周期方法

- 被创建：执行init方法，只执行一次

  - 默认情况下，第一次被访问时，Servlet被创建

  - 可以配置执行Servlet的创建时机。

    ```xml
    在<servlet>标签下配置
    第一次被访问时，创建
    <load-on-startup>的值为负数
    
    在服务器启动时，创建
    <load-on-startup>的值为0或正整数
    ```

- Servlet的init方法，只执行一次，说明一个Servlet在内存中只存在一个对象，Servlet是单例的
  - 多个用户同时访问时，可能存在线程安全问题。
  - 解决：尽量不要在Servlet中定义成员变量。即使定义了成员变量，也不要对修改值

----

- 提供服务：执行service方法，执行多次
  - 每次访问Servlet时，service方法都会被调用一次。

----

- 被销毁：执行destroy方法，只执行一次
  - Servlet被销毁时执行。服务器关闭时，Servlet被销毁
  - 只有服务器正常关闭时，才会执行destroy方法。
  - destroy方法在Servlet被销毁之前执行，一般用于释放资源

## 3.4 Servlet3.0

* 好处：
	
* 支持注解配置。可以不需要web.xml了。
	
* 步骤：
	- 创建JavaEE项目，选择Servlet的版本3.0以上，可以不创建web.xml
	
	- 定义一个类，实现Servlet接口
	
	- 复写方法
	
	- 在类上使用@WebServlet注解，进行配置
	
	* @WebServlet("资源路径")`
	  * **注意：name是指类的名称不是资源路径**


```java
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface WebServlet {
    //相当于<Servlet-name>
    String name() default "";
    
	//代表urlPatterns()属性配置
    String[] value() default {};
    
	//相当于<url-pattern>
    String[] urlPatterns() default {};
    
	//相当于<load-on-startup>
    int loadOnStartup() default -1;

    WebInitParam[] initParams() default {};

    boolean asyncSupported() default false;

    String smallIcon() default "";

    String largeIcon() default "";

    String description() default "";

    String displayName() default "";
}
```

> **IDEA与tomcat的相关配置**

- IDEA会为每一个tomcat部署的项目单独建立一份配置文件
- 工作空间项目和tomcat部署的web项目
  - tomcat真正访问的是“tomcat部署的web项目”，"tomcat部署的web项目"对应着"工作空间项目" 的web目录下的所有资源
  - WEB-INF目录下的资源不能被浏览器直接访问。
- **WEB-INF目录下的资源不能被浏览器直接访问。**

## 3.5 Servlet继承关系

### 3.5.1 Servlet的体系结构	
​	Servlet -- 接口
​		|
​	GenericServlet -- 抽象类
​		|
​	HttpServlet  -- 抽象类

- GenericServlet：将Servlet接口中其他的方法做了默认空实现，只将service()方法作为抽象
  - 将来定义Servlet类时，可以继承GenericServlet，实现service()方法即可
- HttpServlet：对http协议的一种封装，简化操作
  - 定义类继承HttpServlet
  - 复写doGet/doPost方法

### 3.5.2 Servlet相关配置

- `urlpartten：Servlet访问路径`

- 一个Servlet可以定义多个访问路径 ： 
  
- `@WebServlet({"/d4","/dd4","/ddd4"})`
  
- 路径定义规则：
  - /xxx：路径匹配
  - /xxx/xxx:多层路径，目录结构
  - *.do：扩展名匹配
  - **不能混合使用路径匹配和扩展名匹配。**

  ```xml
  <!-- 合法 -->
  <servlet>
      <servlet-name>demo1</servlet-name>
      <servlet-class>cn.servlet</servlet-class>
  </servlet>
  
  <servlet-mapping>
      <servlet-name>demo1</servlet-name>
      <url-pattern>/user/demo1</url-pattern>
  </servlet-mapping>
  
  <!-- 合法 -->
  <servlet>
      <servlet-name>demo1</servlet-name>
      <servlet-class>cn.servlet</servlet-class>
  </servlet>
  
  <servlet-mapping>
      <servlet-name>demo1</servlet-name>
      <url-pattern>*.action</url-pattern>
  </servlet-mapping>
  
  <!-- 不合法 -->
  <servlet>
      <servlet-name>demo1</servlet-name>
      <servlet-class>cn.servlet</servlet-class>
  </servlet>
  <!-- 不合法 -->
  <servlet-mapping>
      <servlet-name>demo1</servlet-name>
      <url-pattern>/user/*.action</url-pattern>
  </servlet-mapping>
  ```

### 3.5.2 初始化资源配置

```xml
<servlet>
    <servlet-name>demo1</servlet-name>
    <servlet-class>cn.servlet</servlet-class>
    <init-param>
    	<param-name>username</param-name>
        <param-value>hello</param-value>
    </init-param>
    <init-param>
    	<param-name>config</param-name>
        <param-value>/WEB-INF/config.xml</param-value>
    </init-param>
    <load-on-startup>2</load-on-startup>
</servlet>
```

```java
public void init(ServletConfig config){
	super.init(config);
    // 获得参数
    String username = config.getInitParamenter("username");
}
```

## 3.6 Servlet配置一探究竟

- xml配置方式测试

  - `/xxx`
  - `/xx/xx`
  - `*.do` 匹配以 **.do**结尾的
  - / 不能省略

- 注解配置方式测试

  - 路径写错会报错
  - `WebServlet("/demo1")` 正确写法，不能省略/

- ==注解和xml不能同给一个类，配置相同的名字，但是可以配置不同的名字。==**[同时书写下面的，不会报错]**

  - ```java
    @WebServlet("/Servlet3")
    public class Servlet3 extends HttpServlet {
        protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
    
        }
    
        protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
            response.getWriter().write("3");
    
        }
    ```

  - ```xml
    <servlet>
        <!-- 此处是小写！ -->
        <servlet-name>servlet3</servlet-name>
        <servlet-class>com.demo.Servlet3</servlet-class>
    </servlet>
    <servlet-mapping>
        <servlet-name>servlet3</servlet-name>
        <url-pattern>/servlet3</url-pattern>
    </servlet-mapping>
    ```

----

# 四、Request&Response

## 4.1 基本原理

- request和response对象是由服务器创建的。我们来使用它们
- request对象是来获取请求消息，response对象是来设置响应消息

----

request对象继承体系结构：	
	ServletRequest		--	接口
		|	继承
	HttpServletRequest	-- 接口
		|	实现
	org.apache.catalina.connector.RequestFacade 类(tomcat)

----

response对象继承体系结构：	
	ServletRequest		--	接口
		|	继承
	HttpServletResponse   -- 接口
		|	实现
	org.apache.catalina.connector.ResponseFacade类(tomcat)

----

## 4.2 Request功能

> 由web容器自动创建。每次web服务器接收到HTTP请求时，会自动创建request对象。web服务器处理HTTP请求，向客户端发送HTTP响应后，会自动销毁请求对象。保存在对象中的数据也就消失了。

总结：

- 服务器接收到HTTP请求时创建request对象
- 服务器发送HTTP响应结束后销毁request对象
- request对象仅存活于一次请求转发

### 4.2.1 获取请求消息数据

- **获取请求行数据**
- GET /day14/demo1?name=zhangsan HTTP/1.1

- **方法**
  - 获取请求方式 ：`String getMethod()  ` == GET
  - 获取虚拟目录：`String getContextPath()`==> /day14
  - 获取Servlet路径: `String getServletPath()`==> /demo1
  - 获取get方式请求参数：`String getQueryString()`==> name=zhangsan
  - 获取请求URI：/day14/demo1
    - `String getRequestURI():`	/day14/demo1	
    - `StringBuffer getRequestURL()`  :http://localhost/day14/demo1
  - URL：统一资源定位符 ： http://localhost/day14/demo1
  - URI：统一资源标识符 : /day14/demo1	
  - 获取协议及版本：`String getProtocol()`==HTTP/1.1
  - 获取客户机的IP地址：`String getRemoteAddr()`

----

- **获取请求头数据:**
- 方法:
  - `String getHeader(String name):`通过请求头的名称获取请求头的值
  - `Enumeration<String> getHeaderNames():`获取所有的请求头名称

----

- **获取请求体数据:**
  - 请求体：只有POST请求方式，才有请求体，在请求体中封装了POST请求的请求参数
  - `BufferedReader getReader()：`获取字符输入流，只能操作字符数据
  - `ServletInputStream getInputStream()：`获取字节输入流，可以操作所有类型数据

----

- **其他功能：**
  - `String getParameter(String name):`根据参数名称获取参数值    
  - `String[] getParameterValues(String name):`根据参数名称获取参数值的数组  hobby=xx&hobby=game【复选框】
  - Enumeration<String> getParameterNames():获取所有请求的参数名称.
  - Map<String,String[]> getParameterMap():获取所有参数的map集合.

----

### 4.2.2 请求转发

> **请求转发：一种在服务器内部的资源跳转方式**

- 通过request对象获取请求转发器对象：`RequestDispatcher` `getRequestDispatcher(String path)`
- 使用`RequestDispatcher`对象来进行转发：`forward(ServletRequest request, ServletResponse response)` 
- **特点**：【容易被问】
  - 浏览器地址栏路径不发生变化
  - 只能转发到当前服务器内部资源中。
  - 转发是一次请求

**转发地址的写法！**

> **request.getRequestDispatcher("/requestDemo9").forward(request,response);**

### 4.2.3 共享数据

域对象：一个有作用范围的对象，可以在范围内共享数据

request域：代表一次请求的范围，一般用于请求转发的多个资源中共享数据

- void setAttribute(String name,Object obj):存储数据
- Object getAttitude(String name):通过键获取值
- void removeAttribute(String name):通过键移除键值对

### 4.2.4 获取ServletContext

```java
ServletContext context = request.getServletContext();
```

> `ServletContext`官方叫`servlet`上下文。服务器会为每一个工程创建一个对象，这个对象就是`ServletContext`对象。这个对象全局唯一，而且==工程内部的所有servlet都共享这个对象。所以叫全局应用程序共享对象。==

### 4.2.5 获取输入流

> **上传文件**

```java
ServletInputStream is = request.getInputStream();
```

完成一次请求后会自动地销毁嘛？

## 4.3 中文乱码问题

- get方式：tomcat 8 已经将get方式乱码问题解决了
- post方式：会乱码
  - 解决：在获取参数前，设置request的编码`request.setCharacterEncoding("utf-8");`

## 4.4 表单路径写法

login.html中form表单的action路径的写法

虚拟目录+Servlet的资源路径

如： /blog/login.do    ==》 项目名为blog的login.do  Servlet

## 4.5 BeanUtils工具类

- setProperty()
- getProperty()
- populate(Object obj , Map map):将map集合的键值对信息，封装到对应的JavaBean对象中

自行看文档

----

## 4.6 Response功能

> response对象由web容器创建。web容器接收到HTTP请求，自动创建响应对象，web容器完成HTTP响应，客户端接收完响应后自动销毁对象。

总结：

- 服务器接收到HTTP请求自动创建
- 服务器向客户端完成HTTP响应，客户端接收完响应后自动销毁。

**response是用来设置响应消息**

### 4.6.1 设置响应行

- 格式：HTTP/1.1 200 OK
- 设置状态码：setStatus(int sc);

### 4.6.2 设置响应头

- setHeader(String name, String value);

### 4.6.3 设置响应体

- 获取输出流

  - 字符输出流：PrintWriter getWriter()
  - 字节输出流：ServletOutputStream getOutputStream()

  ```java
  PrintWriter writer = resp.getWriter();
  ServletOutputStream os = resp.getOutputStream();
  ```

- 使用输出流，将数据输出到客户端

### 4.6.4 重定向

> **资源跳转的方式，可以用于服务器与服务器之间。浏览器的URL会改变。**

告诉浏览器重定向：状态码302

告诉浏览器资源的路径：响应头   ("location","资源的路径")

```java
response.setStatus(302);
response.setHeader("location","/blog/responseDemo.do");
```

简单的重定向

```java
response.sendRedirect("/blog/xx.do")
```

### 4.6.5 重定向和转发的区别

- 重定向的特点:redirect
  - 地址栏发生变化
  - 重定向可以访问其他站点(服务器)的资源
  - 重定向是两次请求。不能使用request对象来共享数据
- 转发的特点：forward
  - 转发地址栏路径不变
  - 转发只能访问当前服务器下的资源
  - 转发是一次请求，可以使用request对象来共享数据

### 4.6.6 路径的写法

- **相对路径**：通过相对路径不可以确定唯一资源

  - 如：./index.html
  - 不以/开头，以.开头路径

- 规则：找到当前资源和目标资源之间的相对位置关系

  - **./：当前目录**

    ```html
    当前资源：http://localhost:8080/day15/location.html
    目标资源：
    http://localhost:8080/day15/response
    当前资源和目标资源同级目录
    <a href="./response"></a>
    <a href="response"></a>
    ./可以省略
    ```

  - **../:后退一级目录**

- **绝对路径：通过绝对路径可以确定唯一资源**

  - 如：http://localhost/day15/responseDemo2	
  - 以/开头的路径：	/day15/responseDemo2

- 规则：判断定义的路径是给谁用的？判断请求将来从哪儿发出

  - 给客户端浏览器使用：需要加虚拟目录(项目的访问路径)
    - <a> , <form> 重定向...
    - 就是浏览器向服务请求，需要加项目名【虚拟目录】
  - 给服务器使用：不需要加虚拟目录
    - 转发路径。转发是项目内部资源的访问。
  - **为什么？**
    - 同一个项目之间，不会有重复的路径名称。
    - 不同项目之间可能会有重复的路径名称。
    - 怎么办？写上项目名进行区分！

----

### 4.6.7 Response输出

> **response获取地流在一次响应后会自动关闭流，销毁对象。**

> response输出流不刷新也可以把数据写出到浏览器

**乱码问题**

response流是我们获取出来的，不是new出来的。如果是new出来的，编码是和当前操作系统一致的。但是现在的流是tomcat提供的，和tomcat中配置的编码是一样的。tomcat默认是IOS-8859-1。

在获取流对象之前设置编码，让流以这个编码进行。

> **response.setCharacterEncoding("utf-8");**

告诉浏览器，服务器发送的消息体数据的编码，建议浏览器使用该编码进行解码。【这个建议了，浏览器就会照做】

response.setHeader("content-type","text/html;character=utf-8");

其实写了上面那句，就不用写response.setCharacterEncoding("utf-8");了。

简单的形式，设置编码，是在获取流之前设置

**response.setContentType("text/html;charset=utf-8");**

```java
// 解决乱码的代码
response.setContentType("text/html;charset=utf-8");
response.getWriter().write("你好");
```

----

# 五、ServletContext对象

> **代表整个web应用，可以和程序的容器(服务器)来通信**

> ServletContext官方叫servlet上下文。服务器会为每一个工程创建一个对象，这个对象就是ServletContext对象。这个对象全局唯一，而且工程内部的所有servlet都共享这个对象。所以叫全局应用程序共享对象。

## 5.1 ServletContext的获取

- 通过request对象获取  request.getServletContext();
- 通过HttpServlet获取  this.getServletContext();

## 5.2 ServletContext的功能

- 获取MIME类型：

  - MIME类型:在互联网通信过程中定义的一种文件数据类型
  - 格式： 大类型/小类型   text/html		image/jpeg

- 域对象：共享数据

  - setAttribute(String name,Object value)
  - getAttribute(String name)
  - removeAttribute(String name)

- 获取文件的真实(服务器)路径

  - String getRealPath(String path)  

  - ```java
    String b = context.getRealPath("/b.txt");//web目录下资源访问
    System.out.println(b);
    
    String c = context.getRealPath("/WEB-INF/c.txt");//WEB-INF目录下的资源访问
    System.out.println(c);
    
    String a = context.getRealPath("/WEB-INF/classes/a.txt");//src目录下的资源访问
    System.out.println(a);
    ```

## 5.3 文件的下载

> 需求：点击连接，进行下载

注意：任何文件都要是下载，不能让它被浏览器解析！

```java
public class DownLoadUtils {

    public static String getFileName(String agent, String filename) throws UnsupportedEncodingException {
        if (agent.contains("MSIE")) {
            // IE浏览器
            filename = URLEncoder.encode(filename, "utf-8");
            filename = filename.replace("+", " ");
        } else if (agent.contains("Firefox")) {
            // 火狐浏览器
            BASE64Encoder base64Encoder = new BASE64Encoder();
            filename = "=?utf-8?B?" + base64Encoder.encode(filename.getBytes("utf-8")) + "?=";
        } else {
            // 其它浏览器
            filename = URLEncoder.encode(filename, "utf-8");
        }
        return filename;
    }
}



@WebServlet("/downloadServlet")
public class DownloadServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        //1.获取请求参数，文件名称
        String filename = request.getParameter("filename");
        //2.使用字节输入流加载文件进内存
        //2.1找到文件服务器路径
        ServletContext servletContext = this.getServletContext();
        String realPath = servletContext.getRealPath("/img/" + filename);
        //2.2用字节流关联
        FileInputStream fis = new FileInputStream(realPath);

        //3.设置response的响应头
        //3.1设置响应头类型：content-type
        String mimeType = servletContext.getMimeType(filename);//获取文件的mime类型
        response.setHeader("content-type",mimeType);
        //3.2设置响应头打开方式:content-disposition

        //解决中文文件名问题
        //1.获取user-agent请求头、
        String agent = request.getHeader("user-agent");
        //2.使用工具类方法编码文件名即可
        filename = DownLoadUtils.getFileName(agent, filename);

        response.setHeader("content-disposition","attachment;filename="+filename);
        //4.将输入流的数据写出到输出流中
        ServletOutputStream sos = response.getOutputStream();
        byte[] buff = new byte[1024 * 8];
        int len = 0;
        while((len = fis.read(buff)) != -1){
            sos.write(buff,0,len);
        }

        fis.close();


    }

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        this.doPost(request,response);
    }
}

```

### 5.3.1 中文乱码解决思路

- 获取客户端使用的浏览器版本信息
- 根据不同的版本信息，设置filename的编码方式不同

```java
public class DownLoadUtils {

    public static String getFileName(String agent, String filename) throws UnsupportedEncodingException {
        if (agent.contains("MSIE")) {
            // IE浏览器
            filename = URLEncoder.encode(filename, "utf-8");
            filename = filename.replace("+", " ");
        } else if (agent.contains("Firefox")) {
            // 火狐浏览器
            BASE64Encoder base64Encoder = new BASE64Encoder();
            filename = "=?utf-8?B?" + base64Encoder.encode(filename.getBytes("utf-8")) + "?=";
        } else {
            // 其它浏览器
            filename = URLEncoder.encode(filename, "utf-8");
        }
        return filename;
    }
}
```



----

# 六、Cookie&Session

## 6.1 会话技术

> **会话：一次会话中包含多次请求和响应。**

> 一次会话：浏览器第一次给服务器资源发送请求，会话建立，直到有一方断开【浏览器关闭了，服务器关掉了】为止

客户端会话技术：Cookie

服务器端会话技术：Session

## 6.2 会话跟踪

> **在用户访问的一个会话内，web服务器保存客户的学校，称为会话跟踪。**

## 6.3 会话跟踪方式

- **重写URL**
  - 就是URL地址后面附加参数 
  - `<a href="xxx/xx/xxx?name=zs">`
  - 其缺点为：URL地址过长，不同浏览器对URL传递参数的限制，安全性差【参数明码传输】，编程繁杂。
- **隐藏表单字段**
  - 将会话数据放在隐藏域表单元素。简而言之就是form表单传递数据。
  - 其缺点为：安全性差，可以通过查看网页源代码发现保存的会话信息；编程复杂，如果要保存的数据很多，就很麻烦；无法在超链接模式下工作。
- **Cookie**
  - web服务器保存在客户端的小的文本文件。存储了许多key，value对。
  - Cookie由服务器创建，存储在客户端中。
  - **PS：高版本tomcat中Cookie不能存储空格**
  - 缺点：
    - 存储方式单一，只能存储String类型【Cookie(String name,String name)】
    - 存储位置限制，若存储的Cookie过多，每次请求都发送Cookie，网络数据量过大，影响Web应用性能。
    - Cookie大小受浏览器限制
    - Cookie存储在客户端，若客户端禁用Cookie则无效。
- **HTTPSession对象API**
  - 

## 6.4 Cookie的使用

### 6.4.1 基本使用

- 创建 == new Cookie(String name, String value) 
- 发送==response.addCookie(Cookie cookie) 
- 获取==Cookie[]  request.getCookies()  

## 6.4.2 Cookie的生命周期

- 默认，关闭浏览器就消失了
- **可进行设置【持久化存储】**
  - `setMaxAge(int seconds) == cookie.setMaxAge(60 * 60 * 24 * 30);//一个月`
  - 正数：将Cookie数据写到硬盘的文件中。持久化存储。并指定cookie存活时间，时间到后，cookie文件自动失效
  - 负数：默认值
  - 零：删除cookie信息

### 6.4.3 Cookie细节

> **高版本tomcat的cookie不能有空格**

- 一次可不可以发送多个cookie?
  - 可以
- **cookie能不能存中文？**
  - 在tomcat 8 之前 cookie中不能直接存储中文数据。
    - 需要将中文数据转码---一般采用URL编码(%E3)
  - 在tomcat 8 之后，cookie支持中文数据。特殊字符还是不支持，建议使用URL编码存储，URL解码解析

- **cookie共享问题？**
  - 假设在一个tomcat服务器中，部署了多个web项目，那么在这些web项目中cookie能不能共享？
    - 默认情况下cookie不能共享
  - setPath(String path):设置cookie的获取范围。默认情况下，设置当前的虚拟目录
    - 如果要共享，则可以将path设置为"/"
    - setPath("/") //当前服务器的根目录
    - setPath("/day") //day项目才可以访问
- **不同的tomcat服务器间cookie共享问题？**
  - setDomain(String path):如果设置一级域名相同，那么多个服务器之间cookie可以共享
  -  setDomain(".baidu.com"),那么tieba.baidu.com和news.baidu.com中cookie可以共
    - .baidu.com是一级域名
    - tieba是二级域名

## 6.5 Session

> 服务器端会话技术，在一次会话的多次请求间共享数据，将数据保存在服务器端的对象中。HttpSession

### 6.5.1 Session快速入门

- 获取HttpSession对象：HttpSession session = request.getSession();
- 使用HttpSession对象：
  - Object getAttribute(String name)  
  - void setAttribute(String name, Object value)
  - void removeAttribute(String name)
- **原理：Session的实现是依赖于Cookie的。**session对象创建后，它的sessionID会自动选择Cookie作为存储地

### 6.5.2 注意细节

- 当客户端关闭后，服务器不关闭，两次获取session是否为同一个？

  - 默认情况下。不是。

  - 如果需要相同，则可以创建Cookie,键为JSESSIONID，设置最大存活时间，让cookie持久化保存。

  - ```java
    HttpSession session = request.getSession();
    Cookie c = new Cookie("JSESSIONID",session.getId());
    c.setMaxAge(60*60);
    response.addCookie(c);
    /*
    浏览器禁用Cookie时，如何使用Session?
    代码与上面一样，我们手动设置Session写入Cookie中。
    浏览器没用禁用Cookie时，会自动把Session的Id写入Cookie的。
    */
    ```

- 客户端不关闭，服务器关闭后，两次获取的session是同一个吗？

  - 不是同一个，但是要确保数据不丢失。**【tomcat自动完成以下工作】**
    - session的钝化：在服务器正常关闭之前，将session对象系列化到硬盘上
    - session的活化：在服务器启动后，将session文件转化为内存中的session对象即可。

- session什么时候被销毁？

  - 服务器关闭

  - session对象调用invalidate() 

  - session默认失效时间 30分钟【tomcat的web.xml配置文件中】

    ```xml
    <session-config>
        <session-timeout>30</session-timeout>
    </session-config>
    ```

### 6.5.3 特点

- session用于存储一次会话的多次请求的数据，存在服务器端

- session可以存储任意类型，任意大小的数据

- session与Cookie的区别：
  - session存储数据在服务器端，Cookie在客户端
  - session没有数据大小限制，Cookie有
  - session数据安全，Cookie相对于不安全

----

# 七、Filter

> web中的过滤器：当访问服务器的资源时，过滤器可以将请求拦截下来，完成一些特殊的功能。

## 7.1 应用场景

- 登录校验
- 设置统一编码

## 7.2 入门代码

- xml配置
- 注解配置

```java
// 访问所有资源之前都会执行该过滤器。都会被这个过滤器拦截下来
@WebFilter("/*")
public class FilterDemo implements Filter {
    public void destroy() {
    }

    public void doFilter(ServletRequest req, ServletResponse resp, FilterChain chain) throws ServletException, IOException {
        System.out.println("hello");
        // 放行
        chain.doFilter(req, resp);
    }

    public void init(FilterConfig config) throws ServletException {

    }
}

```

## 7.3 过滤器细节

### 7.3.1 xml&注解配置

```xml
<filter>
    <filter-name>demo1</filter-name>
    <filter-class>cn.itcast.web.filter.FilterDemo1</filter-class>
    <!-- 初始化参数配置 -->
    <init-param>
    	<param-name>key</param-name>
        <param-value>xxxx</param-value>
    </init-param>
    <init-param>
        <param-name>key</param-name>
        <param-value>xxxx</param-value>
    </init-param>
</filter>
<filter-mapping>
    <filter-name>demo1</filter-name>
    <!-- 拦截路径 -->
    <url-pattern>/*</url-pattern>
</filter-mapping>
```

```java
@WebFilter("/*")
@WebInitParam(name="liu",value = "sd")
public class FilterDemo implements Filter {
	///xxxjava
}
```

### 7.3.2 过滤器执行流程

- 执行过滤器
- 执行放行后的资源
- 回来执行过滤器放行代码下边的代码

### 7.3.3 生命周期方法

- init:在服务器启动后，会创建Filter对象，然后调用init方法。只执行一次。用于加载资源
- doFilter:每一次请求被拦截资源时，会执行。执行多次
- destroy:在服务器关闭后，Filter对象被销毁。如果服务器是正常关闭，则会执行destroy方法。只执行一次。用于释放资源

### 7.3.4 配置详解

- 拦截路径配置
  - 具体资源路径： /index.jsp   只有访问index.jsp资源时，过滤器才会被执行
  - 拦截目录： /user/*	访问/user下的所有资源时，过滤器都会被执行
  - 后缀名拦截： *.jsp		访问所有后缀名为jsp资源时，过滤器都会被执行
  - 拦截所有资源：/*		访问所有资源时，过滤器都会被执行
- 拦截方式配置：资源被访问的方式
  - 注解配置`@WebFilter(value = "/index.jsp",dispatcherTypes = DispatcherType.REQUEST)`
    - REQUEST：默认值。浏览器直接请求资源
    - FORWARD：转发访问资源
    - INCLUDE：包含访问资源
    - ERROR：错误跳转资源
    - `@WebFilter(value = "/index.jsp",dispatcherTypes = DispatcherType.FORWARD)`
    - **当xx转发到index.jsp时，会被过滤器拦截**
  - web.xml配置
    - 设置<dispatcher></dispatcher>标签即可

### 7.3.5 过滤器链(配置多个过滤器)

- 执行顺序：如果有两个过滤器：过滤器1和过滤器2
  - 过滤器1;过滤器2;资源执行;过滤器2;过滤器1
- 过滤器先后顺序问题：
  - 注解配置：按照类名的字符串比较规则比较，值小的先执行；如： AFilter 和 BFilter，AFilter就先执行了。【字典顺序小的先执行】
  - web.xml配置： <filter-mapping>谁定义在上边，谁先执行

----

## 7.4 过滤器案例

### 7.4.1 登录

- 防止用户直接通过url进行页面访问！

  - 获取请求资源路径
  - 判断路径是否为登录相关
    - 是则放行，否则跳转至登录页面
  - 注意，过滤器也会拦截静态资源，对静态资源要进行放行

  ```java
  public void doFilter(ServletRequest req, ServletResponse resp, FilterChain chain) throws ServletException, IOException {
      //0.强制转换
      HttpServletRequest request = (HttpServletRequest) req;
  
      //1.获取资源请求路径
      String uri = request.getRequestURI();
      //2.判断是否包含登录相关资源路径,
      //要注意排除掉 css/js/图片/验证码等资源
      if(uri.contains("/login.jsp") ||
         uri.contains("/loginServlet") ||
         uri.contains("/css/") ||
         uri.contains("/js/") ||
         uri.contains("/fonts/") ||
         uri.contains("/checkCodeServlet")  
        ){
          //包含，用户就是想登录。放行
          chain.doFilter(req, resp);
      }else{
          //不包含，需要验证用户是否登录
          //3.从获取session中获取user
          Object user = request.getSession().getAttribute("user");
          if(user != null){
              //登录了。放行
              chain.doFilter(req, resp);
          }else{
              //没有登录。跳转登录页面
              request.setAttribute("login_msg","您尚未登录，请登录");      request.getRequestDispatcher("/login.jsp").forward(request,resp);
          }
      }
  ```

  

### 7.4.2 过滤敏感词

- 需要对request的获取方法进行增强！

```java
/**
 * 敏感词汇过滤器
 */
@WebFilter("/*")
public class SensitiveWordsFilter implements Filter {

    public void doFilter(ServletRequest req, ServletResponse resp, FilterChain chain) {
        //1.创建代理对象，增强getParameter方法
        ServletRequest proxy_req = (ServletRequest) Proxy.newProxyInstance(req.getClass().getClassLoader(), req.getClass().getInterfaces(), new InvocationHandler() {
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                //增强getParameter方法
                //判断是否是getParameter方法
                if(method.getName().equals("getParameter")){
                    //增强返回值
                    //获取返回值
                    String value = (String) method.invoke(req,args);
                    if(value != null){
                        for (String str : list) {
                            if(value.contains(str)){
                                value = value.replaceAll(str,"***");
                            }
                        }
                    }
                    return  value;
                }
                //判断方法名是否是 getParameterMap
                //判断方法名是否是 getParameterValue
                return method.invoke(req,args);
            }
        });

        //2.放行
        chain.doFilter(proxy_req, resp);
    }
    private List<String> list = new ArrayList<String>();//敏感词汇集合
    public void init(FilterConfig config) throws ServletException {

        try{
            //1.获取文件真实路径
            ServletContext servletContext = config.getServletContext();
            String realPath = servletContext.getRealPath("/WEB-INF/classes/敏感词汇.txt");
            //2.读取文件
            BufferedReader br = new BufferedReader(new FileReader(realPath));
            //3.将文件的每一行数据添加到list中
            String line = null;
            while((line = br.readLine())!=null){
                list.add(line);
            }

            br.close();

            System.out.println(list);

        }catch (Exception e){
            e.printStackTrace();
        }
    }
}
```



## 7.5 增强对象的功能

## 7.5.1 装饰模式

### 7.5.2 代理模式

> 分为静态代理和动态代理

- 概念
  - 真实对象：被代理的对象
  - 代理对象
  - 代理模式：代理对象代理真实对象，达到增强真实对象功能的目的。

> **实现方式**

1.静态代理：有一个类文件描述代理模式

2.动态代理：在内存中形成代理类

> **动态代理实现步骤**

- 代理对象和真实对象实现相同的接口
- 代理对象 = Proxy.netProxyInstance();

- 使用代理对象调用方法【实现了相同接口，有相同的方法设定】
- 增强方法
- 增强返回值，如：return "name"，增强后 return “name”+“：price”;
- 增强参数，对参数进行修改，如：售价打折。

```java
public interface ISale {
    public String sale(double money);

    public void show();
}

public class Lenovo implements ISale {
    @Override
    public String sale(double money) {
        System.out.println("花了"+money+"买了一台电脑");
        return "Lenovo";
    }

    @Override
    public void show() {
        System.out.println("展示电脑");
    }
}

public class ProxyDemo {

    public static void main(String[] args) {
        Lenovo lenovo = new Lenovo();
        /**
         * 1.类加载器
         * 2.接口数组，保证代理对象和真实对象实现相同的接口
         * 3.处理器
         */
        Class<Lenovo> lc = Lenovo.class;
        ISale o = (ISale) Proxy.newProxyInstance(lc.getClassLoader(), lc.getInterfaces(), new InvocationHandler() {
            /**
             * 代理逻辑方法
             * @param proxy 代理对象 其实指的就是o
             * @param method 代理对象调用的方法，被封装为的对象
             * @param args
             * @return
             * @throws Throwable
             */
            @Override
            public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
                Object obj = method.invoke(lenovo, args);
                if("sale".equals(method.getName())){
                    // 调用真实对象的方法 obj是执行方法后的返回值
                    System.out.println("该方法执行了");
                }
                return obj;
            }
        });
        String sale = o.sale(1000);
        o.show();
        System.out.println(sale);
    }
}
/**
lambda写法
Proxy.newProxyInstance(lc.getClassLoader(),lc.getInterfaces(),(proxy,method,ags)->{
            if("sale".equals(method)){
                // 调用真实对象的方法 obj是执行方法后的返回值
                System.out.println("该方法执行了");
                Object invoke = method.invoke(lc, ags);
                return invoke;
            }
            return method.invoke(lc,ags);
        });
*/
```



----

# 八、Listener

> **概念：web的三大组件之一。**

> **事件监听机制**

- 事件	：一件事情
- 事件源 ：事件发生的地方
- 监听器 ：一个对象
- 注册监听：将事件、事件源、监听器代码

## 8.1 ServletContextListener

> 监听ServletContext对象的创建和销毁。

```java
// 对象创建后调用该方法
default void contextInitialized(ServletContextEvent sce) {
}
// 对象销毁前，调用该方法
default void contextDestroyed(ServletContextEvent sce) {
}
```

> 步骤

- 定义一个类，实现ServletContextListener接口

- 覆写方法

- 配置

  - web.xml配置方式

    ```xml
    <listener>
    	<listener-class>实现类的全路径</listener-class>
    </listener>
    <!-- 指定初始化参数 -->
    <context-param>
    	<param-name>key</param-name>
        <param-value>value</param-value>
    </context-param>
    <context-param>
        <param-name>123</param-name>
        <param-value>234</param-value>
    </context-param>
    ```

  - 注解配置方式

    ```java
    @WebListener
    public class ListenerDemo implements ServletContextListener{
        //xxx
    }
    ```

    

> 作用？

- 用来加载资源文件

```java
public void contextInitialized(ServletContextEvent sce) {
	servletContext.getInitParameter("");
}
```



# 九、MVC&三层架构

## 9.1 MVC

M：Model，模型，进行业务逻辑操作【JavaBean】

- 如：查询数据库，封装对象

V：View，视图，展示数据。【JSP】

C：Controller，控制器，不做具体业务操作，只是中转。【Servlet】

- 获取用户的输入
- 调用模型
- 将数据交给视图进行展示

请求资源时经过控制器，控制器去调用模型进行业务操作，操纵后模型把数据返回给控制器。控制器再把数据给试图去展示。



## 9.2 三层架构

1. 界面层(表示层)：用户看的得界面。用户可以通过界面上的组件和服务器进行交互
2. 业务逻辑层：处理业务逻辑的。
3. 数据访问层：操作数据存储文件。

# 10、Ajax和JSON

## 10.1 Ajax

> **概念：Asynchronous JavaScript And XML 异步的JavaScript和xml。**【这里说的同步异步与线程关系不大】

----

### 10.1.1 原生JS实现

- 原生JS实现方式==快速入门==

```html
<script>
    var button = document.getElementById("ss");
    button.onclick = function() {
        console.log(123)
        // 发送异步请求
        // 1.创建核心对象
        var xmlhttp;
        if (window.XMLHttpRequest) {
            xmlhttp = new XMLHttpRequest();
        } else {
            xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
        }
        // 2.发送请求
        /**
		 * 1.请求方式：GET POST
		 *    get：参数拼接在URL上
		 *    post：参数在send方法中定义
		 */
        xmlhttp.open("GET", "demo.json", true);
        xmlhttp.onreadystatechange = function() {
            if(xmlhttp.readyState==4 && xmlhttp.status==200){
                console.log(xmlhttp.responseText);
            }
        }
        // 发送请求
        xmlhttp.send();
    }
</script>
```

### 10.1.2 JQuery实现

> $.ajax实现方式

- 语法：`$.ajax({键值对})`
  - 具体参数查API
- 语法：$.get(url, [data], [callback], [type])
- 语法：$.post(url, [data], [callback], [type])
  - url：请求路径
  - data：请求参数
  - callback：回调函数
  - type：响应结果的类型

```js
// $.ajax方式
$("#button2").click(function(){
    $.ajax({
        url:"demo.json",
        type:"GET",
        data:{"name":"liujiawei"},
        timeout:10000,
        success:function(data){
            console.log(data);
        },
        error:function(){

        },
        dataType:"json"
    });
});

// $.get方式
$.get("demo.json", {
    "name": "1233"
}, function(data) {
    console.log(data);
}, "json");

// $.post方式
$.post("demo.json", {
    "name": "1233"
}, function(data) {
    console.log(data);
}, "json");
```

## 10.2 JSON

> **JSON是JS对象的字符串表示法，它使用文本表示一个JS对象信息，本质是一个字符串！**

### 10.2.1 JSON的基本用法

- 对象表示为键值对
- 数据由逗号分隔
- 花括号保存对象
- 方括号保存数组

- JSON与JS对象互转

  - JSON字符串转为JS对象，使用JSON.parse()

    ```js
    var obj = JSON.parse('{"a","hello"}');
    // 控制台的输出结果是 {a:'hello',b:'world'}
    ```

  - JS对象转换为JSON字符串,使用JSON.stringify()

    ```js
    var json = JSON.stringifu({a:'hello',b:'world'}); 
    // 控制台的输出结果是 '{"a":"hello","b":"world"}
    ```

- 代码案例

  ```js
  <script>
  // 1.常规JSON字符串
  var json1 = {
       "name": "liujiawei",
       "age": 18
  };
  // 控制台输出 object
  console.log(typeof(json1));
  // 控制台输出{name: "liujiawei", age: 18}
  console.log(json1);
  
  var json3 = '{"name":"liujiawei","age":18}';
  // 控制台输出 string
  console.log(typeof(json3));
  // 控制台输出{"name":"liujiawei","age":18}
  console.log(json3);
  
  //2.带数组
  var json4 = {
      "name": "liujiawei",
      "age": 18,
      "array": [1, 2, 3, 4, 5]
  };
  console.log(json4.array[0]);
  console.log(json4 === eval(json4)); // true
  
  //3.复合
  var json5 = {
      "name": "liujiawei",
      "age": 18,
      "array": [1, 2, 3, 4, 5],
      "data": {
          "key1": "value1",
          "key2": "value2",
          "key3": "value3"
      }
  }
  console.log(json5.data.key1);
  /**
  * JSON数据的遍历
  */
  var person = {"name": "张三",age: 23,'gender': true};
  
  var ps = [
      {"name": "张三","age": 23,"gender": true},
      {"name": "李四","age": 24,"gender": true},
      {"name": "王五","age": 25,"gender": false}
  ];
  console.log("**************")
  for (var key in person) {
      // string
      console.log(typeof(key));
      // 相当于 person["name"] 
      // 不过person.key是不行的. 相当于person."name"
      console.log(person[key]);
  }
  console.log("**************")
  for (var i = 0, len = ps.length; i < len; i++) {
      var temp = ps[i];
      for (var t in temp) {
          console.log(temp[t])
      }
  }
  console.log("**************")
  // 如果是不规则的数据呢？递归遍历
  var datas = {
      "name": "菜是原罪",
      "age": 18,
      "friends": [
          {"name": "小红","age": 18},
          {"name": "小黑","age": 23},
          {"name": "小蓝","age": 30,"otherPeople": {"name": "bzd","address": "北京","age": 33}},
      ]
  };
  
  // 递归,如果是对象 就继续递归遍历,如果不是对象,则遍历到头
  function travel(data, obj) {
      // 建立一个对象,用来判断遍历是否需要结束
      if (obj == null) obj = new Object();
      // 如果当前遍历到的是对象,则继续递归
      if (typeof(data) == typeof(obj)) {
          for (var i in data) {
              if (typeof(i) != typeof(obj)) console.log(i)
              travel(data[i], obj);
          }
      } 
      // 如果不是对象,则说明递归到头,输出内容,结束递归.
      else {
          console.log(data);
      }
  }
  
  travel(datas);
  </script>
  ```

### 10.2.2 JSON的转换

- JS对象与json格式数据之间的转换

  ```javascript
  var obj = JSON.parse('{"a","hello"}');
  // 控制台的输出结果是 {a:'hello',b:'world'}
  var json = JSON.stringifu({a:'hello',b:'world'}); 
  // 控制台的输出结果是 '{"a":"hello","b":"world"}
  ```

### 10.2.3 JSON与Java对象

- JSON解析器：

  - 常见的解析器：Jsonlib，Gson，fastjson，**jackson**【spring用的】

- **JSON转为Java对象**

  - 导入jackson的相关jar包
  - 创建Jackson核心对象 ObjectMapper
  - 调用ObjectMapper的相关方法进行转换
    - readValue(json字符串数据,Class)

- **Java对象转换JSON**

  - 导入jackson的相关jar包

  - 创建Jackson核心对象 ObjectMapper

  - 调用ObjectMapper的相关方法进行转换

    - writeValue(参数1，obj):

      ```
      参数1：
      File：将obj对象转换为JSON字符串，并保存到指定的文件中
      Writer：将obj对象转换为JSON字符串，并将json数据填充到字符输出流中
      OutputStream：将obj对象转换为JSON字符串，并将json数据填充到字节输出流
      ```

    - writeValueAsString(obj):将对象转为json字符串

- **注解**

  - @JsonIgnore：排除属性。
  - @JsonFormat：属性值得格式化
    - eg：@JsonFormat(pattern = "yyyy-MM-dd")

- 复杂Java对象转换

  - List：数组
  - Map：对象格式一致

```java
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Before;
import org.junit.Test;
import proxy.Person;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class JsonDemo {
    private ObjectMapper obj = null;

    @Before
    public void init() {
        obj = new ObjectMapper();
    }

    @Test
    /**
     * 简单对象转JSON
     */
    public void ObjectToJSON() throws JsonProcessingException {
        //String name, String age, String sex, int weight, String birthday
        Person person = new Person("刘家伟", "18", "nan", 88, "1997-11-11");
        String s = obj.writeValueAsString(person);
        System.out.println(s);
    }

    /**
     * map转JSON
     */
    @Test
    public void MapToJSON() throws JsonProcessingException {
        Map<String,String> map = new HashMap<String,String>();
        map.put("name","tom");
        map.put("friend","jerry");
        System.out.println(obj.writeValueAsString(map));
    }

    /**
     * 复杂map转json
     */
    @Test
    public void ComplexMapToJSON() throws JsonProcessingException {
        Map<String,Person> map = new HashMap<String,Person>();
        map.put("name",new Person("1", "18", "nan", 88, "1997-11-11"));
        map.put("friend",new Person("2", "18", "nan", 88, "1997-11-11"));
        System.out.println(obj.writeValueAsString(map));
    }

    /**
     * list转json
     * [{},{},{}]
     */
    @Test
    public void listToJSON() throws JsonProcessingException {
        Person p1 = new Person("1", "18", "nan", 88, "1997-11-11");
        Person p2 = new Person("2", "18", "nan", 88, "1997-11-11");
        Person p3 = new Person("3", "18", "nan", 88, "1997-11-11");
        ArrayList<Person> list = new ArrayList<>();
        list.add(p1);
        list.add(p2);
        list.add(p3);
        System.out.println(obj.writeValueAsString(list));
    }

    /**
     * 写入文本中
     */
    @Test
    public void writeToFile() throws IOException {
        Person p1 = new Person("1", "18", "nan", 88, "1997-11-11");
        Person p2 = new Person("2", "18", "nan", 88, "1997-11-11");
        Person p3 = new Person("3", "18", "nan", 88, "1997-11-11");
        ArrayList<Person> list = new ArrayList<>();
        list.add(p1);
        list.add(p2);
        list.add(p3);
        obj.writeValue(new File("demo.json"),list);
    }

    /**
     * json转对象
     */
    @Test
    public void jsonToObject() throws JsonProcessingException {
        String str = "{\"name\":\"1\",\"age\":\"18\",\"sex\":\"nan\",\"weight\":88,\"birthday\":\"1997-11-11\"}";
        Person person = obj.readValue(str, Person.class);
        System.out.println(person.toString());
    }
}

```



