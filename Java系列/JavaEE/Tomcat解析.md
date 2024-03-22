# 注意

<a href="https://www.cnblogs.com/alimayun/archive/2004/01/13/10611862.html">博主关于tomcat架构的读书笔记，写的很精炼，值得参考！</a>

[tomcat (u19900101.github.io)](https://u19900101.github.io/2021-09-07-tomcat/#1tomcat)

# 简单入门

```java
package com.baobaoxuxu.controller;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebServlet("*.do")
public class Hello extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        System.out.printf("I am doGet");
    }

    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        System.out.printf("I am do Post");
    }

    /**
     * HttpServlet 是 service 方法调用 doGet，doPost 方法，如果我们像这样重写了 service 方法，
     * 就不会执行 doGet 和 doPost方法了
     * @param req
     * @param resp
     * @throws ServletException
     * @throws IOException
     */
    @Override
    protected void service(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        System.out.printf("I will execution method!");
    }
    
    @Override
    public void init() throws ServletException {
        System.out.printf("I am init");
    }
}
```

# 总体架构

从如何设计服务器的角度说整体架构。

## 总体设计

开始 Server 有开始和结束功能，后面发现请求监听和请求处理放在一起扩展性很差，且 tomcat 需要支持不同的协议，这代码混在一起很不友好，于是用面向对象的思想分开了。分成 Connector 和 Container。【连接器和容器】。

- Connector 负责开启 socket 并监听器客户端的请求，返回数据。
- Container 负责处理具体的请求。
- 这样不同的协议可以继承/实现同一个父类，然后拓展自己的类，达到多态的目的。扩展性更强了。

## tomcat破坏委派

tomcat 为什么破坏双亲委派模型

为了保证各个 web 应用独立，互不干扰。不能出现一个应用中加载的类库会影响到另一个应用的情况。

与 JVM 一样的安全性问题。使用单独的 classloader 去装载 tomcat 自身的类库，以免其他恶意破坏或无意的破坏。

## tomcat的启示

> tomcat 提供了 Bootstrap 作为应用服务器的启动入口。Bootstrap 负责创建 Catalina 实例，根据参数调用 Catalina 相关方法完成针对应用服务器的操作。

为什么 tomcat 不直接用 catalina 启动？而是提供了 Bootstrap?

因为 Bootstrap 依赖 JRE，并通过反射调用 Catalina 实例，与 Tomcat容器是松耦合的。并且为 Tomcat 创建共享类加载器，构造整个 Tomcat 服务器。<span style="color:red">【不是很理解】</span>

Tomcat 的启动方式可以作为非常好的示范来指导中间件产品设计。它实现了启动入口与核心环境的解耦，这样不仅简化了启动（不必配置各种依赖库，因为只有几个独立的 api）且便于我们更灵活的组织中间件产品的结构，尤其是类加载的方案，否则，我们所有依赖库将统一放置到一个类加载器中，而无法做到灵活定制。

## tomcat加载器

`servlet` 规范要求每一个 web 应用都有一个独立的类加载器实例。需要考虑以下特性

- <b>隔离性：</b>可以保证 web 应用库相互隔离。
- <b>灵活性：</b>可以保证移除一个 web 应用不会影响到其他应用。
- <b>性能：</b>每个 web 应用都有一个自己的类加载器，这样它们就不会搜索其他 web 应用的 jar，效率高于只有一个类加载器的情况。

Tomcat 类加载方案

- Bootstrap Class Loader
- Extension Class Loader
- System Class Loader
- Common Class Loader
  - Catalina Class Loader
  - Shared Class Loader
    - Web App1 Class Loader
    - Web App2 Class Loader

Web 应用：以 shared 为父加载器，加载 /WEB-INF/classes 目录下未压缩的 Class 和资源文件以及 /WEB-INF/lib 目录下的 jar 包。该类加载器只对当前 web 应用可见，对其他 web 应用不可见。

