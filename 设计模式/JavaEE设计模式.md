# JavaEE设计模式

JavaEE设计模式！！JavaEE设计模式！！不是GoF23种设计模式！！

# 表达层体系结构

讨论以下三个设计模式

- 模型-视图-控制器（Model-view-Controller）模式
    - 为整个表达层提供了一种能把状态、表达和行为清晰分离开来的体系结构
- `前端控制器（Front Controller）模式`
    - 阐示了基于请求的环境中如何集中访问。
    - 如Spring MVC的前端控制器模式。
        - 前端控制器负责接收请求，然后决定派发给那个Controller进行处理。
- 装饰器（Decorator）模式
    - 描述了如何将功能动态地添加给一个控制器

## 模型-视图-控制器模式

可以定义表达层的完整场景。

MVC将用户接口问题分割为三个截然不同的部分：模型、视图、控制器。

- 模型：存储应用的状态、
    - 模型存储应用状态。应用状态数据可以存储在JavaBean、数据库... 模型的工作是管理对该状态的访问。
- 视图：解释模型中的数据并将它展示给用户
    - 视图从模型中读数据，用这些数据生成应答。
- 控制器：处理用户的输入、然后或者更新模型或者显示新的视图。
    - 控制器是与请求发生联系的起点。它的工作就是协调处理请求、将用户输入转变为模型更新和视图。

MVC模式的关键在于每个组件都很简单且是自包含的，带有良好定义的接口。因为这些接口，组件才可以独立地改变，因为它们很容易共享和重用。

---

### 数据模型

模型的设计以数据为中心。JavaEE中，模型通常就是JavaBean：无公共变量，访问属性使用setXXX、getXXX方法。

JavaBean可拆成两部分：存取方法、业务方法。

- 存取方法：setXXX、getXXX
- 业务方法：计算JavaBean中数据的值，如销售税率。

----

### 控制器Servlet

控制器的工作

- 读取请求：HttpServletRequest请求
- 协调到模型的访问：
- 存储用于视图的模型信息：request.setAttribute("key","value")
- 将控制转给视图：RequestDispatcher对象转移控制权。

### 视图

视图：写入到HTTP响应的任何东西。可以是servlet、jsp或常规html文件。

视图是没有状态的。每次被调用的时候，必须读取模型数据并把它们转换为HTML页面的格式。

## 前端控制器模式

> 为什么要前端控制器模式？

- 为每个界面都构建一个全新地控制器不仅意味着增加了一些类，同时也使得应用很难扩展。如：想实现一个日志机制，就得在每个单独的控制器中增加日志代码。
- 单独使用一个单独的控制器是，添加公共的功能比较简单，但是该控制器必须包含对应于每个页面的特殊功能，这种情形很不容易扩展。

> 前端控制器可解决的问题

前端控制器（Front Controller）模式主张`构建一个单独的控制器执行每个请求中都需要的公共功能`，而将其他功能委托给与页面相关的控制器。装饰器（Decorator）模式则说明了如何动态地扩充前端控制器。

### 概述

前端控制器提供了一个统一的位置来封装公共请求处理。它的工作相当简单：执行公共的任务，然后把控制器`转移给与页面相关的控制器`。虽然前端控制器处理每一个请求，但是它并不包含处理所有事情所需要的代码。更新模型和选择视图等方面的特定功能都委托给了一个与页面相关的控制器。

页面控制器最终选择正确的动作和视图。页面控制器通常都是按照“四人小组”提出的命令（Command）模式实现为许多个更简单的类。

使用前端控制器会产生一定的开销。因为它的代码是在每个请求中执行的，但是并没有要求其实现必须尽可能高效。

### 前端控制器servlet

> 一般性写法

前端控制器一般用servlet来实现。

此处将前端控制器将被实现为一个请求截获servlet。

- 前端控制器 url映射为`/page/*`，是为了避免拦截所有请求。如果拦截了所有请求，那么forward请求这些也会被拦截，会造成无限的forward。递归，一层套一层。
- 此处设置了一个前缀 `/项目名/page/`，`req.getRequestURI()`后把前缀去除，就是单纯的请求路径了，这样就好处理了。

```java
package org.example;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

/**
 * 前端控制器. 不处理具体的业务，只进行请求的转发。
 * 此处的前端控制器，只进行通用业务的处理，处理完毕后将请求转发，由其他Controller进行处理。
 */
@WebServlet(urlPatterns = {"/page/*"})
public class DispatcherServlet extends HttpServlet {
    private static final String PREFIX = "/Servlet/page/";
    // 假设一直都是登陆用户操作的。
    private static final boolean LOGIN = true;

    @Override
    public void init() throws ServletException {
        System.out.println("初始化资源");
    }

    @Override
    public void destroy() {
        System.out.println("销毁资源");
    }

    @Override
    protected void service(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String method = req.getMethod();
        String requestURI = req.getRequestURI(); // requestURI = /Servlet/page/2
        requestURI = requestURI.replace(PREFIX, "/");
        if (LOGIN) { // 判断用户是否登陆这个通用业务。
            req.getRequestDispatcher(requestURI).forward(req, resp);
        } else {
            resp.sendRedirect(req.getContextPath() + "/index.html");
        }
    }
}
```

> 优雅的写法

一种更为优雅的写法是采用过滤器。

不做记录，请自行写Demo。

## 装饰器模式

把多个小的组件组合成一个大的组件。

装饰器为它的子项“装饰”或者添加一项功能。当装饰器上有一个方法被调用的时候，它先做自身的预处理，然后调用子项上的相应方法。最后产生的是一个扩展的响应。

装饰器只含有一个子项，但是可以形成一条链路，层层装饰。

缺点在于，我们不能假定装饰器之间是按我们想要的方式顺序执行，这种假设一旦某个步骤执行顺序出错或漏掉某个步骤，可能会产生很严重的错误。

优点在于，添加功能时代码改动小，主要的逻辑代码可以再一个新的类中书写。

> 前端控制器应用装饰器模式

DispatcherServlet --> LoggerDispatcherServlet

假设DispatcherServlet处理完通用请求后，转发到LoggerDispatcherServlet进行日志处理，最后才跳转到指定Controller。

> 实现一个装饰过滤器

```java
package org.example;

import javax.servlet.*;
import javax.servlet.annotation.WebFilter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Enumeration;

// 拦截所有请求。
@WebFilter(urlPatterns = {"/*"})
public class FilterChain implements Filter {
    private static PrintWriter log = null;

    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        try {
            String realPath = filterConfig.getServletContext().getRealPath("/aa.txt");
            log = new PrintWriter(realPath);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void doFilter(ServletRequest request, ServletResponse response, javax.servlet.FilterChain chain) throws IOException, ServletException {
        log.println("======开始记录日志======");
        Enumeration<String> keys = request.getParameterNames();
        while (keys.hasMoreElements()) {
            log.println(keys.nextElement());
        }
        log.println("======结束记录日志======");
        log.flush();
        chain.doFilter(request, response);
    }

    @Override
    public void destroy() {
    }
}
```

# 高级表达层设计

将表达层细分为更小的、可重用组件的设计模式。

- 服务工作者
    - 将一个控制器细分为可重用的导航和动作。
- 视图助手
    - 将一个视图中的相关功能封装进一个单独的可重用的对象。（<span style="color:green">不理解</span>）
- 复合视图
    - 将视图分割为多个子视图。（MVC中的多个视图解析器）

## 服务工作者模式

### 概述

> 紧耦合的例子

一个大页面中有若干个小页面。其中包含一个导航栏。导航栏中有链接到其中一个小页面的链接。

在其他大页面中，也需要这样的导航栏。但是由于导航栏中有链接到XX页面的链接，导致我们无法重用这个导航栏，只能重新写过。后期如果要做相同的更改，工作量也显得特别大。

> 服务工作者模式简介

去除页面与控制的耦合关系，维持视图-控制器的分离。服务工作者模式可以达到这个目的。

服务工作者模式的目标是维持动作、视图和控制器之间的分离。

- 服务：指的是前端控制器这个处理请求的中心。
- 调度器对象：封装了页面的选择和随后工作者的选择（我感觉就是 <span style="color:green">页面 - 调度对象 - Controller</span>，用一个中间对象进行适配。）

> 服务工作者模式功能

- 控制器提供了每个请求进入的初始点，也使得公共功能的添加更为方便，也可用装饰器模式进行增强。
- 调度器封装了页面的选择：有如下两种模式的调度器
    - 简单调度器：从请求中获取一些参数，根据参数进行动作和视图的挑选。可直接作为前端控制器的一个方法。
    - 复杂调度器：除了挑选动作、视图外，还进行当前页面、用户权限、输入信息有效性等因素。这类调度器一般以一个单独的类的形式实现。

> 服务工作者模式UML图

<img src="..\pics\J2EE_dp\image-20210525173000575.png">

- 控制器提供了每个请求进入的初始点。
- 调度器封装了页面选择。
- 调度器使用一组动作来执行模型的更新，且动作并不负责选择视图。

## 视图助手模式

降低视图特殊性的机制。一个视图助手相当于模型和视图之间的中介，读取特定的业务数据并进行转换。

SpringMVC的ModelAndView这部分的内容算视图助手模式吗？返回值会进行数据转换，`Object-->JSON`

`View <--->Hepler<--->Model`

Helper充当View 和 Model之间的中间，将输出处理成对方需要格式后再进行传输。书中举的例子是自定义标签库（JSP标签）。这快没明白，用另一本书作为补充。

## 复合视图模式

视图中有一些重复的元素。我们希望页面的代码反映高层的组织。希望能够以一种通用的模板来说明页面的结构，然后在每个应用和每个页面的基础上该部内容。

