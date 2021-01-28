# SpringMVC---概述

不太适合如门用哦！

我看到雷丰阳的SpringMVC视频，17年的。但是我用的是JavaConfig风格的配置。

# SpringMVC---基本原理

## 运行流程

 * 1）客户端点击链接发送 xxx/ 请求
 * 2）来到tomcat服务器
 * 3）SpringMVC的前端控制器收到所有请求
 * 4）来看请求地址和@RequestMapping标注的那个匹配，来找到到底使用那个类的那个方法来处理请求。
 * 5）前端控制器找到了目标处理器类和目标方法，直接利用 返回执行目标方法
 * 6）方法执行完成后会有一个返回值，SpringMVC认为这个返回值就是要去的页面地址
 * 7）拿到方法返回值后；用视图解析器进行拼串得到完整的页面地址
 * 8）拿到页面地址值，前端控制器帮我们转发到页面。

## @RequestMapping基本概念

 * 告诉spring mvc这个方法用来处理什么请求。。
 * 这个/是可以省略的，即时省略了，也是默认从当前项目开始。
 * 加上/比较好

## 前端控制器的拦截规则

### tomcat的拦截规则

<img src="../pics/SpringMVC/DispatcherServlet.png" style="float:left">

在使用tomcat的基本api进行开发时，资源的拦截规则，默认用的是tomcat中web.xml中的配置。

```xml
<!-- The mapping for the default servlet -->
<!-- 这里是静态资源的拦截。tomcat的DefaultServlet拦截发现是静态资源后，就回去找对应的静态资源并返回 -->
<servlet-mapping>
    <servlet-name>default</servlet-name>
    <url-pattern>/</url-pattern>
</servlet-mapping>

<!-- 这里是jsp的拦截，找到对应的jsp后就放回 -->
<servlet-mapping>
    <servlet-name>jsp</servlet-name>
    <url-pattern>*.jsp</url-pattern>
    <url-pattern>*.jspx</url-pattern>
</servlet-mapping>
```

### 前端控制器的拦截规则

前端控制器的拦截规则相当于继承自tomcat的那个web.xml的配置，并重写了拦截方式。

 *  <span style="color:green">**DefaultServlet是tomcat处理静态资源的**</span>
     *  除jsp和servlet，其他的都是静态资源；index.html也是静态资源；如果静态资源让tomcat来处理的话，tomcat就会在服务器下找到这个资源并返回。
     *  所以DefaultServlet有效的情况下，index.html才有用
 *  <span style="color:green">**tomcat有配置拦截规则，前端控制器也有，前端控制器相当于子类，重写了拦截规则！**</span>
     *  相当于前端控制器的 / 把tomcat的DefaultServlet禁用掉了。请求的资源被前端控制器拦截了！
     *  请求来到前端控制器，前端控制器看那个方法的RequestMapping的路径是这个。最后发现没有那个方法的RequestMapping路径是index.html；没有！所有无法访问！找资源的方式都错了！！静态资源访问就无效了！！
 *  <span style="color:green">**为什么jsp又能访问？**</span>
     *  因为我们没有覆盖tomcat服务器中的JspServlet的配置，即Jsp的请求不由前端控制器处理，由tomcat自己处理。
     *  如果我们把拦截方式改成 `/*`那么*.Jsp的请求也会经过前端控制器，也有从RequestMapping中找对应的方法，
 *  <span style="color:green">**配置说明**</span>
     *  / 相当于把tomcat中的大web.xml的DefaultServlet重写了（静态资源拦截那个）
     *  /* 直接是拦截所有请求。所以我们写  / ,写 / 也是为了迎合rest风格的url地址
     *  springmvc是先经过前端控制器的，看有没有配对的，没有就报错。

# SpringMVC---常用注解

## @RequestMapping

> <span style="color:green">**@RequestMapping的使用**</span>

Spring MVC使用@RequestMapping注解为控制器指定可以处理那些url请求。

 * 在控制器的类定义及方法定义处都可标准
    * 类定义处：提供初步的请求映射信息。相对于WEB应用的根目录
    * 方法处：提供进一步的细分映射信息。相当于类定义处的URL。
    * 举例 WEB根路径为 localhost:8080/SpringMVC/
       * 类定义处路径为 /user
       * 方法定义处路径为  /add
       * 则该方法的访问路径为  localhost:8080/SpringMVC/user/add
    * DispatcherServlet 截断请求后，就通过控制器上@RequestMapping提供的映射信息确定请求所对应的处理方法。
 * 映射
    * 请求参数
    * 请求方法
    * 请求头

> <span style="color:green">**@RequestMapping--method**</span>

**指定那些请求方式是有效的。默认是所有请求都有效！**

```java
public enum RequestMethod {
	GET, HEAD, POST, PUT, PATCH, DELETE, OPTIONS, TRACE
}
```

**示例代码**

```java
@RestController
@RequestMapping("/method")
public class RequestMappingController {

    @RequestMapping(path = {"/get"}, method = RequestMethod.GET)
    public String t1() {
        return "GET Method";
    }

    // 方法类型对不上会报错 405 方法不对应！
    @RequestMapping(path = {"/post"}, method = RequestMethod.POST)
    public String t2() {
        return "POST Method";
    }

    @RequestMapping(path = {"/get&post"}, method = {RequestMethod.POST, RequestMethod.GET})
    public String t3(Model model, HttpServletRequest request) {
        String method = request.getMethod();
        return "support GET and POST; current  method is " + method;
    }

    @RequestMapping(path = {"/all"}, method = {RequestMethod.POST, RequestMethod.GET})
    public String t4(HttpServletRequest request) {
        String method = request.getMethod();
        return method;
    }
}
```

----

> <span style="color:green">**@RequestMapping--params**</span>

**用于设置请求要带什么参数过来、不能带什么参数过来、参数的值可以是什么、参数的值不能是什么。**

- params={"username"} 参数中必须要有username！！
- params={"!username"} 参数中不能有username！！
- params={"username!=123"} 参数的值不能为123！！
- params={"username=va"} 参数的值必须为va！！
- params={"user","pwd"} 要有user和pwd两个参数！！
- **<span style="color:red">不能用</span>{"age>19"}这种比较大小的写法！！！！**

示例代码

```java
package cn.payphone.controller;

import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/params")
public class RequestMappingParamsController {

    // 必须携带username这个参数
    // get请求，路径为 http://localhost:8080/SpringMVC01_war_exploded/params/need1?username
    @RequestMapping(path = "/need1", params = {"username"})
    public String t1() {
        return "username is ok";
    }

    // 不能带username这个参数
    @RequestMapping(path = "/need2", params = {"!username"})
    public String t2() {
        return "Not username params";
    }

    // 不能带username这个参数
    @RequestMapping(path = "/need3", params = {"username!=123"})
    public String t3() {
        return "username can't equals 123";
    }

    // username要为3 注意是一个 ”=“ 具体用法不记得就看源码注释！
    @RequestMapping(path = "/need4", params = {"username=123"})
    public String t4() {
        return "username equals 123";
    }
}
```

----

> <span style="color:green">**@RequestMapping--headers**</span>

**规定请求头**，也可以写简单的表达式

请求头中的任意字段都可规定！

```java
@RestController
public class RequestMappingHeaderController {

    /**
     * User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0
     * User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36 Edg/88.0.705.53
     * @return
     */
    // 这样就只有火狐浏览器可以访问了
    @RequestMapping(path = {"/header1"}, headers = {"User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0"})
    public String t1() {
        return "firefox is ok";
    }
}
```

----

## @PathVariable

## Rest风格

# SpringMVC---请求处理

#  SpringMVC---数据输出

# SpringMVC---源码解析

