# 一、SpringMVC---概述

不太适合如门用哦！

我看到雷丰阳的SpringMVC视频，17年的。但是我用的是JavaConfig风格的配置。没用视频中的xml配置。

# 二、SpringMVC---基本原理

## 2.1 运行流程

 * 1）客户端点击链接发送 xxx/ 请求
 * 2）来到tomcat服务器
 * 3）SpringMVC的前端控制器收到所有请求
 * 4）来看请求地址和@RequestMapping标注的那个匹配，来找到到底使用那个类的那个方法来处理请求。
 * 5）前端控制器找到了目标处理器类和目标方法，直接利用 返回执行目标方法
 * 6）方法执行完成后会有一个返回值，SpringMVC认为这个返回值就是要去的页面地址
 * 7）拿到方法返回值后；用视图解析器进行拼串得到完整的页面地址
 * 8）拿到页面地址值，前端控制器帮我们转发到页面。

## 2.2 RequestMapping基本概念

- @RequestMapping注解：
     - 告诉spring mvc这个方法用来处理什么请求。。
     - 这个/是可以省略的，即使省略了，也是默认从当前项目开始。
     - 加上/比较好

## 2.3 前端控制器的拦截规则

### 2.3.1 tomcat的拦截规则

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

### 2.3.2 前端控制器的拦截规则

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

# 三、SpringMVC---常用注解

## 3.1 常用注解归纳

- @Controller
- @RequestMapping
- @PathVarible
- @SessionAttribute
- @ModelAttribute

## 3.2 @RequestMapping

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

> <span style="color:green">**@RequestMapping 中的 consumes和produces**</span>

- consumes：只接受内容类型是哪种的请求，规定请求头中的Content-Type
- produces：告诉浏览器返回的内容类型是说明，给响应头中加上Content-Type
    - text/html;charset=utf-8

----

## 3.2 ant风格的URL

**URL地址可以写模糊的通配符**

 * ？ 能替代任意一个字符
 * * 能替代任意多个字符，和一层路径
 * ** 能替代多层路径

```java
package cn.payphone.controller;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 模糊匹配功能
 * URL地址可以写模糊的通配符
 * ？ 能替代任意一个字符
 * * 能替代任意多个字符，和一层路径
 * ** 能替代多层路径
 */
@RestController
@RequestMapping("/ant")
public class AntController {

    @RequestMapping("/antTest01")
    public String antTest1() {
        return "antTest01";
    }

    // antTest01 antTest02 antTest03 都是走这个方法
    // antTest011就不行了，antTest0?中的问号只能匹配一个字符
    // 有精确的肯定优先匹配精确的
    @RequestMapping("/antTest0?")
    public String antTest2() {
        return "antTest?";
    }

    // 先匹配精确一点的antTest0? 在匹配模糊一点的antTest0*
    @RequestMapping("/antTest0*")
    public String antTest3() {
        return "antTest0*";
    }

    // * 匹配一层路径
    @RequestMapping("/a/*/antTest01")
    public String antTest4() {
        return "一层路径";
    }

    // ** 匹配多层路径
    @RequestMapping("/a/**/antTest01")
    public String antTest5() {
        return "两层路径";
    }
}
```

----

## 3.3 @PathVariable

**获取请求路径占位符中的值**

- @PathVariable 获取请求路径中占位符的值
- 占位符的名称和方法中的参数名称一致，就不用在注解里设置占位符的名称
- 占位符的名称和方法中的参数名称不一致，就要在注解里设置占位符的名称

```java
package cn.payphone.controller;

import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class PathVariableController {

    // {id}是占位符
    @RequestMapping("/user/{id}")
    // @PathVariable 获取请求路径中占位符的值
    public String pathVariableTest(@PathVariable("id") String id) {
        return id;
    }

    // 占位符的名称和方法中的参数名称一致就不用在注解里设置别名
    @RequestMapping("/user/info/{id}")
    public String pathVariableTest2(@PathVariable String id) {
        return id;
    }

    // 占位符的名称和方法中的参数名称不一致就要在注解里设置
    @RequestMapping("/user/infos/{id}")
    public String pathVariableTest3(@PathVariable("id") String ids) {
        return ids;
    }
}
```

----

## 3.4 Rest风格

### 3.4.1 概述

Rest--->Representational State Transfer。（资源）表现层状态转化。是目前最流行的一种互联网软件架构。【前段时间提出了一种新的软件架构，是图的】

- 资源（Resource）：网络上的一个实体，或者说是网络上的一个具体信息。
    - URI：统一资源标识符
    - URL：统一资源定位符
- 表现层（Representation）：把资源具体呈现出来的形式，叫做它的表现层。如文本可用txt格式表现，也可用html格式、xml格式、json格式表现。。
- 状态转化（State Transfer）：HTTP协议是无状态的，所有状态都保存在服务器端。所谓的表现层状态转化就是HTTP协议里面，四个表示操作方式的动词：GET、POST、PUT、DELETE。
    - GET：获取资源
    - POST：新建资源
    - PUT：更新资源
    - DELETE：删除资源

### 3.4.2 简单举例

- /book/1 	：GET请求 表示查询1号图书
- /book        ：POST请求 表示添加1号图书
- /book/1     ：PUT请求 表示更新1号图书
- /book/1     ：DELETE 表示删除1号图书

Rest推荐；

<span style="color:green">**url地址这么起名； /资源名/资源标识符**</span>

问题：从页面上只能发起两种请求：GET、POST，其他请求没法使用。

别慌，Spring提供了对Rest风格的支持。

- 1）SpringMVC中有一个Filter，他可以把普通的请求，转化为规定形式的请求。配置Filter。这个Filter叫做，`HiddenHttpMethodFilter`,它的url-pattern写`/*`

- 2）如何发起其他形式的请求？

    - 按照以下要求：

    - 创建post类型的表单

    - 表单项中携带一个`_method`的参数，`_method`的值就是所要的请求形式。

    - ```html
        <form action="book/1" method="post">
            <input name="_method" value="delete">
            <input type="submit" value="删除">
        </form>
        ```

为什么那个Filter可以实现这个功能？？请看源码！

```java
private String methodParam = DEFAULT_METHOD_PARAM;
@Override
protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain filterChain)
    throws ServletException, IOException {

    HttpServletRequest requestToUse = request;
    // 请求方式是POST 且获取的表单参数_method 有值
    if ("POST".equals(request.getMethod()) && request.getAttribute(WebUtils.ERROR_EXCEPTION_ATTRIBUTE) == null) {
        String paramValue = request.getParameter(this.methodParam);
        if (StringUtils.hasLength(paramValue)) {
            String method = paramValue.toUpperCase(Locale.ENGLISH);
            if (ALLOWED_METHODS.contains(method)) {
                // 创建了一个新的request对象
                // 重写了request.getMethod()  获取到的是重写的值
                requestToUse = new HttpMethodRequestWrapper(request, method);
            }
        }
    }
    filterChain.doFilter(requestToUse, response);
}
```

----

# 四、SpringMVC---请求处理

## 4.1 概述

SpringMVC获取请求带来的各种信息

- 入参名称与请求参数名称一致，自动赋值
- @RequestParam
- @RequestHeader
- @CookieValue：获取某个cookie的值
- POJO自动赋值。字段名一致即可。
- 使用Servlet原生API。（session推荐使用原生API）

## 4.2 注解获取请求参数

<span style="color:green">**以下注解都是加载方法的参数上的。**</span>

### 4.2.1 RequestParam

**@RequestParam("user") String username 相当于：**

```java
String username  = request.getPamrameter("user")
// 浏览器传过来一个名为user的形式参数，把user的值存入username的变量中。
```

RequestParam注解的几个重要的值：

* value：指定要获取的参数的key（value和name互为别名。）
* required：这个参数是否必须的
* defaultValue：参数默认值

<span style="color:red">**PS：注意区分RequestParam与PathVarible。**</span>

- RequestParam是获取浏览器传过来的参数，是拿❓后面的值！！
- PathVarible是取的地址中的值！！

### 4.2.2 RequestHeader 

**@RequestHeader 获取请求头中某个key的值。**

request.getHeader("User-Agent")

```java
@RequestHeader("User-Agent") String MyUserAgent 写在方法参数上
等同于 String MyUserAgent = request.getHeader("User-Agent")
```

RequestHeader注解的几个重要的值

- value
- required
- defaultValue

### 4.2.3 CookieValue

**@CookieValue：获取某个cookie的值**

以前获取某个cookie

```java
Cookie[] cookies = request.getCookies();
for (Cookie c: cookies){
	if(c.getName().euqals("JSESSIONID")){
		String ret = c.getValue()
 	}
}
```

现在获取某个cookie

```java
public String index(@CookieValue("JSESSIONID") String jid){
    // pass
}
```

CookieValue注解几个重要的值

* value
* required
* defaultValue

### 4.2.4 SessionAttribute

以前获取Session

```java
request.getSession.getAttribute("user");
```

现在获取Session

```java
public String getSession(@SessionAttribute("user") String user) {
      // pass
}
```

Session还是用原生API获取的好。

## 4.3 POJO自动赋值

<span style="color:red">**注意**</span>

返回对象类型的POJO要映入json库，我用的jackson。

```xml
<dependencies>
    <dependency>
        <groupId>org.springframework</groupId>
        <artifactId>spring-webmvc</artifactId>
        <version>5.3.3</version>
    </dependency>
    <dependency>
        <groupId>javax.servlet.jsp</groupId>
        <artifactId>jsp-api</artifactId>
        <version>2.0</version>
    </dependency>
    <dependency>
        <groupId>javax.servlet</groupId>
        <artifactId>javax.servlet-api</artifactId>
        <version>3.0.1</version>
    </dependency>
    <dependency>
        <!-- https://mvnrepository.com/artifact/org.codehaus.jackson/jackson-core-asl -->
        <groupId>com.fasterxml.jackson.core</groupId>
        <artifactId>jackson-databind</artifactId>
        <version>2.11.4</version>
    </dependency>
</dependencies>
```

```java
@ResponseBody
@RequestMapping("/params/pojo")
public User pojo(User user) {
    return user;
}
```

## 4.4 Servlet原生API

## 4.5 解决提交数据乱码

提交数据可能乱码

### 4.5.1 请求乱码

GET请求：改server.xml 在8080端口处 URIEncoding="UTF-8"

POST 请求：

- 在第一次获取请求参数之前设置，request.setCharacterEncoding("UTF-8")
- 可以自己写一个filter进行过滤：springmvc有这个filter `CharacterEncodingFilter`

印象中可以修改tomcat的默认配置的编码来解决一个乱码问题。具体解决那个忘了。高版本tomcat好像默认是UTF-8的编码格式。

### 4.5.2 响应乱码

response.setContentType("text/html;charset=utf-8")
* 使用SpringMVC前端控制器 写完就直接写字符编码过滤器
* Tomcat一装上，上手就是server.xml的8080处添加URIEncoding=”UTF-8“
* 注意！！字符编码Filter要在其他Filter之前！！！为什么！！因为我们要先设置，让设置生效，之后的操作才有效！！

### 4.5.3 实例配置

Java Config方式的配置！！

**Spring IOC那块的配置**

```java
@Configuration
@ComponentScan(basePackages = "cn.payphone", excludeFilters = {
        @ComponentScan.Filter(type = FilterType.ANNOTATION, classes = {Controller.class})
})
public class RootConfig {
}
```

**SpringMVC IOC的配置**

```java
@EnableWebMvc // 开启mvc的高级配置
@Configuration
@ComponentScan(basePackages = "cn.payphone", includeFilters = {
        @ComponentScan.Filter(type = FilterType.ANNOTATION, classes = {Controller.class})
}, useDefaultFilters = false)
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void configureViewResolvers(ViewResolverRegistry registry) {
        // 这样  视图解析器会自动拼串
        registry.jsp("/WEB-INF/views/", ".jsp");
    }

}
```

**容器相关配置**

```java
public class MyWebServletInitializer extends AbstractAnnotationConfigDispatcherServletInitializer {
    @Override
    protected Class<?>[] getRootConfigClasses() {
        return new Class[]{RootConfig.class};
    }

    @Override
    protected Class<?>[] getServletConfigClasses() {
        return new Class[]{WebConfig.class};
    }

    /**
     * 拦截规则
     * / 拦截所有请求 不拦截jsp页面
     * /* 拦截所有请求 会拦截jsp页面
     * 处理*.jsp请求时tomcat处理的
     *
     * @return
     */
    @Override
    protected String[] getServletMappings() {
        return new String[]{"/"};
    }

    @Override
    protected Filter[] getServletFilters() {
        // 验证字符编码过滤器生效，试验后，真的有效了
        // CharacterEncodingFilter characterEncodingFilter = new CharacterEncodingFilter("ISO-8859-1", true);
        CharacterEncodingFilter characterEncodingFilter = new CharacterEncodingFilter("UTF-8", true);
        characterEncodingFilter.setForceRequestEncoding(true);
        characterEncodingFilter.setForceResponseEncoding(true);
        return new Filter[]{characterEncodingFilter};
    }
}
```



#  五、SpringMVC---数据输出

把数据携带给页面。

前面直接通过响应的方式把数据响应给了浏览器。

但是如果使用的是模板引擎一类的，需要我们携带数据给页面。

## 5.1 Map、Model、ModelMap

1）可以在方法处传入Map、或Model或者ModelMap

- 这些参数都会存放在域中。可以在页面获取。（request域）

2）经过验证

- Map<String,String>
- Model
- ModelMap

Map、Model都是接口，ModelMap是具体的实现类

ModelMap extends java.util.LinkedHashMap

获得Map、Model、ModelMap形参的getClass()发现他是org.springframework.validation.support.BindingAwareModelMap类型。

**类之间的简化后的UML关系如图**

<img style="float:left" src="../pics/SpringMVC/BindingAwareModelMapUML.png">

----

```java
@Controller
@RequestMapping("/carry")
public class CarryController {

    @RequestMapping("/map")
    public String Map(Map<String, Object> map) {
        map.put("name", "liujiawe");
        // class org.springframework.validation.support.BindingAwareModelMap
        System.out.println(map.getClass());
        return "carry";
    }

    @RequestMapping("/model")
    public String Model(Model model) {
        model.addAttribute("name", "liujiawei model");
        // class org.springframework.validation.support.BindingAwareModelMap
        System.out.println(model.getClass());
        return "carry";
    }

    @RequestMapping("/modelMap")
    public String ModelMap(ModelMap map) {
        map.addAttribute("name", "Model Map vale");
        // class org.springframework.validation.support.BindingAwareModelMap
        System.out.println(map.getClass());
        return "carry";
    }
}
```

## 5.2 ModelAndView

1）方法的返回值可以变为ModelAndView类型

即包含视图信息（页面地址）也包含模型数据（给页面），而且数据是放在请求域中。

```java
public ModelAndView handle(){
    // 最后会跳转到 /WEB-INF/views/success.jsp页面。
    // 我设置了视图解析器，会给success拼前缀和后缀。
    // 带前缀的地址：redirect:/xx
   	// 			  forward:/xx 这些就不会被拼串，具体可以看视图解析那块的源码，看下就知道了。
    // 他是先看有没有 前缀，有就用对应前缀的View对象，最后没用，采用拼串的View对象。
	ModelAndViewmv = new ModelAndView("success");
	mv.addObject("msg","你好哦")
	return mv；
}
```

## 5.3 数据暂存Session域

使用一个注解 @SessionAttributes(只能标在类上)

@SessionAttributes(value="msg")：

- 给BindingAwareModelMap中保存的数据,同时给session中放一份。
- value指定保存数据时要给session中存放的数据的key。

@SessionAttributes(value={"msg"},types={String.class}})

- value={“msg”} 只要保存的是这种key的数据，给Session中放一份。
- types={String.class} 只要保存的是这种类型的数据，给Session中也放一份。
- 所以会存两大份！！用value指定的比较多，因为可以精确指定。
- 但是我们不推荐用@SessionAttributes，还是用原生API吧。注解的话可能会引发异常，且移除session麻烦。

## 5.4 ModelAttribute方法

ModelAttribute 方法入参标注该注解后，入参的对象就会放到数据模型中。

参数：取出刚才保存的数据

方法位置：这个方法就会提取于目标方法先运行。

​	可以在这里提前查出数据库中图书的信息。

​	将这个图书信息保存起来（方便下一个方法还能使用）

​	参数Map就是BindAwareMap

```java
/**
* ModelAttribute方法先执行，把数据存在数据模型域中。
* @ModelAttribute("user") User user先拿到模型域中的值，然后才用浏览器传* 过来的值进行数据更新
*/
@ResponseBody
@RequestMapping("/get")
public User get(@ModelAttribute("user") User user) {
    return user;
}

@ModelAttribute
public void ModelAttribute(Model model) {
    User user = new User();
    user.setAddress("address");
    user.setAge(18);
    user.setName("ljw");
    model.addAttribute("user", user);
}
```

**ModelAttribute图解**

<img src="../pics/SpringMVC/ModelAttribute.png" style="float:left">



# 六、SpringMVC---源码解析

SpringMVC源码

- SpringMVC源码如何看？
- SpringMVC所有的请求都会被前端控制器拦截到，所以看SpringMVC怎么处理请求的，就看前端控制器的处理流程，如何处理请求的。
- 只要是finally块的，一般就是清东西。
- try起来的一般是重要的代码。 

梳理完流程后，发现执行流程大概是这样的。

<img src="../pics/SpringMVC/mvc_process5.png" style="float:left">

**文字描述：**

# 不知名笔记


让我们看看createView方法

```java
@Override
protected View createView(String viewName, Locale locale) throws Exception {
   // If this resolver is not supposed to handle the given view,
   // return null to pass on to the next resolver in the chain.
   if (!canHandle(viewName, locale)) {
      return null;
   }

   // Check for special "redirect:" prefix.
   // 如果是redirect前缀 则xxx
   if (viewName.startsWith(REDIRECT_URL_PREFIX)) {
      String redirectUrl = viewName.substring(REDIRECT_URL_PREFIX.length());
      RedirectView view = new RedirectView(redirectUrl,
            isRedirectContextRelative(), isRedirectHttp10Compatible());
      String[] hosts = getRedirectHosts();
      if (hosts != null) {
         view.setHosts(hosts);
      }
      return applyLifecycleMethods(REDIRECT_URL_PREFIX, view);
   }

   // Check for special "forward:" prefix.
   if (viewName.startsWith(FORWARD_URL_PREFIX)) {
      String forwardUrl = viewName.substring(FORWARD_URL_PREFIX.length());
      InternalResourceView view = new InternalResourceView(forwardUrl);
      return applyLifecycleMethods(FORWARD_URL_PREFIX, view);
   }

   // Else fall back to superclass implementation: calling loadView.
   return super.createView(viewName, locale);
}
```

返回View对象。

1）视图解析器得到View对象的流程就是，所有配置的视图解析器都来尝试根据视图名（返回值）得到View对象；如果能得到就返回，得不到就换下一个视图解析器。

2）调用View对象的render方法。



一句话：

视图解析器只是为了得到视图对象；视图对象才能真正的<span style="color:red">转发（将模型数据全部放在请求域中）或者重定向到页面</span>视图对象才能真正的<span style="color:red">渲染视图</span>。

-------

# Other

**携带数据给页面**

mvc如何把数据带到页面？

1）可以在方法处传入Map、或Model或者ModelMap

这些参数都会存放在域中。可以在页面获取。（request域）

${pageScope.msg}

${requestScope.msg}

${sessionScope.msg}



- Map<String,String>
- Model
- ModelMap



Map Model都是接口，ModelMap是具体的实现类

ModelMap extends java.util.LinkedHashMap



获得Map、Model、ModelMap形参的getClass()发现他是org.springframework.validation.support.BindingAwareModelMap类型。





﻿![img](https://api.bilibili.com/x/note/image?image_id=9704)﻿







﻿![img](https://api.bilibili.com/x/note/image?image_id=9707)﻿



\-------------------------------------------

2）方法的返回值可以变为ModelAndView类型

即包含视图信息（页面地址）也包含模型数据（给页面）

而且数据是放在请求域中。

public ModelAndView handle(){

​	ModelAndViewmv = new ModelAndView("success");

​	mv.addObject("msg","你好哦")

​	return mv；

}

\-------------------------------------------

3）数据暂存到Session域中

使用一个注解 @SessionAttributes(只能标在类上)

@SessionAttributes(value="msg"):

​	给BindingAwareModelMap中保存的数据,同时给session中放一份。

​	value指定保存数据时要给session中存放的数据的key。



@SessionAttributes(value={"msg"},types={String.class}})



- value={“msg”} 只要保存的是这种key的数据，给Session中放一份。
- types={String.class} 只要保存的是这种类型的数据，给Session中也放一份。
- 所以会两大份！！用value比较多，精确指定。
- 但是我们不推荐用@SessionAttributes，还是用原生API吧。注解的话可能会引发异常，且移除session麻烦。



4）ModelAttribute 方法入参标注该注解后，入参的对象就会放到数据模型中。

使用场景：书城修改为例。

- 如何保证全字段更新的时候，只更新了页面携带的数据？
- 修改dao
- book对象是如何封装的？
- SpringMVC创建一个book对象
- 将请求中所有与book对应的属性一一设置过去。
- 调用全字段更新就有问题。
- ModelAndView，将数据库中的存起来，用原有的值覆盖mvc封装的null值。
- springmvc要封装请求参数的Book对象不应该是自己new出来的。而应该是【从数据库】拿到的准备好的对象。
- 再来使用这个对象封装请求参数



modelattribute：

​	参数：取出刚才保存的数据



​	方法位置：这个方法就会提取于目标方法先运行。

​			可以在这里提前查出数据库中图书的信息。

​			将这个图书信息保存起来（方便下一个方法还能使用）

​			参数Map就是BindAwareMap



﻿![img](https://api.bilibili.com/x/note/image?image_id=9804)﻿







﻿![img](https://api.bilibili.com/x/note/image?image_id=9806)﻿







﻿![img](https://api.bilibili.com/x/note/image?image_id=9808)﻿





======================================

SpringMVC源码

- SpringMVC源码如何看？
- SpringMVC所有的请求都会被前端控制器拦截到，所以看SpringMVC怎么处理请求的，就看前端控制器的处理流程，如何处理请求的。



梳理完流程后，发现执行流程大概是这样的。





﻿![img](https://api.bilibili.com/x/note/image?image_id=9818)﻿





下面写代码验证以下猜想。





﻿![img](https://api.bilibili.com/x/note/image?image_id=9819)﻿







﻿![img](https://api.bilibili.com/x/note/image?image_id=9820)﻿



**我们看DispatcherServlet的源码**，对它进行debug，看流程。

- ha.handle(processedRequest, response, mappedHandlerxx) 执行目标方法 945行
- processDispatchResult(xxx) 页面放行。  959行



```
protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
   HttpServletRequest processedRequest = request;
   HandlerExecutionChain mappedHandler = null;
   boolean multipartRequestParsed = false;
   // 异步管理器，如果有异步怎么办
   WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);

   try {
      ModelAndView mv = null;
      Exception dispatchException = null;

      try {
         // 检查是否多部件 和文件上传有关
         processedRequest = checkMultipart(request);
         multipartRequestParsed = (processedRequest != request);

         // Determine handler for the current request.
         // 根据当前请求地址决定哪个类能处理
         mappedHandler = getHandler(processedRequest);
         // 如果没有找到那个处理器可以处理这个请求，就404 报异常。
         if (mappedHandler == null) {
            noHandlerFound(processedRequest, response);
            return;
         }
         // 找到了的话，mappedHandler里的handler属性就会封装我们对应的Controller。
         // Determine handler adapter for the current request.
         // 决定当前请求要用那个处理器的适配器。SpringMVC不是直接反射调用对应Controller的方法，而是用一个处理器进行执行。此处的作用1是拿到能执行这个类的所以方法的适配器（反射工具）
         HandlerAdapter ha = getHandlerAdapter(mappedHandler.getHandler());

         // Process last-modified header, if supported by the handler.
         String method = request.getMethod();
         boolean isGet = "GET".equals(method);
         if (isGet || "HEAD".equals(method)) {
            long lastModified = ha.getLastModified(request, mappedHandler.getHandler());
            if (new ServletWebRequest(request, response).checkNotModified(lastModified) && isGet) {
               return;
            }
         }

         if (!mappedHandler.applyPreHandle(processedRequest, response)) {
            return;
         }

         // Actually invoke the handler.
         // 用适配器执行方法；将目标方法执行完成后的返回值作为视图名，设置保存到ModelAndView中。
         // 无论目标方法怎么写，最终适配器执行完成以后都会将执行后的信息封装成ModelAndView
         mv = ha.handle(processedRequest, response, mappedHandler.getHandler());

         if (asyncManager.isConcurrentHandlingStarted()) {
            return;
         }
         // 如果没有视图名，设置一个默认的视图名（方法没有返回值的时候）
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
      // 转发到目标页面。根据方法最终执行完成后封装的ModelAndView 转发到对应页面，而且ModelAndView中的数据可以从请求域中获取。
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

\---------------

**文字总结**





﻿![img](https://api.bilibili.com/x/note/image?image_id=9835)﻿



\-------------------

**DispatcherServlet中的一个方法getHandler() 如何找到那个类可以处理请求的。**

```
 mappedHandler = getHandler(processedRequest);// mappedHandler的类型是HandlerExecutionChain
```





﻿![img](https://api.bilibili.com/x/note/image?image_id=9836)﻿



debug的时候，直接放行到下一个断点，加快阅读速度。



```
protected HandlerExecutionChain getHandler(HttpServletRequest request) throws Exception {
   if (this.handlerMappings != null) {
      // HandlerMapping：处理器映射；他里面保存了每一个处理器能处理那些请求的映射信息。【标了注解】
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





﻿![img](https://api.bilibili.com/x/note/image?image_id=9837)﻿



```
BeanNameUrlHandlerMapping // 以前用xml方式配置
DefaultAnnotationHandlerMapping // 现在用注解方式了
  - 看里面的handlerMap属性。
  - private final Map<String, Object> handlerMap = new LinkedHashMap<String, Object>(); 超类的属性。the registered path as key and the handler object (or handler bean name in case of a lazy-init handler) as value.【即里面保存了对应的请求谁能处理】
```





﻿![img](https://api.bilibili.com/x/note/image?image_id=9838)﻿



我们刚刚看了

 DispatcherServlet 的--> doDispatch()方法的流程--->然后细看了里面的getHandler()方法。

接着我们看下getHandlerAdapter()方法，也是doDispatch里的方法。

为什么我们要看getHandlerAdapter方法？因为：

我拿到处理器还不够，还需要获得适配器！！为什么？（因为是适配器执行目标方法呀！）

如何找到目标处理器类的适配器。（我们要拿适配器执行目标方法！！！补适配器模式！！）

```
// 方法源码如下：
protected HandlerAdapter getHandlerAdapter(Object handler) throws ServletException {
   if (this.handlerAdapters != null) {
      // 找适配器，又是遍历适配器，看那个合适。
      for (HandlerAdapter adapter : this.handlerAdapters) {
         // 如果支持这个处理器就返回，不支持就继续找，没找到就抛异常。
         if (adapter.supports(handler)) {
            return adapter;
         }
      }
   }
   throw new ServletException("No adapter for handler [" + handler +
         "]: The DispatcherServlet configuration needs to include a HandlerAdapter that supports this handler");
}
```





**下面我们看看this.handlerAdapters里有多少适配器：有三个！三种类型的适配器！**

这三个适配器中那个有用？我们猜测是注解那个有用。**AnnotationMethodHandlerAdapter**，因为我们打的是注解！

**AnnotationMethodHandlerAdapter能解析注解方法的适配器；处理器类中只要有标了注解的这些方法就能用。**



﻿![img](https://api.bilibili.com/x/note/image?image_id=9878)﻿









﻿![img](https://api.bilibili.com/x/note/image?image_id=9880)﻿









﻿![img](https://api.bilibili.com/x/note/image?image_id=9881)﻿







﻿![img](https://api.bilibili.com/x/note/image?image_id=9882)﻿





==================================

我们已经找到了适配器，接下来就是看适配器如何执行目标方法了！！！

- mv = ha.handle(xxx) 这里调用的方法，我们点进去看。
- ﻿![img](https://api.bilibili.com/x/note/image?image_id=9888)﻿
- 他的返回值 invokeHandlerMethod执行的方法，我们再点进去看
- ﻿![img](https://api.bilibili.com/x/note/image?image_id=9889)﻿
- 走到下图的 methodInvoker.invokeHandlerMethod就执行了
- ﻿![img](https://api.bilibili.com/x/note/image?image_id=9890)﻿
- 

=========================

**springMvc的九大组件**

上次内容的回顾



﻿![img](https://api.bilibili.com/x/note/image?image_id=9917)﻿





DispatcherServet中有几个引用类型的属性；SpringMVC的九大组件。

SpringMVC在工作的时候，关键位置都是由这些组件完成的；

共同点：九大组件全部都是接口；接口就是规范；提供了非常强大的扩展性；

SpringMVC的九大组件工作原理：大佬级别。

```
/** 文件上传解析器 **/
@Nullable
private MultipartResolver multipartResolver;

/** 区域信息解析器. 和国际化有关*/
@Nullable
private LocaleResolver localeResolver;

/** 主题解析器；强大的主题效果更换 */
@Nullable
private ThemeResolver themeResolver;

/** Handler映射信息.HandlerMapping */
@Nullable
private List<HandlerMapping> handlerMappings;

/** Handler的适配器. */
@Nullable
private List<HandlerAdapter> handlerAdapters;

/** SpringMVC强大的异常解析功能；异常解析器. */
@Nullable
private List<HandlerExceptionResolver> handlerExceptionResolvers;

/** RequestToViewNameTranslator used by this servlet. */
@Nullable
private RequestToViewNameTranslator viewNameTranslator;

/** FlashMap+Manager：SpringMVC中运行重定向携带数据的功能  */
@Nullable
private FlashMapManager flashMapManager;

/** 视图解析器 */
@Nullable
private List<ViewResolver> viewResolvers;
```



DispatcherServlet中九大组件初始化的地方



```
@Override
protected void onRefresh(ApplicationContext context) {
   initStrategies(context);
}

/**
 * Initialize the strategy objects that this servlet uses.
 * <p>May be overridden in subclasses in order to initialize further strategy objects.
 */
protected void initStrategies(ApplicationContext context) {
   initMultipartResolver(context);
   initLocaleResolver(context);
   initThemeResolver(context);
   initHandlerMappings(context);
   initHandlerAdapters(context);
   initHandlerExceptionResolvers(context);
   initRequestToViewNameTranslator(context);
   initViewResolvers(context);
   initFlashMapManager(context);
}
```

============================

```
private void initHandlerMappings(ApplicationContext context) {
   this.handlerMappings = null;
   // 探查所有的HandlerMapping
   if (this.detectAllHandlerMappings) {
      // Find all HandlerMappings in the ApplicationContext, including ancestor contexts.
      Map<String, HandlerMapping> matchingBeans =
            BeanFactoryUtils.beansOfTypeIncludingAncestors(context, HandlerMapping.class, true, false);
      if (!matchingBeans.isEmpty()) {
         this.handlerMappings = new ArrayList<>(matchingBeans.values());
         // We keep HandlerMappings in sorted order.
         AnnotationAwareOrderComparator.sort(this.handlerMappings);
      }
   }
   else {
      try {
         HandlerMapping hm = context.getBean(HANDLER_MAPPING_BEAN_NAME, HandlerMapping.class);
         this.handlerMappings = Collections.singletonList(hm);
      }
      catch (NoSuchBeanDefinitionException ex) {
         // Ignore, we'll add a default HandlerMapping later.
      }
   }

   // Ensure we have at least one HandlerMapping, by registering
   // a default HandlerMapping if no other mappings are found.
   if (this.handlerMappings == null) {
      this.handlerMappings = getDefaultStrategies(context, HandlerMapping.class);
      if (logger.isTraceEnabled()) {
         logger.trace("No HandlerMappings declared for servlet '" + getServletName() +
               "': using default strategies from DispatcherServlet.properties");
      }
   }

   for (HandlerMapping mapping : this.handlerMappings) {
      if (mapping.usesPathPatterns()) {
         this.parseRequestPath = true;
         break;
      }
   }
}
```

组件的初始化：

- 有些组件在容器中是使用类型找的，有些组件是使用id找的。
- 就是去容器中找这个组件，如果没有就用默认的配置。
- 这是教怎么看各大组件的，具体的流程自己去看。



================================

**锁定目标方法的执行。**

给你三天，写一个可以执行任意方法的反射工具类。？？

```
mv = ha.handle(processedRequest, response, mappedHandler.getHandler());
```

debug进去看 进入到handle方法

```
@Override
public ModelAndView handle(HttpServletRequest request, HttpServletResponse response, Object handler)
      throws Exception {

   Class<?> clazz = ClassUtils.getUserClass(handler);
   Boolean annotatedWithSessionAttributes = this.sessionAnnotatedClassesCache.get(clazz);
   if (annotatedWithSessionAttributes == null) {
      // 用注解工具找这个类有没有SessionAttributes注解
      annotatedWithSessionAttributes = (AnnotationUtils.findAnnotation(clazz, SessionAttributes.class) != null);
      this.sessionAnnotatedClassesCache.put(clazz, annotatedWithSessionAttributes);
   }

   if (annotatedWithSessionAttributes) {
      checkAndPrepare(request, response, this.cacheSecondsForSessionAttributeHandlers, true);
   }
   else {
      checkAndPrepare(request, response, true);
   }

   // Execute invokeHandlerMethod in synchronized block if required.
   if (this.synchronizeOnSession) {
      HttpSession session = request.getSession(false);
      if (session != null) {
         Object mutex = WebUtils.getSessionMutex(session);
         synchronized (mutex) {
            return invokeHandlerMethod(request, response, handler);
         }
      }
   }
   // 再点击这个去看
   return invokeHandlerMethod(request, response, handler);
}
```

\---------------

```
protected ModelAndView invokeHandlerMethod(HttpServletRequest request, HttpServletResponse response, Object handler)
      throws Exception {
   // 拿到方法解析器
   ServletHandlerMethodResolver methodResolver = getMethodResolver(handler);
   // 方法解析器根据当前请求地址找到这个请求用什么方法执行
   Method handlerMethod = methodResolver.resolveHandlerMethod(request);
   // 创建一个方法执行器（做什么都整一个xx器）   
ServletHandlerMethodInvoker methodInvoker = new ServletHandlerMethodInvoker(methodResolver);
   // 包装原生的request、response
   ServletWebRequest webRequest = new ServletWebRequest(request, response);
   // 创建了一个BindingAwareModelMap 隐含模型【源码重点内容！！】
   ExtendedModelMap implicitModel = new BindingAwareModelMap();
   // 这是真正执行目标方法的。目标方法利用反射执行期间确定参数值，提前执行modelattribute等所有的操作都在这个方法中。
   Object result = methodInvoker.invokeHandlerMethod(handlerMethod, handler, webRequest, implicitModel);
   ModelAndView mav =
         methodInvoker.getModelAndView(handlerMethod, handler.getClass(), result, implicitModel, webRequest);
   methodInvoker.updateModelAttributes(handler, (mav != null ? mav.getModel() : null), implicitModel, webRequest);
   return mav;
}
```



===========================

ModelAttribute标注的方法提前运行

**164.17、【源码】-ModelAttribute标注的方法提前运行并且把执行P164 - 01:33**



方法执行细节

```
public final Object invokeHandlerMethod(Method handlerMethod, Object handler,
      NativeWebRequest webRequest, ExtendedModelMap implicitModel) throws Exception {
   // 找到我们要执行的目标方法
   Method handlerMethodToInvoke = BridgeMethodResolver.findBridgedMethod(handlerMethod);
   try {
      boolean debug = logger.isDebugEnabled();
      // 获取真正的SessionAttribute注解的名字。我们类上没加什么SessionAttribute，所以没什么关系。
      // 如果加了的话，就遍历每一个，把key对应的value查询出来（retrieve查询），塞在隐含模型中。
      for (String attrName : this.methodResolver.getActualSessionAttributeNames()) {
         // 
         Object attrValue = this.sessionAttributeStore.retrieveAttribute(webRequest, attrName);
         if (attrValue != null) {
            // 如果我们标了SessionAttribute，那么这些数据值会被放在隐含模型中
            implicitModel.addAttribute(attrName, attrValue);
         }
      }
      for (Method attributeMethod : this.methodResolver.getModelAttributeMethods()) {
         Method attributeMethodToInvoke = BridgeMethodResolver.findBridgedMethod(attributeMethod);
         Object[] args = resolveHandlerArguments(attributeMethodToInvoke, handler, webRequest, implicitModel);
         if (debug) {
            logger.debug("Invoking model attribute method: " + attributeMethodToInvoke);
         }
         String attrName = AnnotationUtils.findAnnotation(attributeMethod, ModelAttribute.class).value();
         if (!"".equals(attrName) && implicitModel.containsAttribute(attrName)) {
            continue;
         }
         ReflectionUtils.makeAccessible(attributeMethodToInvoke);
         Object attrValue = attributeMethodToInvoke.invoke(handler, args);
         if ("".equals(attrName)) {
            Class<?> resolvedType = GenericTypeResolver.resolveReturnType(attributeMethodToInvoke, handler.getClass());
            attrName = Conventions.getVariableNameForReturnType(attributeMethodToInvoke, resolvedType, attrValue);
         }
         if (!implicitModel.containsAttribute(attrName)) {
            implicitModel.addAttribute(attrName, attrValue);
         }
      }
      Object[] args = resolveHandlerArguments(handlerMethodToInvoke, handler, webRequest, implicitModel);
      if (debug) {
         logger.debug("Invoking request handler method: " + handlerMethodToInvoke);
      }
      ReflectionUtils.makeAccessible(handlerMethodToInvoke);
      return handlerMethodToInvoke.invoke(handler, args);
   }
   catch (IllegalStateException ex) {
      // Internal assertion failed (e.g. invalid signature):
      // throw exception with full handler method context...
      throw new HandlerMethodInvocationException(handlerMethodToInvoke, ex);
   }
   catch (InvocationTargetException ex) {
      // User-defined @ModelAttribute/@InitBinder/@RequestMapping method threw an exception...
      ReflectionUtils.rethrowException(ex.getTargetException());
      return null;
   }
}
```



```
// 如果没有注解：
// 1）先看是否普通参数；就是确定当前的参数是否是原生API。
resolveCommonArgument
// 2）是否有默认值  看不下去了，放弃，太难了。我要看书。
```

====================

确定POJO的值



**源码总结**





﻿![img](https://api.bilibili.com/x/note/image?image_id=9966)﻿





=============================================================================

**SpringMVC Day03**



**视图解析的应用**

**return "forward:/hello.jsp"//  转发到页面地址。**

- **forward：转发到一个页面**
- **/hello.jsp 转发当前项目下的hello**
- **一定要加/  如果不加/就是相对路径。容易出问题。**
- **forward:/hello.jsp 不会有给你拼串，有前缀的转发，不会由我们配置的视图解析器拼串。**

**forward可以转发到页面，也可以转发到一个请求上。 forward:/hello 转发到hello请求**



**redirect重定向【重定向的地址由浏览器进行解析】**

- 有前缀的转发和重定向不会有视图解析器的拼串操作。
- 原生的servlet重定向需要加上项目名才能重定向。
- springmvc无需写项目名，会为我们自动拼接上项目名。
- returen "redirect:/hello.jsp";





**SpringMVC视图解析器原理**

- 1、方法执行后的返回值会作为页面地址参考，转发或者重定向到页面
- 2、视图解析器可能会进行页面地址的拼串

\--------------------------

1）任何方法的返回值，最终都会被包装成ModelAndView对象。



﻿![img](https://api.bilibili.com/x/note/image?image_id=9976)﻿



2）处理页面的方法 processDispatchResult() 

视图渲染流程：将域中的数据在页面展示；页面就是用来渲染模型数据的。



3）调用render(mv, reuqest, response);渲染页面



4）View 与 ViewResolver；

- ViewResolver的作用是根据视图名（方法的返回值）得到View对象。





﻿![img](https://api.bilibili.com/x/note/image?image_id=9978)﻿



5）怎么根据方法的返回值（视图名）得到View对象？



﻿![img](https://api.bilibili.com/x/note/image?image_id=9979)﻿



想知道怎么初始化视图解析器的话，取看initViewResolvers方法

- 找到的话，就用我们配置的。
- 没找到的话，就用默认的。



﻿![img](https://api.bilibili.com/x/note/image?image_id=9980)﻿







```
@Nullable
protected View resolveViewName(String viewName, @Nullable Map<String, Object> model,
      Locale locale, HttpServletRequest request) throws Exception {

   if (this.viewResolvers != null) {
      for (ViewResolver viewResolver : this.viewResolvers) {
         // 根据视图名，得到view对象。 点进对应的方法去看
         View view = viewResolver.resolveViewName(viewName, locale);
         if (view != null) {
            return view;
         }
      }
   }
   return null;
}
```



```
// resolveViewName细节实现
@Override
@Nullable
public View resolveViewName(String viewName, Locale locale) throws Exception {
   if (!isCache()) {
      return createView(viewName, locale);
   }
   else {
      Object cacheKey = getCacheKey(viewName, locale);
      View view = this.viewAccessCache.get(cacheKey);
      if (view == null) {
         synchronized (this.viewCreationCache) {
            view = this.viewCreationCache.get(cacheKey);
            if (view == null) {
               // Ask the subclass to create the View object.
               // 根据方法的返回值创建出视图对象。debug进去看看。
               view = createView(viewName, locale);
               if (view == null && this.cacheUnresolved) {
                  view = UNRESOLVED_VIEW;
               }
               if (view != null && this.cacheFilter.filter(view, viewName, locale)) {
                  this.viewAccessCache.put(cacheKey, view);
                  this.viewCreationCache.put(cacheKey, view);
               }
            }
         }
      }
      else {
         if (logger.isTraceEnabled()) {
            logger.trace(formatKey(cacheKey) + "served from cache");
         }
      }
      return (view != UNRESOLVED_VIEW ? view : null);
   }
}
```