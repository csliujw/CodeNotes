# 概述

记录如何使用 Spring、Spring MVC 等框架提供的扩展点进行扩展。

# 扩展总揽

- Spring、Spring MVC 自定义注解
    - Spring 可以通过 AOP 进行方法拦截，然后解析拦截的方法是否存在xxx注解，执行相应的操作。
    - Spring MVC 可以通过拦截器的方式进行去解析自定义的注解，看方法上是否有注解，进行相应操作。
- Spring MVC HandlerInterceptor 接口--拦截器
    - HandlerInterceptor，通过自定义拦截器，我们可以在一个请求被真正处理之前、请求被处理但还没输出到响应中、请求已经被输出到响应中之后这三个时间点去做任何我们想要做的事情。
- Spring MVC HandlerMapping接口 -- 处理请求的映射
    - 保存请求url到具体的方法的映射关系，，我们可以编写任意的HandlerMapping实现类，依据任何策略来决定一个web请求到HandlerExecutionChain对象的生成。
- HandlerAdapter接口 – 处理适配器
    - 真正调用Controller的地方，其实就是适配各种Controller。HandlerAdapter就是你可以提供自己的实现类来处理handler对象。
- HandlerMethodArgumentResolver -- 处理方法参数解释绑定器
    - 调用controller方法之前，对方法参数进行解释绑定（实现WebArgumentResolver接口，spring3.1以后推荐使用HandlerMethodArgumentResolver）；

# Spring 的扩展

# Spring MVC 的扩展

可以扩展的内容位于 `WebMvcConfigurerAdapter`

https://codeantenna.com/a/OO1sywTUnO



