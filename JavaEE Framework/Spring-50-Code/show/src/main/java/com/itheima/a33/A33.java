package com.itheima.a33;

import org.springframework.boot.web.servlet.context.AnnotationConfigServletWebServerApplicationContext;

public class A33 {
    public static void main(String[] args) {
        AnnotationConfigServletWebServerApplicationContext context
                = new AnnotationConfigServletWebServerApplicationContext(WebConfig.class);

        /*
            学到了什么
                a. BeanNameUrlHandlerMapping, 以 / 开头的 bean 的名字会被当作映射路径
                b. 这些 bean 本身当作 handler, 要求实现 Controller 接口
                c. SimpleControllerHandlerAdapter, 调用 handler

            对比
                a. RequestMappingHandlerAdapter, 以 @RequestMapping 作为映射路径
                b. 控制器的具体方法会被当作 handler
                c. RequestMappingHandlerAdapter, 调用 handler
         */
    }
}
