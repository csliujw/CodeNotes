package org.springframework.boot.autoconfigure.web.servlet;

import org.springframework.boot.web.servlet.context.AnnotationConfigServletWebServerApplicationContext;
import org.springframework.web.HttpRequestHandler;
import org.springframework.web.servlet.function.HandlerFunction;

public class A35 {
    public static void main(String[] args) {
        AnnotationConfigServletWebServerApplicationContext context
                = new AnnotationConfigServletWebServerApplicationContext(WebConfig.class);

        /*
            学到了什么
            静态资源处理
                a. SimpleUrlHandlerMapping 映射路径
                b. ResourceHttpRequestHandler 作为静态资源 handler
                c. HttpRequestHandlerAdapter, 调用此 handler

            欢迎页处理
                a. WelcomePageHandlerMapping, 映射欢迎页(即只映射 '/')
                    - 它内置了 handler ParameterizableViewController 作用是不执行逻辑, 仅根据视图名找视图
                    - 视图名固定为 forward:index.html       /**
                b. SimpleControllerHandlerAdapter, 调用 handler
                    - 转发至 /index.html
                    - 处理 /index.html 又会走上面的静态资源处理流程

        */

        /*
            小结
                a. HandlerMapping 负责建立请求与控制器之间的映射关系
                    - RequestMappingHandlerMapping (与 @RequestMapping 匹配)
                    - WelcomePageHandlerMapping    (/)
                    - BeanNameUrlHandlerMapping    (与 bean 的名字匹配 以 / 开头)
                    - RouterFunctionMapping        (函数式 RequestPredicate, HandlerFunction)
                    - SimpleUrlHandlerMapping      (静态资源 通配符 /** /img/**)
                    之间也会有顺序问题, boot 中默认顺序如上
                b. HandlerAdapter 负责实现对各种各样的 handler 的适配调用
                    - RequestMappingHandlerAdapter 处理：@RequestMapping 方法
                           参数解析器、返回值处理器体现了组合模式
                    - SimpleControllerHandlerAdapter 处理：Controller 接口
                    - HandlerFunctionAdapter 处理：HandlerFunction 函数式接口
                    - HttpRequestHandlerAdapter 处理：HttpRequestHandler 接口 (静态资源处理)
                    这也是典型适配器模式体现
                c. ResourceHttpRequestHandler.setResourceResolvers 这是典型责任链模式体现
         */
    }
}
