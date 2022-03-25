package com.itheima.a15;

import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.aop.aspectj.annotation.AnnotationAwareAspectJAutoProxyCreator;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ConfigurationClassPostProcessor;
import org.springframework.context.annotation.Scope;
import org.springframework.context.annotation.ScopedProxyMode;
import org.springframework.context.support.GenericApplicationContext;

// 代理能否重复被代理
public class A15_1 {
    public static void main(String[] args) {
        GenericApplicationContext context = new GenericApplicationContext();
        context.registerBean("myConfig", MyConfig.class);
        context.registerBean(ConfigurationClassPostProcessor.class);
        context.registerBean("aspect1", Aspect1.class);
        context.registerBean(AnnotationAwareAspectJAutoProxyCreator.class);
        context.refresh();

        Bean1 bean1 = context.getBean(Bean1.class);
        bean1.foo();
        System.out.println(bean1.getClass());
        // 此代理是为了解决功能增强问题
        System.out.println(context.getBean("scopedTarget.bean1").getClass());
    }

    static class MyConfig {
        // 此代理是为了解决单例注入多例问题
        @Bean
        @Scope(proxyMode = ScopedProxyMode.TARGET_CLASS)
        public Bean1 bean1() {
            return new Bean1();
        }
    }

    static class Bean1 {
        public void foo() {
            System.out.println("bean1 foo");
        }
    }

    @Aspect
    static class Aspect1 {
        @Around("execution(* foo())")
        public Object before(ProceedingJoinPoint pjp) throws Throwable {
            System.out.println("aspect1 around");
            return pjp.proceed();
        }
    }
}
