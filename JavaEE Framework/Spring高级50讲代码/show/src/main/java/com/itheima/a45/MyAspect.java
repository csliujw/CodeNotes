package com.itheima.a45;

import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Before;
import org.springframework.stereotype.Component;

@Aspect
@Component
public class MyAspect {

    // 故意对所有方法增强
    @Before("execution(* com.itheima.a45.Bean1.*(..))")
    public void before() {
        System.out.println("before");
    }
}
