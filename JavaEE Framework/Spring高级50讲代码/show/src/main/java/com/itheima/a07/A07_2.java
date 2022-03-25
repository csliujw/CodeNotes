package com.itheima.a07;

import org.springframework.beans.factory.support.BeanDefinitionBuilder;
import org.springframework.beans.factory.support.DefaultListableBeanFactory;

public class A07_2 {

    public static void main(String[] args) {
        DefaultListableBeanFactory beanFactory = new DefaultListableBeanFactory();
        beanFactory.registerBeanDefinition(
                "myBean",
                BeanDefinitionBuilder.genericBeanDefinition(MyBean.class)
                        .setDestroyMethodName("destroy")
                        .getBeanDefinition()
        );

        System.out.println(beanFactory.getBean(MyBean.class));
        beanFactory.destroySingletons(); // 销毁之后, 仍可创建新的单例
        System.out.println(beanFactory.getBean(MyBean.class));

    }

    static class MyBean {
        public MyBean() {
            System.out.println("MyBean()");
        }

        public void destroy() {
            System.out.println("destroy()");
        }
    }
}
