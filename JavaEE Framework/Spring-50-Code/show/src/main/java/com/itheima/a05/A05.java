package com.itheima.a05;

import com.itheima.a05.mapper.Mapper1;
import com.itheima.a05.mapper.Mapper2;
import org.mybatis.spring.mapper.MapperScannerConfigurer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.support.AbstractBeanDefinition;
import org.springframework.beans.factory.support.BeanDefinitionBuilder;
import org.springframework.beans.factory.support.DefaultListableBeanFactory;
import org.springframework.context.annotation.AnnotationBeanNameGenerator;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.ConfigurationClassPostProcessor;
import org.springframework.context.support.GenericApplicationContext;
import org.springframework.core.annotation.AnnotationUtils;
import org.springframework.core.io.Resource;
import org.springframework.core.type.classreading.CachingMetadataReaderFactory;
import org.springframework.core.type.classreading.MetadataReader;
import org.springframework.stereotype.Component;

import java.awt.*;
import java.io.IOException;
import java.util.Arrays;

/*
    BeanFactory 后处理器的作用
 */
public class A05 {
    private static final Logger log = LoggerFactory.getLogger(A05.class);

    // 容器中只有 Config 这个类。Config 中的 bean 未被解析
    public static void testContext1() {
        // ⬇️GenericApplicationContext 是一个【干净】的容器
        GenericApplicationContext context = new GenericApplicationContext();
        context.registerBean("config", Config.class);

        // ⬇️初始化容器
        context.refresh();

        for (String name : context.getBeanDefinitionNames()) {
            System.out.println(name);
        }

        // ⬇️销毁容器
        context.close();
    }

    // Config 中的 bean 也会被解析
    public static void testContextResovel() {
        // ⬇️GenericApplicationContext 是一个【干净】的容器
        GenericApplicationContext context = new GenericApplicationContext();
        context.registerBean("config", Config.class);
        // bean 工厂后处理器 @ComponentScan @Bean @Import @ImportResource
        context.registerBean(ConfigurationClassPostProcessor.class);

        // ⬇️初始化容器
        context.refresh();

        for (String name : context.getBeanDefinitionNames()) {
            System.out.println(name);
        }

        // ⬇️销毁容器
        context.close();
    }

    // Config 中的 bean 也会被解析
    // 扫描 MyBatis 的 mapper 接口，把它们也注入到 bean 工厂中
    public static void testContextWithMyBatis() {
        // ⬇️GenericApplicationContext 是一个【干净】的容器
        GenericApplicationContext context = new GenericApplicationContext();
        context.registerBean("config", Config.class);
        // bean 工厂后处理器 @ComponentScan @Bean @Import @ImportResource
        context.registerBean(ConfigurationClassPostProcessor.class);
        // MyBatis-Spring 的整合包
        context.registerBean(MapperScannerConfigurer.class, bd -> { // @MapperScanner
            bd.getPropertyValues().add("basePackage", "com.itheima.a05.mapper");
        });
        // ⬇️初始化容器
        context.refresh();

        for (String name : context.getBeanDefinitionNames()) {
            System.out.println(name);
        }

        // ⬇️销毁容器
        context.close();
    }


    // 自定义注解解析
    public static void testContextBySelf() {
        // ⬇️GenericApplicationContext 是一个【干净】的容器
        GenericApplicationContext context = new GenericApplicationContext();
        context.registerBean("config", Config.class);

        context.registerBean(AtBeanPostProcessor.class); // 解析 @Bean
        context.registerBean(MapperPostProcessor.class); // 解析 Mapper 接口


        // ⬇️初始化容器
        context.refresh();

        for (String name : context.getBeanDefinitionNames()) {
            System.out.println(name);
        }

        // ⬇️销毁容器
        context.close();
    }

    // 自定义注解解析 -- 获取注解信息
    public static void testContextBySelf2() throws IOException {
        // ⬇️GenericApplicationContext 是一个【干净】的容器
        GenericApplicationContext context = new GenericApplicationContext();
        context.registerBean("config", Config.class);

        ComponentScan annotation = AnnotationUtils.findAnnotation(Config.class, ComponentScan.class);
        if (annotation != null) {
            String[] classes = annotation.basePackages();
            for (String name : classes) {
                System.out.println(name);
                // 把路径 com.itheima.a05.component 变成 classpath*:com/itheima/a05/component
                name = "classpath*:" + name.replaceAll("\\.", "/") + "/**/*.class";
                System.out.println(name);
                CachingMetadataReaderFactory factory = new CachingMetadataReaderFactory();
                Resource[] resources = context.getResources(name);

                for (Resource resource : resources) {
                    MetadataReader metadataReader = factory.getMetadataReader(resource);
                    System.out.println("类名:" + metadataReader.getClassMetadata().getClassName());
                    System.out.println("是否含有 Component 注解" + metadataReader.getAnnotationMetadata().hasAnnotation(Component.class.getName()));
                    // 间接加了也算
                    System.out.println("是否间接含有 Component 注解" + metadataReader.getAnnotationMetadata().hasMetaAnnotation(Component.class.getName()));
                    System.out.println("==============================================================================");
                }
            }
        }

        // ⬇️初始化容器
        context.refresh();

        for (String name : context.getBeanDefinitionNames()) {
            System.out.println(name);
        }

        // ⬇️销毁容器
        context.close();
    }

    // 自定义注解解析 -- 注册 beanDefinition
    public static void testContextBySelf3() throws IOException {
        // ⬇️GenericApplicationContext 是一个【干净】的容器
        GenericApplicationContext context = new GenericApplicationContext();
        context.registerBean("config", Config.class);

        ComponentScan annotation = AnnotationUtils.findAnnotation(Config.class, ComponentScan.class);
        AnnotationBeanNameGenerator generator = new AnnotationBeanNameGenerator();

        if (annotation != null) {
            String[] classes = annotation.basePackages();
            for (String name : classes) {
                System.out.println(name);
                // 把路径 com.itheima.a05.component 变成 classpath*:com/itheima/a05/component
                name = "classpath*:" + name.replaceAll("\\.", "/") + "/**/*.class";
                System.out.println(name);
                CachingMetadataReaderFactory factory = new CachingMetadataReaderFactory();
                Resource[] resources = context.getResources(name);

                for (Resource resource : resources) {
                    MetadataReader metadataReader = factory.getMetadataReader(resource);
                    if (metadataReader.getAnnotationMetadata().hasAnnotation(Component.class.getName())
                            || metadataReader.getAnnotationMetadata().hasMetaAnnotation(Component.class.getName())) {
                        DefaultListableBeanFactory beanFactory = context.getDefaultListableBeanFactory();
                        AbstractBeanDefinition beanDefinition = BeanDefinitionBuilder.genericBeanDefinition(metadataReader.getClassMetadata().getClassName()).getBeanDefinition();
                        String beanName = generator.generateBeanName(beanDefinition, context.getDefaultListableBeanFactory());
                        beanFactory.registerBeanDefinition(beanName, beanDefinition);
                    }
                }
            }
        }

        // ⬇️初始化容器
        context.refresh();

        for (String name : context.getBeanDefinitionNames()) {
            System.out.println(name);
        }

        // ⬇️销毁容器
        context.close();
    }


    public static void finalTest() {

        // ⬇️GenericApplicationContext 是一个【干净】的容器
        GenericApplicationContext context = new GenericApplicationContext();

        context.registerBean("config", Config.class);
        context.registerBean(AtBeanPostProcessor.class); // 解析 @Bean
        context.registerBean(MapperPostProcessor.class); // 解析 Mapper 接口

        // ⬇️初始化容器
        context.refresh();

        for (String name : context.getBeanDefinitionNames()) {
            System.out.println(name);// com.itheima.a05.component
        }

        Mapper1 mapper1 = context.getBean(Mapper1.class);
        Mapper2 mapper2 = context.getBean(Mapper2.class);

        // ⬇️销毁容器
        context.close();

        /*
            学到了什么
                a. @ComponentScan, @Bean, @Mapper 等注解的解析属于核心容器(即 BeanFactory)的扩展功能
                b. 这些扩展功能由不同的 BeanFactory 后处理器来完成, 其实主要就是补充了一些 bean 定义
         */
    }

    public static void main(String[] args) throws IOException {
        testContextBySelf3();
    }
}
