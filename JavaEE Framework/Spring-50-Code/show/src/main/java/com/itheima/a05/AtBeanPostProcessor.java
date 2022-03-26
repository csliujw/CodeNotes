package com.itheima.a05;

import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.ConfigurableListableBeanFactory;
import org.springframework.beans.factory.support.AbstractBeanDefinition;
import org.springframework.beans.factory.support.BeanDefinitionBuilder;
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.beans.factory.support.BeanDefinitionRegistryPostProcessor;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ClassPathResource;
import org.springframework.core.type.MethodMetadata;
import org.springframework.core.type.classreading.CachingMetadataReaderFactory;
import org.springframework.core.type.classreading.MetadataReader;

import java.io.IOException;
import java.util.Set;

public class AtBeanPostProcessor implements BeanDefinitionRegistryPostProcessor {
    @Override
    public void postProcessBeanFactory(ConfigurableListableBeanFactory configurableListableBeanFactory) throws BeansException {

    }

    @Override
    public void postProcessBeanDefinitionRegistry(BeanDefinitionRegistry beanFactory) throws BeansException {
        try {
            CachingMetadataReaderFactory factory = new CachingMetadataReaderFactory();
            // 不走类加载，效率比反射高。
            MetadataReader reader = factory.getMetadataReader(new ClassPathResource("com/itheima/a05/Config.class"));
            // 拿到所有被 @Bean 标注的方法的信息
            Set<MethodMetadata> methods = reader.getAnnotationMetadata().getAnnotatedMethods(Bean.class.getName());

            // 根据方法信息创建 BeanDefinition
            for (MethodMetadata method : methods) {
                System.out.println(method);
                // 获取注解的 initMethod 属性
                String initMethod = method.getAnnotationAttributes(Bean.class.getName()).get("initMethod").toString();

                // 现在我们要把这些 @Bean 变成工厂方法，变成工厂方法无须指定类名
                BeanDefinitionBuilder builder = BeanDefinitionBuilder.genericBeanDefinition();
                // 方法名字:method.getMethodName()      factoryBean: 先得有工厂对象 Config，才能调用工厂对象中的方法。工厂对象的名字是 config
                builder.setFactoryMethodOnBean(method.getMethodName(), "config");
                // 不加这个 SqlSessionFactoryBean 注入会有问题。因为它多了一个参数，这个参数需要指定自动装配，不然无法解析。
                // 对于工厂方法和构造方法的参数，如果想自动装配，选择 AUTOWIRE_CONSTRUCTOR
                builder.setAutowireMode(AbstractBeanDefinition.AUTOWIRE_CONSTRUCTOR);

                if (initMethod.length() > 0) {
                    builder.setInitMethodName(initMethod);
                }
                AbstractBeanDefinition bd = builder.getBeanDefinition();
                beanFactory.registerBeanDefinition(method.getMethodName(), bd);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
