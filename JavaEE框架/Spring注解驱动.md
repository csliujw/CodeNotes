# 概述

## 学习内容

- 容器
  - `AnnotationConfigApplicationContext`
  - 组件添加
  - 组件赋值
  - 组件注入
  - `AOP`
  - 声明式事务
- 扩展原理
  - `BeanFactoryPostProcessor`
  - `BeanDefinitionRegistryPostProcessor`
  - `ApplicationListener`
  - Spring容器创建过程
- web
  - `servlet3.0`请求
  - 异步请求



## maven报错

这个报错了，怎么办？

```xml
<plugin>
    <artifactId>maven-project-info-reports-plugin</artifactId>
    <version>3.0.0</version>
</plugin>
```

引入这个依赖就行！

```xml
<dependency>
    <groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-project-info-reports-plugin</artifactId>
    <version>3.0.0</version>
    <type>maven-plugin</type>
</dependency>
```

# Spring注解--组件注解

## 导包

导入spring-context包后，其他一些包也会自动导入哦~即核心容器所依赖的所有环境也会被。

```xml
<!-- https://mvnrepository.com/artifact/org.springframework/spring-context -->
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-context</artifactId>
    <version>5.3.3</version>
</dependency>
```

## 原始的xml方式写法

- pojo对象

```java
package org.example.pojo;

public class Person {
    private String name;
    private Integer age;

    public Person() {
    }

    public Person(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    @Override
    public String toString() { return this.name; }
}

```

- 获取bean

```java
package org.example;


import org.example.pojo.Person;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class BeanXMLTest {
    public static void main(String[] args) {
        ClassPathXmlApplicationContext context = new ClassPathXmlApplicationContext("bean.xml");
        Person person = (Person) context.getBean("person");
        Person person2 = (Person)context.getBean("person2");
        System.out.println(person2);
        context.close();
    }
}
```

- `xml`配置文件。在maven的`resrouce`目录下哦。resource目录下的资源最后会变成项目根目录下的文件。所以是直接`Classxxx("bean.xml")`
- `JavaSE`的项目和`JavaEE`项目最后的输出路径好像都是classes，但是`JavaEE`里写路径得是`classpath`！

## 用注解配置类

- @Configuration 可以替代xml，进行类的配置。典型的应用有三方jar包，我们需要把它交给Spring容器进行管理，于是用@Configuration的方式把这个类注入到Spring中。

`JavaConfig`配置类

```java
package org.example.configuration;

import org.example.pojo.Person;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MainConfiguration {
    // 给容器中注册一个Bean  默认是以方法名为bean的名称，如果不想要方法名可以这样 @Bean("person") 或 @Bean({"person1","person2"})
    // 具体看看源码注释 一目了然。
    // value 与 name之间 好像是别名关系
    @Bean("person3")
    public Person person() {
        return new Person();
    }
}
```

测试代码

```java
public class BeanXMLTest {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context2 = new AnnotationConfigApplicationContext(MainConfiguration.class);
        Person person21 = (Person) context2.getBean( "person3");
        System.out.println(person21);
    }
}

// =============================================================================
public class BeanXMLTest {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context2 = new AnnotationConfigApplicationContext(MainConfiguration.class);
        String[] beanNamesForType = context2.getBeanNamesForType(Person.class);
        String[] beanDefinitionNames = context2.getBeanDefinitionNames();
        for (int i = 0; i < beanDefinitionNames.length; i++) {
            System.out.println(beanDefinitionNames[i]);
        }
        System.out.println("===========");
        for (int i = 0; i < beanNamesForType.length; i++) {
            // 同一个类的多个别名好像只会输出其中一个。
            System.out.println(beanNamesForType[i]);
        }
    }
}
```

## 包扫描

用到的注解有

- `@Configuration`
- `@ComponentScan`，如果是`jdk8`，它被设置成了重复注解，可以重复用。

- `xml`的配置方式

  ```xml
  <!-- 配置包扫描 , 只要标注了@Controller、@Service、@Repository、@Component的都会被自动的扫描加入容器中-->
  <context:component-scan base-package="org.example"></context:component-scan>
  ```

- 注解方式，按指定类型排除

  ```java
  // excludeFilters指定排除那些  用@Filter指定排除那些
  // includeFilters指定包含那些  用@Filter指定包含那些
  // 要让includeFilters生效需要设置@ComponentScan的useDefaultFilters=false
  // MainConfiguration的配置对象不会被排除的
  @Configuration
  @ComponentScan(basePackages = "org.example", excludeFilters = {
          @ComponentScan.Filter(type = FilterType.ANNOTATION, classes = {Controller.class, Service.class})
  })
  public class MainConfiguration {
      // 给容器中注册一个Bean
      @Bean(name = {"person1", "person2", "person3"})
      public Person person() {
          return new Person();
      }
  
      @Bean
      public Person person007() {
          return new Person();
      }
  }
  ```

- 注解方式，按指定规则包含

  ```java
  // IncludeConfiguration的配置对象是也会包含的。
  @Configuration
  @ComponentScan(basePackages = "org.example", includeFilters = {
          @ComponentScan.Filter(type = FilterType.ASSIGNABLE_TYPE, classes = DemoService.class)
  }, useDefaultFilters = false)
  public class IncludeConfiguration {
      // 给容器中注册一个Bean
      @Bean(name = {"person1", "person2", "person3"})
      public Person person() {
          return new Person();
      }
  
      @Bean
      public Person person007() {
          return new Person();
      }
  }
  
  ```

  ```java
  public class ScanTest {
  
      @Test
      public void test1() {
          AnnotationConfigApplicationContext anno = new AnnotationConfigApplicationContext(MainConfiguration.class);
          String[] beanDefinitionNames = anno.getBeanDefinitionNames();
          for (int i = 0; i < beanDefinitionNames.length; i++) {
              System.out.println(beanDefinitionNames[i]);
          }
      }
  
      @Test
      public void test2() {
          AnnotationConfigApplicationContext anno = new AnnotationConfigApplicationContext(IncludeConfiguration.class);
          String[] beanDefinitionNames = anno.getBeanDefinitionNames();
          for (int i = 0; i < beanDefinitionNames.length; i++) {
              System.out.println(beanDefinitionNames[i]);
          }
      }
  }
  ```

## @Filter自定义过滤规则

自定义过滤规则的代码

```java
package org.example.configuration;

import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.FilterType;
import org.springframework.core.io.Resource;
import org.springframework.core.type.AnnotationMetadata;
import org.springframework.core.type.ClassMetadata;
import org.springframework.core.type.classreading.MetadataReader;
import org.springframework.core.type.classreading.MetadataReaderFactory;
import org.springframework.core.type.filter.TypeFilter;

import java.io.IOException;

@Configuration
@ComponentScan(basePackages = "org.example", includeFilters = {
        @ComponentScan.Filter(type = FilterType.CUSTOM, classes = {DefineFilter.class})
}, useDefaultFilters = false)
public class DefineFilterConfiguration {
}

class DefineFilter implements TypeFilter {
    // 自定义匹配规则
    @Override
    public boolean match(MetadataReader metadataReader, MetadataReaderFactory metadataReaderFactory) throws IOException {
        AnnotationMetadata annotationMetadata = metadataReader.getAnnotationMetadata();
        // 获得当前正在扫描的类信息
        ClassMetadata classMetadata = metadataReader.getClassMetadata();
        // 获得当前类资源（类路径）
        Resource resource = metadataReader.getResource();
        // 类名
        String className = classMetadata.getClassName();
        System.out.println("---->" + className);
        if (className.contains("Dao")) {
            return true;
        }
        return false;
    }
}
```

测试代码

```java
package org.example;

import org.example.configuration.DefineFilterConfiguration;
import org.example.configuration.IncludeConfiguration;
import org.example.configuration.MainConfiguration;
import org.junit.Test;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;


public class ScanTest {

    @Test
    public void test3() {
        AnnotationConfigApplicationContext anno = new AnnotationConfigApplicationContext(DefineFilterConfiguration.class);
        String[] beanDefinitionNames = anno.getBeanDefinitionNames();
        for (int i = 0; i < beanDefinitionNames.length; i++) {
            System.out.println(beanDefinitionNames[i]);
        }
    }
}
```

## Spring单元测试

引入依赖，需要的`JUnit`的版本有点高

[Spring测试官方文档](https://docs.spring.io/spring-framework/docs/current/reference/html/testing.html)

```xml
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-context</artifactId>
    <version>5.3.3</version>
</dependency>

<dependency>
    <groupId>junit</groupId>
    <artifactId>junit</artifactId>
    <version>4.13.1</version>
    <scope>test</scope>
</dependency>

<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-test</artifactId>
    <version>5.3.3</version>
    <scope>test</scope>
</dependency>
```

编写测试代码 (注入的测试类就不贴代码了)

```java
@ContextConfiguration(classes = ScopeConfiguration.class)
@RunWith(SpringJUnit4ClassRunner.class)
public class ScopeConfigurationTest {

    @Autowired
    Person person1;

    @Autowired
    Person person2;

    @Autowired
    Person person3;

    @Test
    public void test1() {
        System.out.println(person1 == person2);
        System.out.println(person1 == person3);
    }
}
```



## Bean作用域范围

- singleton 单例
- prototype 多例
- request 
- session
- global-session

## 懒加载

- @Lazy ，针对单实例 容器启动时不创建对象，第一次获取bean时再进行初始化。

## 按条件注入

- @Conditional
- [视频看到](https://www.bilibili.com/video/BV1gW411W7wy?p=7)

- 开始PyTorch