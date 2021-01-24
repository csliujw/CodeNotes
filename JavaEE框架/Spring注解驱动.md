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
- 验证代码如下

```java
@Configuration
public class LazyConfiguration {
    @Scope("prototype")
    @Bean
    @Lazy
    public Person person() {
        System.out.println("Create Person");
        return new Person();
    }

    @Bean
    public Person getP() {
        System.out.println("Create getP");
        return new Person();
    }
}
```



## 按条件注入

- @Conditional，

  ```java
  @Target({ElementType.TYPE, ElementType.METHOD}) // 方法
  @Retention(RetentionPolicy.RUNTIME)
  @Documented
  public @interface Conditional {
  
  	/**
  	 * All {@link Condition} classes that must {@linkplain Condition#matches match}
  	 * in order for the component to be registered.
  	 */
  	Class<? extends Condition>[] value();
  
  }
  
  // 再看Class<? exntends Condition>[] 中的Condition
  @FunctionalInterface
  public interface Condition {
  
  	/**
  	 * Determine if the condition matches.
  	 * @param context the condition context
  	 * @param metadata the metadata of the {@link org.springframework.core.type.AnnotationMetadata class}
  	 * or {@link org.springframework.core.type.MethodMetadata method} being checked
  	 * @return {@code true} if the condition matches and the component can be registered,
  	 * or {@code false} to veto the annotated component's registration
  	 */
  	boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata);
  
  }
  // 由此可看出，Conditional传入的是Condition数组
  ```

- 按条件注入具体Demo

  ```java
  package org.example.configuration;
  
  import org.example.pojo.Person;
  import org.springframework.beans.factory.support.BeanDefinitionRegistry;
  import org.springframework.context.annotation.*;
  import org.springframework.core.type.AnnotatedTypeMetadata;
  
  class LinuxCondition implements Condition {
  
      /**
       * @param context  判断能使用的上下文环境
       * @param metadata 当前标注了Condtion注解的标注信息
       * @return
       */
      @Override
      public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
          String property = context.getEnvironment().getProperty("os.name");
          if (property != null && property.contains("linux"))
              return true;
          return false;
      }
  }
  
  class WindowsCondition implements Condition {
  
      @Override
      public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
          String property = context.getEnvironment().getProperty("os.name");
          if (property != null && property.contains("Window"))
              return true;
          return false;
      }
  }
  
  /**
   * 包含某个bean才xxx
   */
  class ConditionDemo implements Condition {
  
      @Override
      public boolean matches(ConditionContext context, AnnotatedTypeMetadata metadata) {
          context.getBeanFactory();
          context.getClassLoader();
          context.getEnvironment();
          BeanDefinitionRegistry registry = context.getRegistry();
          boolean windows = registry.containsBeanDefinition("windows");
          if (windows)
              return true;
          return false;
          // bean的注冊还未学习
      }
  }
  
  
  @Configuration
  public class ConditionConfiguration {
  
      @Bean("linux")
      @Conditional(value = {LinuxCondition.class})
      public Person getLinux() {
          return new Person("linux");
      }
  
      @Bean("windows")
      @Conditional(value = {WindowsCondition.class})
      public Person getWindows() {
          return new Person("windows");
      }
  
      // 包含指定的Bean才注入此obj对象
      @Bean("obj")
      @Conditional(value = {ConditionDemo.class})
      public Object getObj() {
          return new Object();
      }
  }
  ```

- 测试代码

  ```java
  package org.example;
  
  import org.example.configuration.ConditionConfiguration;
  import org.example.pojo.Person;
  import org.junit.Test;
  import org.junit.runner.RunWith;
  import org.springframework.beans.factory.annotation.Autowired;
  import org.springframework.context.ApplicationContext;
  import org.springframework.test.context.ContextConfiguration;
  import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
  
  import java.util.stream.Stream;
  
  @RunWith(SpringJUnit4ClassRunner.class)
  @ContextConfiguration(classes = ConditionConfiguration.class)
  public class ConditionTest {
  
      @Autowired
      ApplicationContext context;
  
      @Test
      public void test1() {
          String[] beanNamesForType = context.getBeanNamesForType(Person.class);
          Stream.of(beanNamesForType).forEach(System.out::println);
      }
  
      @Test
      public void test2() {
          String[] beanDefinitionNames = context.getBeanDefinitionNames();
          Stream.of(beanDefinitionNames).forEach(System.out::println);
      }
  }
  ```

## @Import导入另一组件

### 容器注入组件

- 包扫描+组件标注注解（`@Controller / @Service / @Repository / @Component`）,局限于我们自己写的类
- `@Bean`[导入的第三方包里面的组件]，xml的bean配置方式也可以做到。 
- `@Import`[快速给容器中导入一个组件]，xml也有对应的引入方式。
  - `@ImportSelector`[导入的选择器,返回需要导入的组件的全类名数组]
  - `@ImportBeanDefinitionRegistrar`[也是一个接口]
- 使用Spring提供的`FactoryBean`
  - 默认获取到的是工厂bean调用`getObject`创建的对象
  - 要获取工厂Bean本身，我们需要给id前面加一个& 如：`&ColorFactoryBean`
  - 这个的特点或者是优势到底是什么？为什么会提供这种方法？

import注解的具体定义及注释

```java
public @interface Import {

	/**
	 * {@link Configuration @Configuration}, {@link ImportSelector},
	 * {@link ImportBeanDefinitionRegistrar}, or regular component classes to import.
	 */
	Class<?>[] value();

}
```

#### Import的基本用法

```java
package org.example.configuration;

import org.example.pojo.Person;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;

@Configuration
@Import(Color.class)
public class ImportConfiguration {

    @Bean
    public Person person() {
        return new Person();
    }
}

// 这个Color什么注解都不用加！太棒了！！
class Color {
}

```

测试代码

```java
// 测试代码
package org.example;

import org.example.configuration.ImportConfiguration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.stream.Stream;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = ImportConfiguration.class)
public class ImportTest {

    @Autowired
    ApplicationContext context;

    @Test
    public void test1() {
        String[] beanDefinitionNames = context.getBeanDefinitionNames();
        Stream.of(beanDefinitionNames).forEach(System.out::println);
    }
}
```

#### Import的高级用法一

- `ImportSelector`

```java

/**
 * Interface to be implemented by types that determine which @{@link Configuration}
 * class(es) should be imported based on a given selection criteria, usually one or
 * more annotation attributes.
 *
 * <p>An {@link ImportSelector} may implement any of the following
 * {@link org.springframework.beans.factory.Aware Aware} interfaces,
 * and their respective methods will be called prior to {@link #selectImports}:
 * <ul>
 * <li>{@link org.springframework.context.EnvironmentAware EnvironmentAware}</li>
 * <li>{@link org.springframework.beans.factory.BeanFactoryAware BeanFactoryAware}</li>
 * <li>{@link org.springframework.beans.factory.BeanClassLoaderAware BeanClassLoaderAware}</li>
 * <li>{@link org.springframework.context.ResourceLoaderAware ResourceLoaderAware}</li>
 * </ul>
 *
 * <p>Alternatively, the class may provide a single constructor with one or more of
 * the following supported parameter types:
 * <ul>
 * <li>{@link org.springframework.core.env.Environment Environment}</li>
 * <li>{@link org.springframework.beans.factory.BeanFactory BeanFactory}</li>
 * <li>{@link java.lang.ClassLoader ClassLoader}</li>
 * <li>{@link org.springframework.core.io.ResourceLoader ResourceLoader}</li>
 * </ul>
 *
 * <p>{@code ImportSelector} implementations are usually processed in the same way
 * as regular {@code @Import} annotations, however, it is also possible to defer
 * selection of imports until all {@code @Configuration} classes have been processed
 * (see {@link DeferredImportSelector} for details).
 *
 * @author Chris Beams
 * @author Juergen Hoeller
 * @since 3.1
 * @see DeferredImportSelector
 * @see Import
 * @see ImportBeanDefinitionRegistrar
 * @see Configuration
 */
public interface ImportSelector {

	/**
	 * Select and return the names of which class(es) should be imported based on
	 * the {@link AnnotationMetadata} of the importing @{@link Configuration} class.
	 * @return the class names, or an empty array if none
	 */
	String[] selectImports(AnnotationMetadata importingClassMetadata);

	/**
	 * Return a predicate for excluding classes from the import candidates, to be
	 * transitively applied to all classes found through this selector's imports.
	 * <p>If this predicate returns {@code true} for a given fully-qualified
	 * class name, said class will not be considered as an imported configuration
	 * class, bypassing class file loading as well as metadata introspection.
	 * @return the filter predicate for fully-qualified candidate class names
	 * of transitively imported configuration classes, or {@code null} if none
	 * @since 5.2.4
	 */
	@Nullable
	default Predicate<String> getExclusionFilter() {
		return null;
	}

}
```



```java
package org.example.configuration;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.context.annotation.ImportSelector;
import org.springframework.core.type.AnnotationMetadata;

import java.util.ArrayList;
import java.util.Set;

@Configuration
@Import({ImportSelectorDemo.class})
public class ImportSelectorConfiguration {
    @Bean
    public Object getObj() {
        return new Object();
    }
}

class ImportSelectorDemo implements ImportSelector {

    /**
     * @param importingClassMetadata 当前标注@Import注解的类的所有注解信息，
     *                               简而言之，可以获取到Import注解和其他注解的信息
     * @return 要导入到组件的全类名
     */
    @Override
    public String[] selectImports(AnnotationMetadata importingClassMetadata) {
        Set<String> annotationTypes = importingClassMetadata.getAnnotationTypes();
        ArrayList<String> list = new ArrayList<String>();
        String[] strings = new String[annotationTypes.size()];
        for (int i = 0; i < list.size(); i++) {
            strings[i] = list.get(i);
        }
        return strings;
    }

}

class SelectorClassDemo {

}

```

```java
// 单元测试代码
package org.example;

import org.example.configuration.ImportSelectorConfiguration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.stream.Stream;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = ImportSelectorConfiguration.class)
public class ImportSelectorTest {

    @Autowired
    ApplicationContext context;

    @Test
    public void test1() {
        String[] beanDefinitionNames = context.getBeanDefinitionNames();
        Stream.of(beanDefinitionNames).forEach(System.out::println);
    }
}
```

#### Import的高级用法二

- ImportBeanDefinitionRegistrar接口

```java
public interface ImportBeanDefinitionRegistrar {

	/**
	 * Register bean definitions as necessary based on the given annotation metadata of
	 * the importing {@code @Configuration} class.
	 * <p>Note that {@link BeanDefinitionRegistryPostProcessor} types may <em>not</em> be
	 * registered here, due to lifecycle constraints related to {@code @Configuration}
	 * class processing.
	 * <p>The default implementation delegates to
	 * {@link #registerBeanDefinitions(AnnotationMetadata, BeanDefinitionRegistry)}.
	 * @param importingClassMetadata annotation metadata of the importing class
	 * @param registry current bean definition registry
	 * @param importBeanNameGenerator the bean name generator strategy for imported beans:
	 * {@link ConfigurationClassPostProcessor#IMPORT_BEAN_NAME_GENERATOR} by default, or a
	 * user-provided one if {@link ConfigurationClassPostProcessor#setBeanNameGenerator}
	 * has been set. In the latter case, the passed-in strategy will be the same used for
	 * component scanning in the containing application context (otherwise, the default
	 * component-scan naming strategy is {@link AnnotationBeanNameGenerator#INSTANCE}).
	 * @since 5.2
	 * @see ConfigurationClassPostProcessor#IMPORT_BEAN_NAME_GENERATOR
	 * @see ConfigurationClassPostProcessor#setBeanNameGenerator
	 */
	default void registerBeanDefinitions(AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry,
			BeanNameGenerator importBeanNameGenerator) {

		registerBeanDefinitions(importingClassMetadata, registry);
	}

	/**
	 * Register bean definitions as necessary based on the given annotation metadata of
	 * the importing {@code @Configuration} class.
	 * <p>Note that {@link BeanDefinitionRegistryPostProcessor} types may <em>not</em> be
	 * registered here, due to lifecycle constraints related to {@code @Configuration}
	 * class processing.
	 * <p>The default implementation is empty.
	 * @param importingClassMetadata annotation metadata of the importing class
	 * @param registry current bean definition registry
	 */
    /**
    通过调这个方法，给容器自己添加一些组件
    AnnotationMetadata 是当前类的注解信息
    BeanDefinitionRegistry Bean定义的注册类，通过它给容器注册Bean
    */
    
	default void registerBeanDefinitions(AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry) {
	}

}
```

如果存在xxbean，就把oobean注册进去

- JavaConfig代码

```java
package org.example.configuration;

import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.beans.factory.support.RootBeanDefinition;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Import;
import org.springframework.context.annotation.ImportBeanDefinitionRegistrar;
import org.springframework.core.type.AnnotationMetadata;

@Configuration
@Import({ImportBeanDefinitionDemo.class})
public class ImportBeanDefinitionRegistrarConfiguration {

    @Bean("red")
    public Red red() {
        return new Red();
    }

}

class ImportBeanDefinitionDemo implements ImportBeanDefinitionRegistrar {
    @Override
    public void registerBeanDefinitions(AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry) {
        boolean red = registry.containsBeanDefinition("red");
        if (red) {
            RootBeanDefinition rootBeanDefinition = new RootBeanDefinition(Rain.class);
            registry.registerBeanDefinition("Rain", rootBeanDefinition);
        }
    }
}

class Red {

}

class Rain {

}
```

- 测试代码

```java
package org.example;

import org.example.configuration.ImportBeanDefinitionRegistrarConfiguration;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.ApplicationContext;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;

import java.util.stream.Stream;

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = ImportBeanDefinitionRegistrarConfiguration.class)
public class ImportBeanDefinitionTest {
    @Autowired
    ApplicationContext context;

    @Test
    public void test1() {
        String[] beanDefinitionNames = context.getBeanDefinitionNames();
        Stream.of(beanDefinitionNames).forEach(System.out::println);
    }
}
```

## FactoryBean创建

使用Spring提供的`FactoryBean`

- 默认获取到的是工厂bean调用`getObject`创建的对象
- 要获取工厂Bean本身，我们需要给id前面加一个& 如：`&ColorFactoryBean`
- 这个的特点或者是优势到底是什么？为什么会提供这种方法？

代码

```java
package org.example.configuration;

import org.example.pojo.Person;
import org.springframework.beans.factory.FactoryBean;

public class FactoryBeanConfiguration implements FactoryBean<Person> {
    @Override
    public Person getObject() throws Exception {
        return new Person();
    }

    @Override
    public Class<?> getObjectType() {
        return Person.class;
    }

    /**
     * @return true 单实例 容器中只保存一份
     * false 多实例 每次调用创建新对象
     */
    @Override
    public boolean isSingleton() {
        return false;
    }
}
```

测试代码

```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = FactoryBeanConfiguration.class)
public class FactoryBeanTest {
    @Autowired
    ApplicationContext context;

    @Test
    public void test1() {
        Object factoryBeanDemo = context.getBean("factoryBeanDemo");
        System.out.println(factoryBeanDemo.getClass());
        // 加上&符号 获取的是工厂对象 而非getObject返回的Bean
        Object bean = context.getBean("&factoryBeanDemo");
        System.out.println(bean.getClass());
    }
}
```

解释，请看源码

```java
public interface BeanFactory {

	/**
	 * Used to dereference a {@link FactoryBean} instance and distinguish it from
	 * beans <i>created</i> by the FactoryBean. For example, if the bean named
	 * {@code myJndiObject} is a FactoryBean, getting {@code &myJndiObject}
	 * will return the factory, not the instance returned by the factory.
	 */
	String FACTORY_BEAN_PREFIX = "&";
}
```

# Spring注解--生命周期

## Bean指定初始化和销毁方法

- `@Bean(initMethod = "init", destroyMethod = "destroy")`

- 原本在xml中的配置方式

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:context="http://www.springframework.org/schema/context"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd http://www.springframework.org/schema/context https://www.springframework.org/schema/context/spring-context.xsd">
    <bean id="demo" class="org.example.pojo.Person" init-method="toString" destroy-method="toString"></bean>
    <!-- 配置包扫描 -->
    <context:component-scan base-package="org.example"></context:component-scan>
</beans>
```

- pojo类

```java
package org.example.pojo;

public class Car {
    public Car() {
        System.out.println("Car constructor...");
    }

    public void init() {
        System.out.println("car ... init");
    }

    public void destroy() {
        System.out.println("car ... destroy");
    }
}

```

- JavaConfig

```java
package org.example.configuration.lifecycle;

import org.example.pojo.Car;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

@Configuration
public class LifeCycleConfiguration {

    // 可以在自定义数据源，用init和destroy进行数据源的初始化和关闭
    @Scope("prototype")
    @Bean(initMethod = "init", destroyMethod = "destroy")
    public Car car() {
        return new Car();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext annotationConfigApplicationContext = new AnnotationConfigApplicationContext(LifeCycleConfiguration.class);
//        Object car = annotationConfigApplicationContext.getBean("car");
//        annotationConfigApplicationContext.close();
        // 多实例的bean在获取时才创建对象
    }
}
```

## Bean实现接口

通过实现接口，自定义初始化和销毁的逻辑

```java
package org.example.pojo;

import org.springframework.beans.factory.DisposableBean;
import org.springframework.beans.factory.InitializingBean;

public class Car2 implements InitializingBean, DisposableBean {

    @Override
    public void destroy() throws Exception {
        System.out.println("car2 destroy");
    }

    @Override
    public void afterPropertiesSet() throws Exception {
        System.out.println("car2 init afterPropertiesSet");
    }
}

package org.example.configuration.lifecycle;

import org.example.pojo.Car;
import org.example.pojo.Car2;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

@Configuration
public class LifeCycleConfiguration {

    // 可以在自定义数据源，用init和destroy进行数据源的初始化和关闭
    @Scope("prototype")
    @Bean(initMethod = "init", destroyMethod = "destroy")
    public Car car() {
        return new Car();
    }

    @Bean
    public Car2 car2(){
        return new Car2();
    }
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(LifeCycleConfiguration.class);
        Object car2 = context.getBean("car2");

//        Object car = annotationConfigApplicationContext.getBean("car");
//        annotationConfigApplicationContext.close();
        // 多实例的bean在获取时才创建对象
    }
}

```

## JS250规范定义的注解

- `@PostConstruct`,在bean创建完成并属性赋值完成，来执行初始化方法
- `@PreDestroy`，在容器销毁bean之前通知我们进行清理操作
- 这几个注解是Java提供的，好像是需要提供J2EE的依赖。

```java
package org.example.pojo;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

public class Car3 {
    public Car3() {
        System.out.println("car3 construct");
    }

    @PostConstruct
    public void init() {
        System.out.println("PostConstruct");
    }

    @PreDestroy
    public void destroy() {
        System.out.println("PreDestroy");
    }

}
```

test code

```java
package org.example.configuration.lifecycle;

import org.example.pojo.Car;
import org.example.pojo.Car2;
import org.example.pojo.Car3;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

@Configuration
public class LifeCycleConfiguration {

    // 可以在自定义数据源，用init和destroy进行数据源的初始化和关闭
    @Scope("prototype")
    @Bean(initMethod = "init", destroyMethod = "destroy")
    public Car car() {
        return new Car();
    }

    @Bean
    public Car2 car2(){
        return new Car2();
    }

    @Bean Car3 car3(){
        return new Car3();
    }
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(LifeCycleConfiguration.class);
        Object car2 = context.getBean("car2");
        Object car3 = context.getBean("car3");
        context.close();
//        Object car = annotationConfigApplicationContext.getBean("car");
//        annotationConfigApplicationContext.close();
        // 多实例的bean在获取时才创建对象
    }
}
```

## Bean后置处理器

- `interface BeanPostProcessor`：bean的后置处理器；在bean初始化前后进行一些处理工作
  - `postProcessBeforeInitialization`：初始化之前工作
  - `postProcessAfterInitialization`：初始化之后工作

### 用法

源码注释如下

```java
public interface BeanPostProcessor {

	/**
	 * Apply this {@code BeanPostProcessor} to the given new bean instance <i>before</i> any bean
	 * initialization callbacks (like InitializingBean's {@code afterPropertiesSet} 
	 * or a custom init-method). The bean will already be populated with property values.
	 * The returned bean instance may be a wrapper around the original.
	 * <p>The default implementation returns the given {@code bean} as-is.
	 * @param bean the new bean instance
	 * @param beanName the name of the bean
	 * @return the bean instance to use, either the original or a wrapped one;
	 * if {@code null}, no subsequent BeanPostProcessors will be invoked
	 * @throws org.springframework.beans.BeansException in case of errors
	 * @see org.springframework.beans.factory.InitializingBean#afterPropertiesSet
	 */
	@Nullable
	default Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException {
		return bean;
	}

	/**
	 * Apply this {@code BeanPostProcessor} to the given new bean instance <i>after</i> any bean
	 * initialization callbacks (like InitializingBean's {@code afterPropertiesSet}
	 * or a custom init-method). The bean will already be populated with property values.
	 * The returned bean instance may be a wrapper around the original.
	 * <p>In case of a FactoryBean, this callback will be invoked for both the FactoryBean
	 * instance and the objects created by the FactoryBean (as of Spring 2.0). The
	 * post-processor can decide whether to apply to either the FactoryBean or created
	 * objects or both through corresponding {@code bean instanceof FactoryBean} checks.
	 * <p>This callback will also be invoked after a short-circuiting triggered by a
	 * {@link InstantiationAwareBeanPostProcessor#postProcessBeforeInstantiation} method,
	 * in contrast to all other {@code BeanPostProcessor} callbacks.
	 * <p>The default implementation returns the given {@code bean} as-is.
	 * @param bean the new bean instance
	 * @param beanName the name of the bean
	 * @return the bean instance to use, either the original or a wrapped one;
	 * if {@code null}, no subsequent BeanPostProcessors will be invoked
	 * @throws org.springframework.beans.BeansException in case of errors
	 * @see org.springframework.beans.factory.InitializingBean#afterPropertiesSet
	 * @see org.springframework.beans.factory.FactoryBean
	 */
	@Nullable
	default Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException {
		return bean;
	}

}
```

我测试了一下，`@Configuration 的@Bean注解注册的Bean，用下面 实现 接口的方式 无效`

```java
package org.example.configuration.lifecycle;

import org.springframework.beans.BeansException;
import org.springframework.beans.factory.config.BeanPostProcessor;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;

@Configuration
@ComponentScan(basePackages = "org.example.configuration.lifecycle")
public class BeanPostProcessConfiguration {

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(BeanPostProcessConfiguration.class);
        context.close();
    }
}

@Component
class BeanPostProcessDemo implements BeanPostProcessor {
    public BeanPostProcessDemo() {
        System.out.println("construct");
    }

    @Override
    public Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException {
        System.out.println("before initialization");
        return bean;
    }

    @Override
    public Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException {
        System.out.println("after initialization");
        return bean;
    }

}
```



### 原理

原理那两个视频没看，记得补 `P16&17`

```java
 遍历得到容器中所有的BeanPostProcessor；挨个执行beforeInitialization
 一旦返回null，跳出for循环，不会执行后面的BeanPostProcess.postProcessors
 BeanPostProcessor的大致执行流程
 populateBean(beanName, mbd, instanceWrapper);给bean进行属性赋值
 initializeBean{
 applyBeanPostProcessorsBeforeInitialization//for循环得到全部beanPost
 invokeInitMethods(beanName, wrappedBean, mbd);//初始化方法
 applyBeanPostProcessorsAfterInitialization//for循环得到全部beanPost
}
```

# Spring注解--属性赋值

## `@Value`

Value的用法，请看源码注释！这个注解还可作用于字段上。

```java
@Target({ElementType.FIELD, ElementType.METHOD, ElementType.PARAMETER, ElementType.ANNOTATION_TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface Value {

	/**
	 * The actual value expression such as <code>#{systemProperties.myProp}</code>
	 * or property placeholder such as <code>${my.app.myProp}</code>.
	 */
	String value();

}
```

----

使用`@Value`赋值

- 基本数值
- 可以写SpEL；#{}
- 可以写${}; 取出配置文件中的值（在运行环境变量里面的值）；**properties配置文件，放在resource目录下！！**

- pojo对象

  ```java
  package org.example.pojo;
  
  import org.springframework.beans.factory.annotation.Value;
  
  public class Person {
      // 使用@Value赋值
      // 1 基本数值
      // 2 可以写SpEL， #{}，取出配置文件中的值
      @Value("张三")
      private String name;
      @Value("#{20-5}")
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
      public String toString() {
          return "Person{" +
                  "name='" + name + '\'' +
                  ", age=" + age +
                  '}';
      }
  }
  
  ```

- JavaConfig

  ```java
  package org.example.configuration.assign;
  
  import org.example.pojo.Person;
  import org.springframework.context.annotation.AnnotationConfigApplicationContext;
  import org.springframework.context.annotation.Bean;
  import org.springframework.context.annotation.Configuration;
  
  @Configuration
  public class ValueConfig {
      @Bean
      public Person person() {
          return new Person();
      }
  
      public static void main(String[] args) {
          AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(ValueConfig.class);
          Person person = (Person) context.getBean("person");
          System.out.println(person);
      }
  }
  // output  Person{name='张三', age=15} 赋值成功
  ```

  

## `@propertySource`

- properties配置文件，在resource根目录下哦

  ```properties
  person.name=zhangsan
  ```

为什么是在根目录下？请看该注解的注释！！

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Repeatable(PropertySources.class)
public @interface PropertySource {

	/**
	 * Indicate the name of this property source. If omitted, the {@link #factory}
	 * will generate a name based on the underlying resource (in the case of
	 * {@link org.springframework.core.io.support.DefaultPropertySourceFactory}:
	 * derived from the resource description through a corresponding name-less
	 * {@link org.springframework.core.io.support.ResourcePropertySource} constructor).
	 * @see org.springframework.core.env.PropertySource#getName()
	 * @see org.springframework.core.io.Resource#getDescription()
	 */
	String name() default "";

	/**
	 * Indicate the resource location(s) of the properties file to be loaded.
	 * <p>Both traditional and XML-based properties file formats are supported
	 * &mdash; for example, {@code "classpath:/com/myco/app.properties"}
	 * or {@code "file:/path/to/file.xml"}.
	 * <p>Resource location wildcards (e.g. *&#42;/*.properties) are not permitted;
	 * each location must evaluate to exactly one {@code .properties} or {@code .xml}
	 * resource.
	 * <p>${...} placeholders will be resolved against any/all property sources already
	 * registered with the {@code Environment}. See {@linkplain PropertySource above}
	 * for examples.
	 * <p>Each location will be added to the enclosing {@code Environment} as its own
	 * property source, and in the order declared.
	 */
	String[] value();

	/**
	 * Indicate if a failure to find a {@link #value property resource} should be
	 * ignored.
	 * <p>{@code true} is appropriate if the properties file is completely optional.
	 * <p>Default is {@code false}.
	 * @since 4.0
	 */
	boolean ignoreResourceNotFound() default false;

	/**
	 * A specific character encoding for the given resources, e.g. "UTF-8".
	 * @since 4.3
	 */
	String encoding() default "";

	/**
	 * Specify a custom {@link PropertySourceFactory}, if any.
	 * <p>By default, a default factory for standard resource files will be used.
	 * @since 4.3
	 * @see org.springframework.core.io.support.DefaultPropertySourceFactory
	 * @see org.springframework.core.io.support.ResourcePropertySource
	 */
	Class<? extends PropertySourceFactory> factory() default PropertySourceFactory.class;

}
```

----

- pojo

  ```java
  package org.example.pojo;
  
  import org.springframework.beans.factory.annotation.Value;
  
  public class Person {
      // 使用@Value赋值
      // 1 基本数值
      // 2 可以写SpEL， #{}，取出配置文件中的值
      @Value("${person.name}")
      private String name;
      @Value("#{20-5}")
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
      public String toString() {
          return "Person{" +
                  "name='" + name + '\'' +
                  ", age=" + age +
                  '}';
      }
  }
  ```

- JavaConfig

  ```java
  package org.example.configuration.assign;
  
  import org.example.pojo.Person;
  import org.springframework.context.annotation.AnnotationConfigApplicationContext;
  import org.springframework.context.annotation.Bean;
  import org.springframework.context.annotation.Configuration;
  import org.springframework.context.annotation.PropertySource;
  
  @Configuration
  // 使用@PropertySource读取外部配置文件中的k/v保存到运行的环境中
  @PropertySource(value = {"classpath:/person.properties"})
  public class PropertySourceConfig {
  
      @Bean
      public Person person() {
          return new Person();
      }
  
      public static void main(String[] args) {
          AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(PropertySourceConfig.class);
          ConfigurableEnvironment environment = context.getEnvironment();
          System.out.println(environment.getProperty("person.name"));
          Person person = context.getBean(Person.class, "person");
          System.out.println(person);
      }
  }
  ```

# Spring注解--自动装配

## 自动装配概述

**Spring利用依赖注入（DI），完成对IOC容器中各个组件的依赖关系赋值；**

* 1、`@AutoWired`：自动注入【Spring定义的】
    * 默认按照类型去容器中找对应的组件 `applicationContext.getBean(BookService.class)`，找到就赋值
    * 如果找到相同类型的组件，再将属性的名称作为组件的id去容器中查找      `applicationContext.getBean("bookDao")`
* 2、`@Qualifier("bookDao")`：使用该注解来指定需要装配的组件的id，而不是使用属性名
* 3、自动装配默认一定要将属性赋值好，没有就会报错，可通过在`Autowire`的注解中将required=false来使该配置设置为非必需
* 4、`@Primary`：让Spring进行自动装配的时候，默认使用首选的bean,也可以继续使用@Qualifier来指定需要装配的bean

**Spring还支持使用@Resource（JSR250）和@Inject（JSR330）【java规范】**

 * 1、@Resource：
     *              可以和@Autowired一样实现自动装配功能；默认是按照组件名称进行装配的；没有能支持@Primary的功能以及@Autowired（required=false）的功能
 * 2、@Inject（需要导入依赖）：
     *              导入javax.inject的包，和Autowired的功能一样，没有required=false的功能

## `@Autowired`

<span style="color:red">先按类型来，找到就赋值；如果找到相同类型的组件，再将属性名作为组件的id去容器中查找。</span>

<span style="color:red">以前常见的一个错误，如果是按接口注入，找到了很多相同类型的组件，且属性名查找失败，则会提示`NoUniqueBeanDefinitionException`</span>

- `@Autowired`
- `@Autowired(required=false)` 能装配上就装，不能就不装

```java
@Configuration
@ComponentScan(basePackages = "org.example")
public class AutowiredConfig {

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AutowiredConfig.class);
        Book book = context.getBean(Book.class, "book");
        System.out.println("book's person name is " + book.person);
    }
}

@Service
class Book {
    @Autowired
    public Person person;
}
```

## `@Qualifier`

与@Autowired结合，指定装配什么名称的Bean

## @Primary

首选的，主要的注解

让Spring进行自动装配时，默认使用首选的Bean

```java
@Configuration
@ComponentScan(basePackages = "org.example")
public class AutowiredConfig {

    @Bean("p1")
    public Person person() {
        return new Person("1");
    }

    @Bean("p2")
    @Primary // 首选装配这个bean
    public Person person2() {
        return new Person("2");
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AutowiredConfig.class);
        Books book = context.getBean(Books.class, "book");
        System.out.println("book's person name is " + book.person);
    }
}

@Service("book")
class Books {
    @Autowired(required = false)
    public Person person;
}
```

## JSR250 @Resource

@Resource是Java规范。

@Resource(name="p1")

## JSR330 @Inject

@Inject是Java规范

@Inject Autowired的功能一样，没有required=false的功能，支持@Primary，但是没有required=false的功能

## @Autowired还是JSRxxx

JSRxx是会被其他IOC框架支持的，使用JSR的，脱离了Spring，换其他IOC框架也可。（JSR是规范！！）

## 自动装配功能原理

`AutowiredAnnotationBeanPostProcessor`解析完成自动装配功能

- AutowiredAnnotationBeanProcessor类

```java
public class AutowiredAnnotationBeanPostProcessor implements SmartInstantiationAwareBeanPostProcessor,
		MergedBeanDefinitionPostProcessor, PriorityOrdered, BeanFactoryAware {
  	public AutowiredAnnotationBeanPostProcessor() {
		this.autowiredAnnotationTypes.add(Autowired.class);
		this.autowiredAnnotationTypes.add(Value.class);
		try {
			this.autowiredAnnotationTypes.add((Class<? extends Annotation>)
					ClassUtils.forName("javax.inject.Inject", AutowiredAnnotationBeanPostProcessor.class.getClassLoader()));
			logger.trace("JSR-330 'javax.inject.Inject' annotation found and supported for autowiring");
		}
		catch (ClassNotFoundException ex) {
			// JSR-330 API not available - simply skip.
		}
	}
}
```

## 方法、构造器位置

### 方法

`@Autowired`：构造器，参数，方法，属性

- 1）**标注在方法位置：**标注在方法，Spring容器创建当前对象，就会调用方法，完成赋值，方法使用的参数，自定义类型的值从ioc容器中获取,@Bean标注的方法创建对象的时侯，方法参数的值默认从ioc容器中获取，默认不写Autowired，效果是一样的。

 *        2）**标注在构造器位置：**默认加在ioc容器中的组件，容器启动会调用无参构造器创建对象，再进行初始化赋值等操作。标注在构造器上可以默认调用该方法，方法中用的参数同样从IOC容器中获取，如果容器只有一个有参构造器，这个有参构造器的Autowired可以省略，参数位置的组件还是可以自动从容器中获取
 *        3）**标注在参数位置：**从ioc容器中获取参数组件的值

### 构造器

@Component注解。

默认再加载ioc容器中的组件，容器启动会调用无参构造器创建对象，再进行初始化赋值等操作。

如果当前类只有一个有参构造器，那么Autowired是可以省略的。@Bean注入，若只有一个有参构造则也是可以省略的。

```java
@Component
public class Boss{
    private Car car;
    
    public Boss(@Autowired Car car){
        this.car = car;
    }
}

// 等价于
@Component
public class Boss{
    private Car car;
    
    public Boss(Car car){
        this.car = car;
    }
}
```

P23 Spring注解驱动

## Aware注入Spring底层原理

### 概述

自定义组件想要使用Spring容器底层的一些组件（ApplicationContext，BeanFactory，xxx）

只需要让自定义组件实现 xxxAware接口：在创建对象的时候，会调用接口规定的方法注入相关组件

基本Demo

```java
package org.example.configuration.automatically;

import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;


@Configuration
@ComponentScan(basePackages = "org.example.configuration.automatically")
public class AwareConfig {

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AwareConfig.class);
        Dog bean = context.getBean(Dog.class);
//        context.close();
    }

}

@Component
class Dog implements ApplicationContextAware {


    private ApplicationContext context;

    /**
     * 再IOC创建启动dog对象时，这个方法会被调用
     *
     * @param context
     * @throws BeansException
     */
    @Override
    public void setApplicationContext(ApplicationContext context) throws BeansException {
        this.context = context;
        System.out.println("context hashcode is" + context);
    }

    public Dog() {
        System.out.println("class dog construct");
    }

    @PostConstruct
    public void init() {
        System.out.println("class dog post construct");
    }

    @PreDestroy
    public void destroy() {
        System.out.println("class dog destroy construct");
    }
}
```

### 常用的Aware接口

- 1、`ApplicationContextAware` 设置ApplicationContext对象
- 2、`BeanNameAware` 设置BeanName
- 3、`EmbeddedValueResolverAware` 解析字符用
- 4、字符串解析，如解析`#{} ${}`，表达式解析？【占位符解析】
    - `${}`取出环境变量中的值。`#{}`Spring的表达式语言
- 使用`xxxProcessor`进行处理的，每个`xxxAware`都有对应的`xxxProcessor`，
    - 利用后置处理器，判断这个Bean。是这个Aware接口，然后把组件传过来。

```java
package org.example.configuration.automatically;

import org.springframework.beans.BeansException;
import org.springframework.beans.factory.BeanNameAware;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.EmbeddedValueResolverAware;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.stereotype.Component;
import org.springframework.util.StringValueResolver;

/**
 * 常用的Aware接口
 */
@Configuration
@ComponentScan(basePackages = "org.example.configuration.automatically")
public class AwareCommonConfig {
    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(AwareCommonConfig.class);
        AwareCommonDemo bean = context.getBean(AwareCommonDemo.class);
        System.out.println(bean);
    }
}

@Component
class AwareCommonDemo implements ApplicationContextAware, BeanNameAware, EmbeddedValueResolverAware {
    private ApplicationContext context;

    @Override
    public void setBeanName(String name) {
        // 设置bean的名字
        System.out.println("setBeanName is " + name);
    }

    @Override
    public void setApplicationContext(ApplicationContext context) throws BeansException {
        this.context = context;
        System.out.println("setApplicationContext " + context);
    }

    @Override
    public void setEmbeddedValueResolver(StringValueResolver resolver) {
        String s = resolver.resolveStringValue("os.name is ${os.name} Spring EL #{12-5}");
        System.out.println("setEmbeddedValueResolver is " + s);
    }
}
```

## Profile注解

### 概述

可根据当前环境，动态激活和切换一系列组件的功能。环境被激活了，才可用。如何激活？【使用命令参数 】

- 开发环境
- 测试环境
- 生产环境

如不同环境的数据库不一样。可用数据源切换达成。

- 写在方法上，在指定环境中方法注入的Bean才生效
- 写在类上，在指定环境中，该类的相关信息和配置信息才生效

```java
@Target({ElementType.TYPE, ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Conditional(ProfileCondition.class)
public @interface Profile {

	/**
	 * The set of profiles for which the annotated component should be registered.
	 */
	String[] value();

}
```

### 数据源切换

- 添加C3P0数据源

```xml
<dependency>
    <groupId>c3p0</groupId>
    <artifactId>c3p0</artifactId>
    <version>0.9.1.2</version>
</dependency>
```

- 注册数据源

```java
package org.example.configuration.automatically;

import com.mchange.v2.c3p0.ComboPooledDataSource;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.EmbeddedValueResolverAware;
import org.springframework.context.annotation.*;
import org.springframework.util.StringValueResolver;

import javax.sql.DataSource;
import java.beans.PropertyVetoException;
import java.util.stream.Stream;

@Configuration
public class ProfileConfig {

    public static void test1() {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(ProfileDemo.class);
        String[] beanNamesForType = context.getBeanNamesForType(DataSource.class);
        Stream.of(beanNamesForType).forEach(System.out::println);
    }

    public static void main(String[] args) {
        test1();
    }
}

@PropertySource("classpath:/dbconfig.properties")
class ProfileDemo implements EmbeddedValueResolverAware {

    private StringValueResolver resolver;

    @Value("${db.user}")
    private String user;
    @Value("${db.password}")
    private String password;
    @Value("${db.driverClass}")
    private String driverClass;

    @Profile("test")
    @Bean("testDataSource")
    public DataSource dataSourceTest() throws PropertyVetoException {
        ComboPooledDataSource dataSource = new ComboPooledDataSource();
        dataSource.setUser(user);
        dataSource.setPassword(password);
        dataSource.setDriverClass(driverClass);
        dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC");
        return dataSource;
    }

    @Profile("dev")
    @Bean("devDataSource")
    public DataSource dataSourceDev() throws PropertyVetoException {
        ComboPooledDataSource dataSource = new ComboPooledDataSource();
        dataSource.setUser(user);
        dataSource.setPassword(password);
        String driverClassString = resolver.resolveStringValue("${db.driverClass}");
        dataSource.setDriverClass(driverClassString);
        dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mybatis05?serverTimezone=UTC");
        return dataSource;
    }

    @Profile("prod")
    @Bean("prodDataSource")
    public DataSource dataSourceProd() throws PropertyVetoException {
        ComboPooledDataSource dataSource = new ComboPooledDataSource();
        dataSource.setUser(user);
        dataSource.setPassword(password);
        dataSource.setDriverClass(driverClass);
        dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mysql_book?serverTimezone=UTC");
        return dataSource;
    }

    @Override
    public void setEmbeddedValueResolver(StringValueResolver resolver) {
        this.resolver = resolver;
    }
}
```

- **激活环境后bean才有效。如何激活？**

- 1、使用命令行动态参数：在虚拟机参数位置加载 `-Dspring.profiles.active=test`

    - IDEA是在`VM options`里面写参数`-Dspring.profiles.active=test`
    - `Eclipse`是在`VM arguments`里面写参数

- 2、使用代码：

    - 以前我使用注解用的是有参构造器

        ```java
        public AnnotationConfigApplicationContext(Class<?>... componentClasses) {
            this();
            register(componentClasses);
            refresh();
        }
        ```

    - 要用代码的方式的话，就不能有有参构造器。**比起有参，它在注册前多了一个设置环境的步骤！！**

    ```java
    package org.example.configuration.automatically;
    
    import com.mchange.v2.c3p0.ComboPooledDataSource;
    import org.springframework.beans.factory.annotation.Value;
    import org.springframework.context.EmbeddedValueResolverAware;
    import org.springframework.context.annotation.*;
    import org.springframework.util.StringValueResolver;
    
    import javax.sql.DataSource;
    import java.beans.PropertyVetoException;
    import java.util.stream.Stream;
    
    public class ProfileConfig {
    
        public static void main(String[] args) {
            AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext();
            context.getEnvironment().setActiveProfiles("test", "prod");
            context.register(ProfileDemo.class);
            context.refresh();
            String[] beanNamesForType = context.getBeanNamesForType(DataSource.class);
            Stream.of(beanNamesForType).forEach(System.out::println);
        }
    }
    
    @PropertySource("classpath:/dbconfig.properties")
    @Configuration
    class ProfileDemo implements EmbeddedValueResolverAware {
    
        private StringValueResolver resolver;
    
        @Value("${db.user}")
        private String user;
        @Value("${db.password}")
        private String password;
        @Value("${db.driverClass}")
        private String driverClass;
    
        @Profile("test")
        @Bean("testDataSource")
        public DataSource dataSourceTest() throws PropertyVetoException {
            ComboPooledDataSource dataSource = new ComboPooledDataSource();
            dataSource.setUser(user);
            dataSource.setPassword(password);
            dataSource.setDriverClass(driverClass);
            dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC");
            return dataSource;
        }
    
        @Profile("dev")
        @Bean("devDataSource")
        public DataSource dataSourceDev() throws PropertyVetoException {
            ComboPooledDataSource dataSource = new ComboPooledDataSource();
            dataSource.setUser(user);
            dataSource.setPassword(password);
            String driverClassString = resolver.resolveStringValue("${db.driverClass}");
            dataSource.setDriverClass(driverClassString);
            dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mybatis05?serverTimezone=UTC");
            return dataSource;
        }
    
        @Profile("prod")
        @Bean("prodDataSource")
        public DataSource dataSourceProd() throws PropertyVetoException {
            ComboPooledDataSource dataSource = new ComboPooledDataSource();
            dataSource.setUser(user);
            dataSource.setPassword(password);
            dataSource.setDriverClass(driverClass);
            dataSource.setJdbcUrl("jdbc:mysql://localhost:3306/mysql_book?serverTimezone=UTC");
            return dataSource;
        }
    
        @Override
        public void setEmbeddedValueResolver(StringValueResolver resolver) {
            this.resolver = resolver;
        }
    }
    ```

# Spring注解--IOC小结

## 容器

- `AnnotationConfigApplicationContext`
- 组件添加
    - @ComponentScan
    - @Bean
    - @Configuration
    - @Component
    - @Service
    - @Controller
    - @Repository
    - @Conditional  ★
    - @Primary
    - @Lazy
    - @Scope 
    - @Import ★
    - `ImportSelector`
    - 工厂模式
- 组件赋值
    - @Value [ ${} 读properties文件  #{} 表达式语言 ]
    - @Autowired
        - @Qualifier
        - 其他方式 [ @Resource (JSR250)  @Inject (JSR330,需要导入 javax.inject) ]
    - @PropertySource
    - @PropertySources
    - @Prifile
        - Environment
        - -Dspring.profiles.active=test
- 组件注入
    - 方法参数 
    - 构造器注入
    - ApplicationContextAware --> ApplicationContextAwareProcessor
    - xxxAware
- AOP
- 声明式事务

## 扩展原理

- `BeanFactoryPostProcessor`
- `BeanDefinitionRegistryPostProcessor`
- `ApplicationListener`
- `Spring容器创建过程`

# Spring注解--AOP

## 概述

- 配置环境

```xml
<!-- aop需要再额外导入 切面包 -->
<dependency>
    <groupId>org.springframework</groupId>
    <artifactId>spring-aspects</artifactId>
    <version>5.3.3</version>
</dependency>
```

- JavaConfig

```java
package org.example.configuration.aop;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.*;
import org.springframework.context.annotation.*;
import org.springframework.stereotype.Component;

import java.util.stream.Stream;

/**
 * AOP[动态代理]
 * 指程序运行期间动态的将某段代码切入到指定位置进行运行的编程方式
 * 1、导入AOP模块：Spring AOP（spring-aspects）
 * 2、定义一个业务逻辑类（MathCalculator），在业务逻辑运行的时候将日志进行打印（方法运行之前，方法运行之后，方法出现异常，xxx）。
 * 3、定义一个日志切面类（logAspects），切面里面的方法需要动态感知    MathCalculator.div运行到了哪里，然后执行。
 * --------通知方法：
 * -----------前置通知(@Before)：logStart 在目标方法（div） 运行之前运行
 * -----------后置通知(@After)：logEbd 在目标方法（div） 运行结束之后运行
 * -----------返回通知(@AfterReturning)：logReturn 在目标方法（div） 正常返回之后
 * -----------异常通知(@AfterThrowing)：logException 在目标方法（div） 出现异常以后运行
 * -----------环绕通知(@Around)：动态代理，手动推进目标方法运行（joinPoint.procced()）
 * 4、给切面类的目标方法标准何时何地运行(通知注解)
 * 5、将切面类和业务逻辑类（目标方法所在类）都加入到容器中
 * 6、必须告诉Spring，那个类是切面类（给切面类加注解）
 * 7、给配置类中加 @EnableAspectJAutoProxy [开启基于注解的AOP模式]
 * 在Spring中 EnableXxx都是开启某项功能的。
 */
@EnableAspectJAutoProxy
@Configuration
public class MainConfigOfAOP {

    @Bean
    public MathCalculator calculator() {
        return new MathCalculator();
    }

    @Bean
    public LogAspects logAspects() {
        return new LogAspects();
    }

    public static void main(String[] args) {
        AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(MainConfigOfAOP.class);
        MathCalculator bean = context.getBean(MathCalculator.class);
        bean.div(2, 3);
    }
}


/**
 * 用AOP做个日志
 */
class MathCalculator {
    public int div(int i, int j) {
        System.out.println("div method");
        return i / j;
    }
}

@Aspect
//告诉Spring容器 当前类是一个切面类
class LogAspects {
    /**
     * 抽取公共的表达式 需要使用execution
     */
    @Pointcut("execution(public int org.example.configuration.aop.MathCalculator.*(..))")
    public void pointCut() {
    }

    // 指定只切入某个方法 @Before("public int org.example.configuration.aop.MathCalculator.div(int,int)")
    // 指定切入该类的所有方法..任意多参数 @Before("public int org.example.configuration.aop.MathCalculator.*(..)")
    @Before("pointCut()")
    public void logStart(JoinPoint joinPoint) {
        System.out.println("log Start " + joinPoint.getSignature());
    }

    @After("pointCut()")
    public void logEnd() {
        System.out.println("log End");
    }

    @AfterReturning("pointCut()")
    public void logRet() {
        System.out.println("log Return");
    }

    @AfterThrowing("pointCut()")
    public void logException() {
        System.out.println("log Exception");
    }
}
```

