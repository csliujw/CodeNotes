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



- [视频看到](https://www.bilibili.com/video/BV1gW411W7wy?p=7)
- 开始PyTorch