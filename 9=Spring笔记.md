## 第一章 Spring之旅

> 概念居多，利于面试！不做笔记，自行看书！

- JavaConfig优于xml。我们更倾向于使用JavaConfig。

> Spring常见的几个用于从容器中获取对象的类

- ```java
  默认xml是在src目录下！
  new AnnotationConfigApplicationContext(Clazz.class)// 传入JavaConfig对象的class
  new ClassPathXmlApplicationContext("knight.xml");//传入xml文件的位置
  ```

**JavaConfig对象的写法如下**

```java
/**
* @Bean的配置是为了这些：
	1.我们需要使用第三方的工具包，但是无法在工具包的类上打注解，所以需要我们采用JavaConfig或xml的配置方式把这些类放到Spring的容器中去。
	2.
*/
@Configuration
@ComponentScanner("基础扫描包")
//或者是@ComponentScanner(basePackages={"基础扫描包","基础扫描包}),如果ComponentScanner中不显式地指定包名的话默认是扫描当前JavaConfig所在的包
public class KingConfig {
    @Bean
    public Knight knight(){
        return new BraveKnight(quest());
    }

    @Bean
    public Quest quest(){
        return new Quest();
    }
}
/**
* @Bean是告诉Spring该对象要注册为Spring应用上下文中的Bean。
* 可也@Bean(name="beanName")
*/
```

**xml写法如下**

```
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:aop="http://www.springframework.org/schema/aop"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd http://www.springframework.org/schema/aop https://www.springframework.org/schema/aop/spring-aop.xsd">
    <bean id="BraveKnight" class="com.spr.day01.demo2.BraveKnight">
        <constructor-arg ref="quest"></constructor-arg>
    </bean>
    <!-- aop样例 暂时不看。 表达式还没看 -->
    <bean id="quest" class="com.spr.day01.demo2.Quest"></bean>
</beans>
```

**测试Spring中的代码**

```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes=JavaConfig.class)
public class TestSpring{
	
}
```



## 第二章 装配Bean

### 2.1 装配的方式

- xml中显示配置
- Java中显示配置
- 隐式的bean发现机制和自动装配

一般推荐 自动装配>JavaConfig>xml

### 2.2 自动化装配Bean

> 实现自动化装配，从以下两个角度出发

- 组件扫描（component scanning）：Spring自动发现上下文创建的beean
- 自动装配（autiwiring）：自动满足bean之间的依赖关系

#### 2.2.1 纯注解配置

```java
// 接口
public interface CompactDisc {
    void play();
}

// 实现类 这个注解是将类注册到IOC容器中可以@Component("beanName") 这样指定bean的名称。不写的话
// 默认是sgtPeppers，类名称首字母小写
@Component 
public class SgtPeppers implements CompactDisc {
    private String title = "I am SgtPeppers title";
    private String artist = "I am SgtPeppers artist";

    @Override
    public void play() {
        System.out.println("Playing " + title + " by " + artist);
    }
}

// JavaConfig配置扫描
@Configuration
@ComponentScan
public class CDPlayConfig {
    /**
     * 写上了这两个注解 会自动扫描其所在包和子包的所有Class。
     * 也可以自行指定基础包
     * @ComponentScan("包名称")
     * @ComponentScan(basePackage="包名称")
     * @ComponentScan(basePackage={"包名称1","包名称2"})
     */
}

// 编写测试类 classes可以用数组，从而指定多个配置文件进行加载
// 如 classes = {xx.class,oo.class}
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes=CDPlayConfig.class)
public class CDPlayTest {
    @Autowired
    private CompactDisc cd;
    @Test
    public void test1(){
        assertNotNull(cd);
    }
}

```



#### 2.2.2 xml配置

```java
// 接口
public interface CompactDisc {
    void play();
}

// 实现类 这个注解是将类注册到IOC容器中可以@Component("beanName") 这样指定bean的名称。不写的话
// 默认是sgtPeppers，类名称首字母小写
@Component 
public class SgtPeppers implements CompactDisc {
    private String title = "I am SgtPeppers title";
    private String artist = "I am SgtPeppers artist";

    @Override
    public void play() {
        System.out.println("Playing " + title + " by " + artist);
    }
}

// 多个文件的用法 
// @ContextConfiguration(locations = {"classpath:spring1.xml","classpath:spring2.xml"}) 
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = "classpath:com/spr/day02/spring.xml")
public class CDPlayTest {
    @Autowired
    private CompactDisc cd;
    @Test
    public void test1(){
        assertNotNull(cd);
    }
}

// xml的配置
<context:component-scan base-package="com.spr.day02"></context:component-scan>
```

#### 2.2.3 PS 

- 可以用@Named替代@Component 只有细微差别

#### 2.2.4 通过注解实现自动装配

- 需要考虑的问题
  - 如何解决歧义
  - 如何自动装配

```java
@Component("cd") // 自行指定bean的名称
public class CDPlayer implements CompactDisc {
    private CompactDisc cop;

    // 通过构造器注入
    @Autowired
    public CDPlayer(CompactDisc cop){
        this.cop = cop;
    }

    // 通过set方法或其他方法注入
    @Autowired
    public void setCop(CompactDisc cop){
        this.cop = cop;
    }

    @Override
    public void play() {
        cop.play();
    }
}


@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes=CDPlayConfig.class)
public class CDPlayTest {
    @Autowired
    private CompactDisc cd;

    // 解决歧义
    @Autowired
    @Qualifier("cd")
    private CompactDisc cds;

    @Test
    public void test1(){
        assertNotNull(cd);
    }

    @Test
    public void test2(){
        cds.play();
    }
}
```

- **解决歧义用：@Autowired @Qualifier("beanName");**

#### 2.2.5 验证自动装配

### 2.3 通过Java代码装配Bean

> 不是所有情况都可以自动化装配的，有时候需要导入第三方库，无法进行自动化装配，只能手动

- 常见的注解
  - @Bean，JavaConfig中配置Bean的，在方法上打的注解
  - @Component 在类上 将类加入容器中
  - @Autowired 自动装配类，用在方法，变量上
  - @Qualifier 指定对应类进行装配，一般和Autowired一起用
  - @Scope 指定Bean的范围，在@Configuration标记的类中使用。和@Bean一起用
    - 研究了一下scope的作用域。默认是单例模式，即scope=”singleton”。另外scope还有prototype、request、session、global session作用域。scope=”prototype”多例。再配置bean的作用域时，它的头文件形式如下：

```java
@Configuration
@ComponentScan
public class CDPlayerConfig {
    @Bean
    @Scope("prototype")
    public CompactDisc sgtPeppers(){
        return new SgtPeppers();
    }

    /**
     * 借助Config中的实现注入
     * 看似是没次都是通过方法调用得到的sgt，其实是Spring发现sgt有Bean注解
     * 对sgt的调用进行了拦截，并确保直接返回该方法所创建的bean，而不是每次
     * 都调用sgt创建新的类
     * bean默认是单例模式
     */
    @Bean("cdPlayer")
    public CDPlayer cdPlayer(){
        return new CDPlayer(sgtPeppers());
    }
    
    /**
     * 这种装配 如果存在重复 无法确定用那个 可以在传入的参数中使用注解 测试的时候应为Bean当初
     * 没注意现在一直报错，以后注意，现在错误也没解决！
     */
    @Bean("cdPlayer")
    public CDPlayer CDPlayer2(@Qualifier("sgtPeppers") CompactDisc cd){
        return new CDPlayer(cd);
    }
}

@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes=CDPlayerConfisg.class)
public class CDPlayerTest {
	// 这个cdPlayer没有指定scope 默认是单例模式
    @Autowired
    @Qualifier("cdPlayer")
    private CDPlayer cd;
    @Test
    public void test1(){
        CompactDisc cop = cd.getCop();
        System.out.println(cd == cd.getCop()); // false 用的多例 所有每次获得的cop不一样
    }
}

```



### 2.4 通过XML配置bean

#### 2.4.1 xml规范，idea自动集成

#### 2.4.2 声明一个简单的bean

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean id="beanDemo" class="com.spr.day04.DemoCreateBean"></bean>
</beans>
```

#### 2.4.3 借助构造器注入初始化bean

- 配置bean及构造函数传入的引用对象

```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(locations = "classpath:com/spr/day04/bean.xml")
public class DemoCreateBeanTest {
    @Autowired
    @Qualifier("beanDemo")
    public DemoCreateBean d1;

    @Autowired
    @Qualifier("beanDemo2")
    public DemoCreateBean d2;

    @Autowired
    @Qualifier("beanDemo3")
    public DemoCreateBean d3;

    @Autowired
    @Qualifier("beanDemo4")
    public DemoCreateBean d4;

    @Test
    public void test1() {
        d1.say();
        System.out.println(d1==d2);
        System.out.println(d2==d3);
        System.out.println(d3==d4);
    }
}
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:c="http://www.springframework.org/schema/c"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean id="beanDemo" class="com.spr.day04.DemoCreateBean"></bean>
    <bean id="stu" class="com.spr.day04.Student"></bean>
    <bean id="beanDemo2" class="com.spr.day04.DemoCreateBean">
        <constructor-arg ref="stu"></constructor-arg>
    </bean>
    <!-- c命名空间 c:构造器参数名称-ref -->
    <bean id="beanDemo3" class="com.spr.day04.DemoCreateBean" c:stu-ref="stu"/>

    <!-- c命名空间 c:_参数索引-ref -->
    <bean id="beanDemo4" class="com.spr.day04.DemoCreateBean" c:_0-ref="stu"/>
</beans>
```

- 字面量的注入

```xml
<!-- 构造函数中字面量的配置 -->
<bean id="beanDemo" class="com.spr.day05.DemoCreateBean" c:_0="title1" c:_1="123"></bean>
<bean id="beanDemo" class="com.spr.day05.DemoCreateBean" c:title="title2" c:artist="123"></bean>
```

