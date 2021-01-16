# SpringBoot笔记Day01
## 创建SpringBoot项目
- IDEA创建项目总是出错，于是直接取官网选好依赖，下载过来，导入到IDEA中。
- 导入后 用maven的Reload All Maven Projects 导入所有的依赖
- 导入后发现下面这个报错
    ```xml
      <plugin>
          <groupId>org.springframework.boot</groupId>
          <artifactId>spring-boot-maven-plugin</artifactId>
      </plugin>  
    ```
  加个对应的版本号就可以了
  ```xml
    <plugin>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-maven-plugin</artifactId>
        <version>2.4.2</version>
    </plugin>  
  ```
- resources下的文件最终都会被部署到classpath文件下

## 一些代码和配置信息的说明
- 启动类说明
```java
@SpringBootApplication // 标明这是一个引导类。也可以不在这里的，没事。 
@MapperScan("cn.baobaoxuxu.community.mapper") // 配置MyBatis的Mapper扫描
public class CommunityApplication {
    // 程序的启动入口。
    public static void main(String[] args) {
        SpringApplication.run(CommunityApplication.class, args);
    }
}
```
- 热部署(IDEA进行SpringBoot热部署失败的原因是，IDEA默认情况下不会自动编译，需要对IDEA进行自动编译的设置)
    - Settings -->Compiler
    - Ctrl + Shift + Alt + / -->选择Registry-->compiler.automake.allow.when.app.running ✔
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-devtools</artifactId>
    <scope>runtime</scope>
    <optional>true</optional>
</dependency>
```

## SpringBoot原理分析
- 父坐标parent
```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.4.2</version>
    <relativePath/> <!-- lookup parent from repository -->
</parent>
```
- parent的父坐标dependencies，这里面都是当前版本的SpringBoot，所集成的一些依赖
```xml
  <parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-dependencies</artifactId>
    <version>2.4.2</version>
  </parent>
```

## SpringBoot自动配置
- 注解的层次关系，从最开始指定启动类那个@SpringBootApplication开始.
- @SpringBootApplication上有这些注解
    - @SpringBootConfiguration
        - @Configuration,不就是Spring的那个注解吗？就是指定了他是配置类！！
    - @EnableAutoConfiguration,开启自动配置
        - @AutoConfigurationPackage
        - @Import(AutoConfigurationImportSelector.class), Import的作用是当前配置文件引入其他配置类
            - AutoConfigurationImportSelector这个类很重要。单独拎出来讲！
    - @ComponentScan,组件扫描。有这个注解的，会以加了这个注解的类所在的包为基础路径，进行类的扫描。@SpringBootApplication类上打了@ComponentScan注解，相当于@SpringBootApplication也有这个扫描的功能。
    
```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(excludeFilters = { @Filter(type = FilterType.CUSTOM, classes = TypeExcludeFilter.class),
      @Filter(type = FilterType.CUSTOM, classes = AutoConfigurationExcludeFilter.class) })
public @interface SpringBootApplication {
}
```
```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@AutoConfigurationPackage
@Import(AutoConfigurationImportSelector.class)
public @interface EnableAutoConfiguration {
}
```
```java
public class AutoConfigurationImportSelector implements DeferredImportSelector, BeanClassLoaderAware,
      ResourceLoaderAware, BeanFactoryAware, EnvironmentAware, Ordered {

   @Override
   public String[] selectImports(AnnotationMetadata annotationMetadata) {
      if (!isEnabled(annotationMetadata)) {
         return NO_IMPORTS;
      }
      AutoConfigurationEntry autoConfigurationEntry = getAutoConfigurationEntry(annotationMetadata);
      return StringUtils.toStringArray(autoConfigurationEntry.getConfigurations());
   }

    // getAutoConfigurationEntry方法
   protected AutoConfigurationEntry getAutoConfigurationEntry(AnnotationMetadata annotationMetadata) {
      if (!isEnabled(annotationMetadata)) {
         return EMPTY_ENTRY;
      }
      AnnotationAttributes attributes = getAttributes(annotationMetadata);
        // 加载某些配置。应该是一个全包名。
      List<String> configurations = getCandidateConfigurations(annotationMetadata, attributes);
      configurations = removeDuplicates(configurations);
      Set<String> exclusions = getExclusions(annotationMetadata, attributes);
      checkExcludedClasses(configurations, exclusions);
      configurations.removeAll(exclusions);
      configurations = getConfigurationClassFilter().filter(configurations);
      fireAutoConfigurationImportEvents(configurations, exclusions);
      return new AutoConfigurationEntry(configurations, exclusions);
   }

    // getCandidateConfigurations
   protected List<String> getCandidateConfigurations(AnnotationMetadata metadata, AnnotationAttributes attributes) {
      List<String> configurations = SpringFactoriesLoader.loadFactoryNames(getSpringFactoriesLoaderFactoryClass(),
            getBeanClassLoader());
        // META-INF/spring.factories  一般这个META-INF是当前类所在的那个jar包的META-INF下
        // 我们去看META-INF下的这个类org.springframework.boot.autoconfigure.web.servlet.ServletWebServerFactoryAutoConfiguration,\
      // 找这个类的注解 @EnableConfigurationProperties(ServerProperties.class)
        // 点进这个类 ServerProperties
        // 定位到 @ConfigurationProperties(prefix = "server", ignoreUnknownFields = true)
        // public class ServerProperties { }
        // 结论 去 spring-configuration-metadata.json里找配置信息！
        Assert.notEmpty(configurations, "No auto configuration classes found in META-INF/spring.factories. If you "
            + "are using a custom packaging, make sure that file is correct.");
      return configurations;
   }

}
```

## 配置信息的加载顺序
- 先加载yml
- 再加载yaml
- 再加载properties
- 后加载的覆盖先加载的哦~
```xml
  <resource>
    <directory>${basedir}/src/main/resources</directory>
    <filtering>true</filtering>
    <includes>
      <include>**/application*.yml</include>
      <include>**/application*.yaml</include>
      <include>**/application*.properties</include>
    </includes>
  </resource>
```

## yml语法
通过空格隔开key和value的
```yaml
# 普通数据配置
name: zhangsan

# 对象配置
person:
    name: zhangsan
    age: 13

# 行内对象配置
person2: {name: zhangsan}

# 配置数据、集合
city:
    - beijing
    - tianjin
    - chongqing

city2: [beijing,tianjin]

# 配置数据、集合（对象数据）
student:
    - name: tom
      age: 18
      addr: beijing
    - name: lucy
      age: 19
      addr: nanchang
student2: [{name: tom,age: 18},{name: tom2,age: 18}] 

# map配置
map:
    key1: value1
    key2: value2
```

## 获得yml的配置数据
- 使用@Value映射
```java
package cn.baobaoxuxu.community.controller;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController 
@ConfigurationProperties(prefix = "person") // 加了这个配置文件，会自动从配置文件中，把person对应的属性字段赋值完毕。
                                            // 字段要一致才能匹配成功~~
public class YamlController {
    // Spring的表达式语言
    
    @Value("${name}")
    private String name;

    @GetMapping("/name")
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setPersonName(String personName) {
        this.personName = personName;
    }
}
```

## SpringBoot配置MyBatis
- Mapper接口上要加Mapper注解。或者应用启动哪里加个@MapperScan("mapper的文件名")
```properties
server.port=8080
# 设置当前web应用的名称
server.servlet.context-path=/community
# jdbc相关配置
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.datasource.url=jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC
spring.datasource.username=root
spring.datasource.password=root
# 配置mybatis信息
mybatis.mapper-locations=classpath:mapper/*Mapper.xml
mybatis.type-aliases-package=cn.baobaoxuxu.community.pojo
# 开启驼峰命名
mybatis.configuration.map-underscore-to-camel-case=true 
```

## SpringBoot集成JUnit
- 导入测试依赖
```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-test</artifactId>
</dependency>
```
别导错包哦！
```java
@RunWith(SpringRunner.class) // 固定写法
@SpringBootTest(classes = CommunityApplication.class) // 启动类的class字节码
public class MyBatisTest {
    @Autowired
    private RoleMapper mapper;

    @Test
    public void test() {
        List<Role> all = mapper.findAll();
        Assert.assertNotNull(all);
    }
}
```

## SpringBoot集成JPA

```xml
<!--jdk9需要导入这种依赖-->
<dependency>
    <groupId>javax.xml.bind</groupId>
    <artifactId>jaxb-api</artifactId>
    <version>2.3.0</version>
</dependency>

<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-jpa</artifactId>
</dependency>

<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <scope>runtime</scope>
</dependency>
```

```properties
server.port=8080
# 设置当前web应用的名称
server.servlet.context-path=/community
# jdbc相关配置
spring.datasource.driver-class-name=com.mysql.cj.jdbc.Driver
spring.datasource.url=jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC
spring.datasource.username=root
spring.datasource.password=root

# jpa信息配置
spring.jpa.database=MySQL
spring.jpa.show-sql=true
spring.jpa.generate-ddl=true
spring.jpa.hibernate.ddl-auto=update
spring.jpa.hibernate.naming_strategy=org.hibernate.cfg.ImprovedNamingStrategy
```

```java
package cn.baobaoxuxu.community.pojo;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Entity
public class Role {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;
    private String roleName;
    private String roleDesc;

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getRoleName() {
        return roleName;
    }

    public void setRoleName(String roleName) {
        this.roleName = roleName;
    }

    public String getRoleDesc() {
        return roleDesc;
    }

    public void setRoleDesc(String roleDesc) {
        this.roleDesc = roleDesc;
    }

    @Override
    public String toString() {
        return "Role{" +
                "id=" + id +
                ", roleName='" + roleName + '\'' +
                ", roleDesc='" + roleDesc + '\'' +
                '}';
    }
}
```

```java
package cn.baobaoxuxu.community.repository;

import cn.baobaoxuxu.community.pojo.Role;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface RoleRepository extends JpaRepository<Role, Long> {
    public List<Role> findAll();
}
```

## SpringBoot集成Redis

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
```

```properties
# redis配置信息
spring.redis.host=localhost
spring.redis.port=6379
```

```java
package cn.baobaoxuxu.community;

import cn.baobaoxuxu.community.pojo.Role;
import cn.baobaoxuxu.community.repository.RoleRepository;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.List;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = CommunityApplication.class)
public class RedisTest {

    @Autowired
    RoleRepository dao;

    @Autowired
    RedisTemplate<String, String> redisTemplate;

    @Test
    public void test1() throws JsonProcessingException {
        // 1. 从redis拿数据
        String s = redisTemplate.boundValueOps("user.findAll").get();
        if (s == null) {
            List<Role> all = dao.findAll();
            ObjectMapper objectMapper = new ObjectMapper();
            String s1 = objectMapper.writeValueAsString(all);
            redisTemplate.boundValueOps("user.findAll").set(s1);
            System.out.println("从数据库拿了 存到了redis");
        }else{
            System.out.println("直接從redis拿");
        }
        //2. 判断redis是否存在数据
    }
}
```

