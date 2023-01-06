# MyBatis-Plus 基础

- 整合 MyBatis-Plus
- 通用 CRUD 写法
- 基本配置
- 条件构造器

## 概述

MyBatis-Plus（简称 MP）是一个 MyBatis 的增强工具，在 MyBatis 的基础上只做增强不做改变，为简化开发、提高效率而生（向 JPA 看起？）

官网：https://mp.baomidou.com/

> MyBatis-Plus 架构图

[(2条消息) Mybatis-plus 实践与架构原理图解_骏马逸动，心随你动的博客-CSDN博客_mybatis-plus原理](https://blog.csdn.net/a1036645146/article/details/105449641)

<div align="center"><img src="img/ibatis/plus/mybatis-plus-framework.jpg"></div>

MyBatis-Plus 在 MyBatis 的 xml 和注解注入之后，通过反射分析实体，将通用的 CRUD 方法注入；在注入前会进行判断，如果有一样的方法注入了就不会再注入。

> MyBatis vs JPA

MyBatis 的优势

- SQL 语句可以自由控制，更灵活，性能比 JPA 略高；但是新版的 JPA SQL 语句也很灵活了。
- SQL 与代码分离，易于阅读和维护
- 提供 XML 标签，支持编写动态 SQL 语句

JPA 的优势

- JPA 一致性比较好
- 提供了很多 CRUD 方法，开发效率高
- 对象化程度更高

MyBatis 的劣势

- 简单 CRUD 还得些 SQL 语句
- XML 中有大量的 SQL 要维护
- MyBatis 自身功能很有限，但支持 Plugin

> MyBatis-Plus 特性

<div align="center"><img src="img/ibatis/MyBatis_quick_star.png"></div>

- <b>损耗小：</b>启动即会自动注入基本 CURD，性能基本无损耗，直接面向对象操作
- <b>强大的 CRUD 操作：</b>内置通用 Mapper、通用 Service，仅仅通过少量配置即可实现单表大部分 CRUD 操作，更有强大的条件构造器，满足各类使用需求
- <b>支持 Lambda 形式调用：</b>通过 Lambda 表达式，方便的编写各类查询条件，无需再担心字段写错
- <b>支持主键自动生成：</b>支持多达 4 种主键策略（内含分布式唯一 ID 生成器 - Sequence），可自由配置，完美解决主键问题
- <b>支持 ActiveRecord 模式：</b>支持 ActiveRecord 形式调用，实体类只需继承 Model 类即可进行强大的 CRUD 操作
- <b>支持自定义全局通用操作：</b>支持全局通用方法注入（ Write once, use anywhere ）
- <b>内置代码生成器：</b>采用代码或者 Maven 插件可快速生成 Mapper 、 Model 、 Service 、 Controller 层代码，支持模板引擎，更有超多自定义配置等您来使用
- <b>内置分页插件：</b>基于 MyBatis 物理分页，开发者无需关心具体操作，配置好插件之后，写分页等同于普通 List 查询
- <b>分页插件支持多种数据库：</b>支持 MySQL、MariaDB、Oracle、DB2、H2、HSQL、SQLite、Postgre、SQLServer 等多种数据库
- <b>内置性能分析插件：</b>可输出 SQL 语句以及其执行时间，建议开发测试时启用该功能，能快速揪出慢查询
- <b>内置全局拦截插件：</b>提供全表 delete 、 update 操作智能分析阻断，也可自定义拦截规则，预防误操作

## 快速开始

### 创建表

```mysql
create database mybatis_plus;
use mybatis_plus;

CREATE TABLE `tb_user` (
    `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '主键ID',
    `user_name` varchar(20) NOT NULL COMMENT '用户名',
    `password` varchar(20) NOT NULL COMMENT '密码',
    `name` varchar(30) DEFAULT NULL COMMENT '姓名',
    `age` int(11) DEFAULT NULL COMMENT '年龄',
    `email` varchar(50) DEFAULT NULL COMMENT '邮箱',
     PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8;

-- 插入测试数据
INSERT INTO `tb_user` (`id`, `user_name`, `password`, `name`, `age`, `email`) VALUES
('1', 'zhangsan', '123456', '张三', '18', 'test1@itcast.cn');
INSERT INTO `tb_user` (`id`, `user_name`, `password`, `name`, `age`, `email`) VALUES
('2', 'lisi', '123456', '李四', '20', 'test2@itcast.cn');
INSERT INTO `tb_user` (`id`, `user_name`, `password`, `name`, `age`, `email`) VALUES
('3', 'wangwu', '123456', '王五', '28', 'test3@itcast.cn');
INSERT INTO `tb_user` (`id`, `user_name`, `password`, `name`, `age`, `email`) VALUES
('4', 'zhaoliu', '123456', '赵六', '21', 'test4@itcast.cn');
INSERT INTO `tb_user` (`id`, `user_name`, `password`, `name`, `age`, `email`) VALUES
('5', 'sunqi', '123456', '孙七', '24', 'test5@itcast.cn');
```

### 导入依赖

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.4.2</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>
    <groupId>com.example</groupId>
    <artifactId>demo</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>springboot</name>
    <description>Demo project for Spring Boot</description>
    <properties>
        <java.version>8</java.version>
    </properties>
    <dependencies>
        <!-- MyBatis-Plus -->
        <dependency>
            <groupId>com.baomidou</groupId>
            <artifactId>mybatis-plus-boot-starter</artifactId>
            <version>3.4.3</version>
        </dependency>
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
        </dependency>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
        </dependency>
        
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-configuration-processor</artifactId>
            <optional>true</optional>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>org.slf4j</groupId>
            <artifactId>slf4j-log4j12</artifactId>
        </dependency>
    </dependencies>
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
```

### 创建 Boot + 整合

```yml
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/mp?serverTimezone=UTC
    username: root
    password: root

mybatis-plus:
  configuration:
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl 
    # 打印SQL语句到控制台
```

其他整合方式查阅官网。

### 测试代码

```java
@Mapper
public interface UserMapper extends BaseMapper<User> {}

@Data
@AllArgsConstructor
@NoArgsConstructor
@TableName("tb_user")
public class User {
    private Integer id;
    private String userName;
    private String password;
    private String name;
    private Integer age;
    private String email;
}

@SpringBootApplication
@RestController
public class MPApplication {

    public static void main(String[] args) {
        SpringApplication.run(MPApplication.class, args);
    }

    @Autowired
    UserMapper userMapper;

    @GetMapping("/all")
    public List<User> queryAll() {
        return userMapper.selectList(null);
    }
}
```

单元测试的方式进行代码测试

```java
@SpringBootTest
public class ApplicationTest {
    @Autowired
    UserMapper mapper;

    @Test
    public void first() {
        mapper.selectList(null).forEach(System.out::println);
    }
}
```

注意：MyBatis-Plus 中，如果数据库表中有下划线字段会默认采用驼峰转换。如数据中的字段为 `user_name`，Java 代码中为 `userName` 是可以自动对应上的。

### 常用注解

- @TableName -- 做类名和表名的映射，如果表名和类名不一致可以使用该注解。
- @TableId -- 表示这是主键，MP 默认是找名字为 id 的作为数据库主键，如果没有名为 id 的字段就会找不到主键，此时可以用 @TableId 标识某个字段为主键。
- @TableField -- 普通列，数据库中和类中的名字不一样，此时可以用 @TableField 进行映射，但是只有普通列才有效果！

> 修改数据库字段 age 为 ages，id 为 user_id

```java
@Data
@AllArgsConstructor
@NoArgsConstructor
@TableName("tb_user") // User 对应数据库表 tb_user
public class User {
    @TableId
    private Integer userId;
    private String userName;
    private String password;
    private String name;
    @TableField("ages")
    private Integer age;
    private String email;
    // @TableField(exist = false)
    // private String remark;
}
```

```java
import com.mp.mapper.UserMapper;
import com.mp.pojo.User;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.List;

@SpringBootTest
public class MPApplicationTest {
    @Autowired
    UserMapper userMapper;

    @Test
    void testSelect() {
        List<User> users = userMapper.selectList(null);
        users.stream().forEach(System.out::println);
    }

    @Test
    void testInsert() {
        // id 为空的话，mp 会用雪花算法生成 id 进行填充。
        User user = new User(null, "Jack", "123", "Jack", 12, "hello@qq");
        userMapper.insert(user);
        System.out.println(user);
    }
}
```

### 排除非表字段

一共有三种方式

- 为字段加上 `transient` 关键字
- 用 `static` 修饰字段
- 为字段加上注解 `@TableField(exist=false)`，表示它不是数据库中的字段

在 User 中添加一个数据库中不存在的字段，不加任何处理进行数据库查询时报错，用上述方案解决时就不再报错了。

## 查询方法

通过继承 BaseMapper 就可以获取到各种各样的单表操作，接下来详细讲解这些操作。

<div align="center"><img src="img/ibatis/plus/image-20211105220903263.png"></div>



没啥好记的，现查现用。只记录一个分页查询

### select 基本方法

> <b>按 id 查询 / 按 id 批量查询 / 按 map 中的条件等值查询</b>

```java
@Test
void testId() {
    User user = userMapper.selectById(1);
    log.info("user {}", user);
}

@Test
void testIds() {
    List<User> users = userMapper.selectBatchIds(Arrays.asList(1, 2, 3, 4));
    users.forEach(System.out::println);
}


@Test
void testQueryByMapCondition() {
    Map<String, Object> queryMap = new HashMap<>();
    // queryMap.put("ages", "12"); // 虽然 ages 是 int 类型，但是 map 中可以存 String
    queryMap.put("ages", 12);
    List<User> users = userMapper.selectByMap(queryMap);
    users.forEach(System.out::println);
}
```

> <b>根据 Wrapper 封装条件进行查询</b>

所有的 Wrapper 普通类都继承自 AbstractWrapper。（抽象类和接口的区别，抽象类中可以预先定义一些具体的方法复用；虽然 Java8 的接口也可以有具体的方法了，这点感觉模糊了抽象类和接口的边界）

```java
@Test
void testWrapperCondition() {
    QueryWrapper<User> query = new QueryWrapper<>();
    // query.like("user_name", "z"); // %z%
    // query.likeLeft("user_name", "z"); // %z
    query.likeRight("user_name", "z"); // z%
    List<Map<String, Object>> maps = userMapper.selectMaps(query);
    System.out.println(maps.size());
}

@Test
void testWrapperCondition2() {
    QueryWrapper<User> query = new QueryWrapper<>();
    // select * from tb_user where (user_id between 1,3 and user_name like "z%");
    query.between("user_id", 1, 3).likeRight("user_name", "z");
    List<Map<String, Object>> maps = userMapper.selectMaps(query);
    System.out.println(maps.size());
}

@Test
void testWrapperCondition3() {
    QueryWrapper<User> query = new QueryWrapper<>();
    // select * from tb_user where (user_id between 1,3 or user_name like "z%");
    query.between("user_id", 1, 3).or().likeRight("user_name", "z");
    List<Map<String, Object>> maps = userMapper.selectMaps(query);
    System.out.println(maps.size());
}

@Test
void testWrapperCondition4() {
    QueryWrapper<User> query = new QueryWrapper<>();
    // apply 可以用于 where 中执行某些自定义的 sql 片段，如下面这种
    // 用 {0} 防止 sql 注入
    query.apply("ages-1 = {0}", 11);
    // 还可以用于执行某些函数 apply("date_format(dateColumn,'%Y-%m-%d') = {0}", "2008-08-08")
    List<Map<String, Object>> maps = userMapper.selectMaps(query);
    System.out.println(maps.size());
}

@Test
void testWrapperInSql() {
    // inSql -- 子查询
    QueryWrapper<User> query = new QueryWrapper<>();
    query.inSql("ages", "select ages from tb_user where ages>11");
    List<Map<String, Object>> maps = userMapper.selectMaps(query);
    System.out.println(maps.size());
}

@Test
void testWrapperAnd() {
    // select * from tb_user where user_name like 'z%' or (ages<17 and user_name like 'l%');
    QueryWrapper<User> query = new QueryWrapper<>();
    query.likeRight("user_name", "z")
        .or(w -> w.lt("ages", 17).likeRight("user_name", "l"));
    System.out.println(query.getTargetSql());
    List<User> users = userMapper.selectList(query);
    users.forEach(System.out::println);
}

@Test
void testWrapperAnd2() {
    // select * from tb_user where (ages<17 and user_id > 0) or user_name like 'l%';
    QueryWrapper<User> query = new QueryWrapper<>();
    query.nested(w -> w.lt("ages", 17).gt("user_id", 0))
        .or().likeRight("user_name", "l");
    System.out.println(query.getTargetSql());
    List<User> users = userMapper.selectList(query);
    users.forEach(System.out::println);
}
```

> <b>一些需要注意的 Wrapper 中的方法</b>

- nested，nested 内的查询条件会多套一个 `()`，`nested(i -> i.eq("name", "李白").ne("status", "活着")) ---> (name = '李白' and status <> '活着')`
- apply，用于 where 中执行一些 sql 片段，可以执行诸如 where age-1 = 10 这种语句，和 `"date_format(dateColumn,'%Y-%m-%d') = {0}"` 这种语句
- last，无视优化规则直接拼接到 sql 的最后，有 sql 注入风险

### selectPage

需要注册一个分页插件到 IOC 容器中

```java
@SpringBootApplication
public class ApplicationContext {
    public static void main(String[] args) {
        ConfigurableApplicationContext run = SpringApplication.run(ApplicationContext.class);
    }

    @Bean // 注册分页插件
    public MybatisPlusInterceptor mybatisPlusInterceptor() {
        MybatisPlusInterceptor plus = new MybatisPlusInterceptor();
        plus.addInnerInterceptor(new PaginationInnerInterceptor(DbType.MYSQL));
        return plus;
    }
}
```

分页代码

```java
@Test
public void selectPage() {
    Page<User> page = new Page<>();
    page.setCurrent(1); // 设置起始页
    page.setSize(2); // 每页查询的数据量
    mapper.selectPage(page, null); // 查询条件为 null
}
```

### 不列出所有字段

使用 select 限定出现那些字段，不出现那些字段。

```java
@Test
void testSelectSomeField() {
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    // SELECT user_id,user_name FROM tb_user
    // 这种方式字段一多久很麻烦。
    queryWrapper.select("user_id,user_name"); // 此处要传数据库的字段名
    userMapper.selectList(queryWrapper).forEach(System.out::println);
}
```

指定哪些字段不出出现，但是这种方式 select 过滤的字段不包括主键！

```java
@Test
void testSelectSomeField2() {
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    // 不查询 user_name 字段
    queryWrapper.select(User.class, fields -> !fields.getColumn().equals("user_name"));
    userMapper.selectList(queryWrapper).forEach(System.out::println);
}
```

### 带 condition 的查询

```java
// 条件为 true 才进行 sql 拼接
like(boolean condition, R column, Object val);
```

### 以实体为查询条件

以实体为查询条件默认是等值查询，但是可以在实体字段上加上注解进行修改。

```java
@Test
void testSelectByEntity() {
    User user = new User();
    user.setUserId(1);
    // SELECT user_id,user_name,password,name,ages AS age,email FROM tb_user WHERE user_id=?
    QueryWrapper<User> queryWrapper = new QueryWrapper<>(user);
    userMapper.selectList(queryWrapper).forEach(System.out::println);
}
```

```java
// 参照 SqlCondition 写的小于
@TableField(value = "ages", condition = "%s&lt;#{%s}")
private Integer age;

@Test
void testSelectByEntity() {
    User user = new User();
    user.setAge(15);
    // SELECT user_id,user_name,password,name,ages AS age,email FROM tb_user WHERE ages<?
    QueryWrapper<User> queryWrapper = new QueryWrapper<>(user);
    userMapper.selectList(queryWrapper).forEach(System.out::println);
}
```

### allEq

```java
queryWrapper.allEq(params, false); // 为 null 的等值查询会被忽略
```

## 更新及删除

### 更新方法

常用的更新方法如下

- updateById -- 实体中不为 null 的字段值会进行更新
- update(Entity, Wrapper) -- Entity 为需要 set 的内容，Wrapper 为 set 的条件；也可以 Entity 为 null，Wrapper 中设置好 set 条件和需要 set 的内容。
- LambdaUpdateChainWrapper -- 直接链式调用完成更新

updateById -- 实体中不为 null 的字段值会进行更新

```java
@SpringBootTest
@Slf4j
public class MPApplicationTestUpdate {
    @Autowired
    UserMapper userMapper;

    UpdateWrapper<User> update = new UpdateWrapper<>();
    
    User user = new User();
    
    @Test
    public void testUpdate() {
        // UPDATE tb_user SET ages=? WHERE user_id=?
        // 记录中不为 null 的字段值会进行更新
        user.setUserId(1);
        user.setAge(26);
        int rows = userMapper.updateById(user);
        log.info("受影响的记录数 {}", rows);
    }
}
```

update(Entity, Wrapper) -- Entity 为需要 set 的内容，Wrapper 为 set 的条件；也可以 Entity 为 null，Wrapper 中设置好 set 条件和需要 set 的内容。

```java
@SpringBootTest
@Slf4j
public class MPApplicationTestUpdate {
    @Autowired
    UserMapper userMapper;

    UpdateWrapper<User> update = new UpdateWrapper<>();

    User user = new User();
    
    @Test
    public void testUpdateByWrapper1() {
        // 只设置 set 的条件
        UpdateWrapper<User> set = update.eq("user_id", 1);
        user.setAge(10);
        // UPDATE tb_user SET ages=? WHERE (user_id = ?)
        int rows = userMapper.update(user, set);
        log.info("受影响的记录数 {}", rows);
    }

    @Test
    public void testUpdateByWrapper2() {
        UpdateWrapper<User> set = update.eq("user_id", 1).set("ages", 100);
        // UPDATE tb_user SET ages=? WHERE (user_id = ?)
        int rows = userMapper.update(null, set);
        log.info("受影响的记录数 {}", rows);
        
        // 也可以用 lambda 表达式的 Wrapper
        LambdaUpdateWrapper<User> lambda = new LambdaUpdateWrapper<>();
        lambda.eq(User::getUserId, 1).set(User::getAge, 10);
        rows = userMapper.update(null, lambda);
        log.info("受影响的记录数 {}", rows);
    }
}
```

LambdaUpdateChainWrapper -- 直接链式调用完成更新

```java
@Test
public void testLambdaChainWrapper() {
    boolean update = new LambdaUpdateChainWrapper<User>(userMapper)
        .eq(User::getUserId, 1)
        .set(User::getAge, 10)
        .update();
    log.info("操作结果 {}", update);

}
```

### 删除方法

- deleteById
- deleteByMap
- deleteBatchIds
- delete( Wrapper\<T\> )

## 自定义 SQL

执行自定义 SQL 查询有两种方式，一种是使用注解/XML，一种是使用 SqlRunner。

### 自定义注解/XML

可以使用注解自定义，也可以使用 XML

```java
import com.baomidou.mybatisplus.core.conditions.Wrapper;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.toolkit.Constants;
import com.mp.pojo.User;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

// 使用注解自定义
@Mapper
public interface UserMapper extends BaseMapper<User> {

    @Select("select user_id,user_name from tb_user ${ew.customSqlSegment}")
    List<User> selectByDefaultCondition(@Param(Constants.WRAPPER) Wrapper<User> wrapper);
}
```

修改 yml 配置

```yaml
spring:
  datasource:
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/mybatis_plus?serverTimezone=UTC
    username: root
    password: root

mybatis-plus:
  configuration:
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
  mapper-locations: classpath:com/mapper/*.xml # 增加这句配置，指定 xml 文件的位置
```

使用 XML 自定义 SQL

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper
        PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.mp.mapper.UserMapper">
    <select id="selectByDefaultCondition" resultType="com.mp.pojo.User">
        select *
        <!-- 不用加 where，mp 会自动加上去的 -->
        from tb_user ${ew.customSqlSegment}
    </select>
</mapper>
```

<b style="color:orange">如果需要 xml 文件和接口文件存储在一个路径下，则需要为 maven 配置下面这个属性</b>

```xml
<build>
    <resources>
        <resource>
            <directory>src/main/java</directory>
            <includes>
                <include>**/*.xml</include>
            </includes>
            <filtering>true</filtering>
        </resource>
    </resources>
</build>
```

测试代码

```java
@Test
void testXML() {
    QueryWrapper<User> query = new QueryWrapper<User>().eq("user_id", 1);
    // 传入的对象 query 不能为 null！
    userMapper.selectByDefaultCondition(query).forEach(System.out::println);
}

@Test
void testXML2() {
    // 没有设置条件那就是查询所有数据了。
    QueryWrapper<User> query = new QueryWrapper<User>();
    userMapper.selectByDefaultCondition(query).forEach(System.out::println);
}
```

### SqlRunner

配置文件，启用 SqlRunner

```properties
mybatis-plus.global-config.enable-sql-runner=true
```

执行代码

```java
private void updateUserRating(Player one, boolean win) {
    int rating = win ? 5 : -5;
    // SqlRunner 无需手动注入
    SqlRunner.db().update("update user set rating = rating + {0},count=count+1 where id = {1}", rating, one.getId());
}

private void testDefineSql() {
    // 执行自定义sql
    Object o = SqlRunner.db().selectObj("explain select count(*) from tb_user");
    System.out.println(o);
}
```

## 分页查询

- MyBatis 分页
- MP 分页插件实现物理分页

### 基本的分页查询

注册分页查询插件

```java
import com.baomidou.mybatisplus.annotation.DbType;
import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MPConfig {
    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor() {
        MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
        interceptor.addInnerInterceptor(new PaginationInnerInterceptor(DbType.H2));
        return interceptor;
    }
}
```

使用 Page 测试分页查询（可以通过设置 searchCount 为 false，不查询总记录数，如果需要 count 的话可以用 Redis 缓存或 explain 得到模糊的 count）

```java
@Test
void testPage() {
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    
    // 页数默认从1开始，大于1才会重设 current page 的值
    Page<User> page = new Page<>(1, 3);
    userMapper.selectPage(page, queryWrapper);
    System.out.println(page.getPages());
    System.out.println(page.getCurrent());
    System.out.println(page.getRecords().size());
}

@Test
void testPageIgnoreTotalPage() {
    QueryWrapper<User> queryWrapper = new QueryWrapper<>();
    // searchCount 设置为 false，不查询总记录数
    Page<User> page = new Page<>(0, 3, false);
    userMapper.selectPage(page, queryWrapper);
    System.out.println(page.getPages());
    System.out.println(page.getCurrent());
    System.out.println(page.getRecords().size());
}
```

### 基于 XML 的分页查询

接口中定义方法

```java
List<User> selectByPage(IPage<User> page, Integer ages);
```

XML 中编写 sql

```xml
<select id="selectByPage" resultType="com.mp.pojo.User">
    select *
    from mybatis_plus.tb_user
    where ages &lt; #{ages}
</select>
```

测试代码，依旧是不查询总页数

```java
@Test
void testPageIgnoreTotalPageXML() {
    // searchCount 设置为 false，不查询总记录数
    Page<User> page = new Page<>(1, 3, false);
    userMapper.selectByPage(page, 20);
    System.out.println(page.getPages());
    System.out.println(page.getCurrent());
    System.out.println(page.getRecords().size());
}
```

## ActiveRecord 模式

AR，活动记录，一种领域模型模式，一个模型类对应关系型数据库中的一个表，模型类的一个实例对应表中的一行记录。通过实体对象直接对表进行增删改查操作。

## SQL 注入原理

### 基本原理

MP 在启动后会将 BaseMapper 中的一系列的方法注册到 meppedStatements 中，那么究竟是如何注入的？流程又是怎么样的？

在 MP 中，ISqlInjector 负责 SQL 的注入工作，它是一个接口，AbstractSqlInjector 是它的实现类，实现关系如下

<div align="center"><img src="img/ibatis/plus/image-20211105224958774.png"></div>

在 AbstractSqlInjector 中，主要是由 inspectInject() 方法进行注入的

```java
@Override
public void inspectInject(MapperBuilderAssistant builderAssistant, Class<?> mapperClass) {
    Class<?> modelClass = extractModelClass(mapperClass);
    if (modelClass != null) {
        String className = mapperClass.toString();
        Set<String> mapperRegistryCache = GlobalConfigUtils.getMapperRegistryCache(builderAssistant.getConfiguration());
        if (!mapperRegistryCache.contains(className)) {
            List<AbstractMethod> methodList = this.getMethodList(mapperClass);
            if (CollectionUtils.isNotEmpty(methodList)) {
                TableInfo tableInfo = TableInfoHelper.initTableInfo(builderAssistant, modelClass);
                // 循环注入自定义方法 inject 注入
                methodList.forEach(m -> m.inject(builderAssistant, mapperClass, modelClass, tableInfo));
            } else {
                logger.debug(mapperClass.toString() + ", No effective injection method was found.");
            }
            mapperRegistryCache.add(className);
        }
    }
}

public void inject(MapperBuilderAssistant builderAssistant, Class<?> mapperClass, Class<?> modelClass, TableInfo tableInfo) {
    this.configuration = builderAssistant.getConfiguration();
    this.builderAssistant = builderAssistant;
    this.languageDriver = configuration.getDefaultScriptingLanguageInstance();
    /* 注入自定义方法 */
    injectMappedStatement(mapperClass, modelClass, tableInfo);
}
```

在实现方法中，`methodList.forEach(m -> m.inject(builderAssistant, mapperClass, modelClass, tableInfo));` 是关键，循环遍历方法，进行注入。 最终调用抽象方法 injectMappedStatement 进行真正的注入；该抽象方法的实现如下：

<div align="center"><img src="img/ibatis/plus/image-20211105225432807.png"></div>

### 典例

以 DeleteById 为例

```java
public class DeleteById extends AbstractMethod {

    @Override
    public MappedStatement injectMappedStatement(Class<?> mapperClass, Class<?> modelClass, TableInfo tableInfo) {
        String sql;
        SqlMethod sqlMethod = SqlMethod.LOGIC_DELETE_BY_ID;
        if (tableInfo.isWithLogicDelete()) { // 是否是逻辑删除
            sql = String.format(sqlMethod.getSql(), tableInfo.getTableName(), sqlLogicSet(tableInfo),
                tableInfo.getKeyColumn(), tableInfo.getKeyProperty(),
                tableInfo.getLogicDeleteSql(true, true));
            SqlSource sqlSource = languageDriver.createSqlSource(configuration, sql, Object.class);
            return addUpdateMappedStatement(mapperClass, modelClass, getMethod(sqlMethod), sqlSource);
        } else { // 物理删除
            sqlMethod = SqlMethod.DELETE_BY_ID;
            sql = String.format(sqlMethod.getSql(), tableInfo.getTableName(), tableInfo.getKeyColumn(),
                tableInfo.getKeyProperty());
            SqlSource sqlSource = languageDriver.createSqlSource(configuration, sql, Object.class);
            return this.addDeleteMappedStatement(mapperClass, getMethod(sqlMethod), sqlSource);
        }
    }
}
```

<div align="center"><img src="img/ibatis/plus/image-20211105225944851.png"></div>

## 配置

MP 中有大量配置，相当一部分是 MyBatis 的原生配置，另一部分是 MP 的配置。[使用配置 | MyBatis-Plus (baomidou.com)](https://mp.baomidou.com/config/#基本配置)

### 基本配置

#### configLocation

- 类型：`String`
- 默认值：`null`

MyBatis 配置文件位置，如果您有单独的 MyBatis 配置，请将其路径配置到 configLocation 中。 MyBatis Configuration 的具体内容请参考 MyBatis 官方文档

```properties
mybatis-plus.config-location = classpath:mybatis-config.xml
```

#### mapperLocations

MyBatis Mapper 所对应的 XML 文件位置，如果您在 Mapper 中有自定义方法（XML 中有自定义实现），需要进行该配置，告诉 Mapper 所对应的 XML 文件位置。

```properties
# 这个 * 不加也能定位到 xml 所在的文件夹
mybatis-plus.mapper-locations = classpath*:mybatis/*.xml
```

Maven 多模块项目的扫描路径需以 classpath*: 开头 （即加载多个 jar 包下的 XML 文件）

#### typeAliasesPackage

MyBaits 别名包扫描路径，通过该属性可以给包中的类注册别名，注册后在 Mapper 对应的 XML 文件中可以直接使 用类名，而不用使用全限定的类名（即 XML 中调用的时候不用包含包名）。

```properties
mybatis-plus.type-aliases-package = cn.itcast.mp.pojo
```

### 进阶配置

本部分（Configuration）的配置大都为 MyBatis 原生支持的配置，这意味着您可以通过 MyBatis XML 配置文件的形式进行配置。

#### mapUnderscoreToCamelCase

- 类型： boolean 
- 默认值： true

是否开启自动驼峰命名规则（camel case）映射，即从经典数据库列名 A_COLUMN（下划线命名） 到经典 Java 属性名 aColumn（驼峰命名） 的类似映射。

>此属性在 MyBatis 中原默认值为 false，在 MyBatis-Plus 中，此属性也将用于生成最终的 SQL 的 select body 如果您的数据库命名符合规则无需使用 @TableField 注解指定数据库字段名

```properties
#关闭自动驼峰映射，该参数不能和mybatis-plus.config-location同时存在
mybatis-plus.configuration.map-underscore-to-camel-case=false
```

#### cacheEnabled

- 类型： boolean 
- 默认值： true

全局地开启或关闭配置文件中的所有映射器已经配置的任何缓存，默认为 true。

```properties
mybatis-plus.configuration.cache-enabled=false
```

### DB 策略配置

#### idType

- 类型： com.baomidou.mybatisplus.annotation.IdType 
- 默认值： ID_WORKER

全局默认主键类型，设置后，即可省略实体对象中的 @TableId(type = IdType.AUTO) 配置。

#### tablePrefix

- 类型： String 
- 默认值： null

表名前缀，全局配置后可省略 @TableName() 配置。

# MyBatis-Plus 高级

