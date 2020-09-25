# 快捷键基础篇

## 前言

常用快捷键

## w10快捷键

- Alt + Tab 选择活动窗口

## idea快捷键

- ctrl + Tab 切换窗口
- ctrl + E 最近编辑的窗口 
- Alt + 1 显示/隐藏侧边栏
- ctrl + F4 关闭当前窗口
- Alt + Insert 插入代码【如：生成set get方法】
- Alt + Shift + R 重命名
- Ctrl + Shift + F10 运行代码
- Ctrl + W 关闭侧边栏

----

# `MyBatis`中的设计模式

绝对路径：

相对路径 `src/java/main/文件名.xml`

读配置文件：①用类加载器，读类路径下的

​					 ②用`Servlet Context`对象的`getRealPath`

创建工程`MyBatis`用了构建者模式。告诉需求，根据需求创建我们想要的。

```java
build.build(in) // in形式下创建的工厂，多了几个类，操作看起来麻烦了，但是组合更加灵活的。
```

生成`SqlSession`用了工厂模式

创建`Dao`接口实现类用了代理模式

在看`MyBatis`源码的时候，通过一些类的名称大概知道了`MyBatis`用到了什么技术。`MyBatis`解析的时候应该用到了词法分析，分析字符串。在动态生成代理类的时候用到了字节码增强技术。

# `MyBatis` 基础篇

## 基本环境搭建

> **Maven工程使用`MyBatis`的时候，配置文件需要放在`resrouces`目录下，否则无法找到。**

> **Maven整合Druid的时候，需要的是数据源，需要我们手动new出Druid的数据源。**

- 基本配置文件
- mapper文件
- 日志文件

**maven的`pom`文件**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.bbxx</groupId>
    <artifactId>MyBatis02</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>jar</packaging>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <configuration>
                    <source>8</source>
                    <target>8</target>
                </configuration>
            </plugin>
        </plugins>
    </build>

    <dependencies>
        <dependency>
            <groupId>org.mybatis</groupId>
            <artifactId>mybatis</artifactId>
            <version>3.4.5</version>
        </dependency>
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <version>4.13</version>
            <scope>test</scope>
        </dependency>
        <dependency>
            <groupId>log4j</groupId>
            <artifactId>log4j</artifactId>
            <version>1.2.17</version>
        </dependency>
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>8.0.21</version>
        </dependency>
    </dependencies>

</project>
```

**`SqlConfig`配置文件**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE configuration
        PUBLIC "-//mybatis.org//DTD Config 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-config.dtd">
<configuration>
    <!--配置包别名【resultType】-->
    <typeAliases>
        <package name="com.bbxx.pojo"/>
    </typeAliases>
    <!-- 配置 mybatis 的环境 -->
    <environments default="mysql">
        <!-- 配置 mysql 的环境 -->
        <environment id="mysql">
            <!-- 配置事务的类型 -->
            <transactionManager type="JDBC"></transactionManager>
            <!-- 配置连接数据库的信息：用的是数据源(连接池) 如果用的是三方数据源，如阿里的druid-->
            <!-- <dataSource type="xx.xx.DruidPoolConfig">这个DruidPoolConfig是我们自己new出来的 -->
            <dataSource type="POOLED">
                <property name="driver" value="com.mysql.cj.jdbc.Driver"/>
                <property name="url" value="jdbc:mysql://localhost:3306/mybatis?serverTimezone=UTC"/>
                <property name="username" value="root"/>
                <property name="password" value="root"/>
            </dataSource>
        </environment>
    </environments>
    <!-- 告知 mybatis 映射配置的位置 -->
    <!-- com/bbxx/dao/UserDao.xml是mavenresouce目录下的哦！ -->
    <mappers>
        <mapper resource="com/bbxx/dao/UserDao.xml"/>
        <!-- 如果是用的注解SQL，则采用,因为注解方式不用Mapper文件！ -->
        <mapper class="com.bbxx.dao.IUserDao"/>
    </mappers>
</configuration>
```

**mapper文件示例**

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE mapper
        PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<!-- namespace是接口的类全名 resultType是返回类型的类全民，可通过配置简写 -->
<mapper namespace="xx.dao.xx">
    <select id="select" resultType="xx.Article">
		select title,username from article;
	</select>
</mapper>
```

## 日志相关

**`log4j`日志配置**

```properties
#log4j基本配置
log4j.rootLogger=DEBUG,console,file
#设置输出控制台信息
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.Target = System.out
log4j.appender.console.Threshold=DEBUG     #控制台只会打印INFO级别及以上的日志信息
log4j.appender.console.layout = org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%c-%m%n
#设置输出文件信息
log4j.appender.file = org.apache.log4j.RollingFileAppender
log4j.appender.file.File=log/mybatis.log     #日志文件路径，在应用下的log/mybatis.log文件中
log4j.appender.file.Append=true   #追加
log4j.appender.file.MaxFileSize=100mb    #达到100m后产生一个新文件
log4j.appender.file.Threshold=ERROR      #只会写ERROR级别及以上的日志信息
log4j.appender.file.layout=org.apache.log4j.PatternLayout     #布局器
log4j.appender.file.layout.ConversionPattern=%c-%m%n   #布局器格式
```

**`log4j`仅打印`SQL`语句**

```properties
# 全局日志配置
log4j.rootLogger=ERROR, stdout
# MyBatis 日志配置  com.bbxx.dao是包名。
log4j.logger.com.bbxx.dao=TRACE
# 控制台输出
log4j.appender.stdout=org.apache.log4j.ConsoleAppender
log4j.appender.stdout.layout=org.apache.log4j.PatternLayout
log4j.appender.stdout.layout.ConversionPattern=%5p [%t] - %m%n
```

----

## Mapper映射文件

<a href="https://mybatis.org/mybatis-3/zh/configuration.html#mappers">官方链接</a>

Mapper映射文件放在maven工程resource下com/daily/mapper也是resource的子目录

1、用文件路径引入

```xml
<mappers>
    <mapper resource="com/daily/mapper/UserMapper.xml" />
    <mapper resource="com/daily/mapper/ProductMapper.xml" />
    <mapper resource="com/daily/mapper/BlogMapper.xml" />
</mappers>
```

2、用包名引入

这种引入方式相当于批量引入一个包下的所有映射器。此种方式要求xml和接口名称一致？

```xml
<mappers>
    <package name="com.daily.mapper"/>
</mappers>
```

3、用类注册引入

```xml
<mappers>
    <mapper class="com.daily.mapper.UserMapper" />
    <mapper class="com.daily.mapper.ProductMapper" />
    <mapper class="com.daily.mapper.BlogMapper" />
</mappers>
```

4、使用URL方式引入

```xml
<mappers>
    <mapper url="xml文件访问URL" />
</mappers>
```

maven的resource是项目的资源根目录哦！

`mybatis` 多对多 是两个一对一组成的哦

​	user 一对多role

​	role 一对多user

这样 user 和 role就是多对多关系了。 数据库的多对多需要一个中间表来描述 两表的多对多关系。

多对多`SQL`的写法？

<a href="https://github.com/csliujw/MyBatis-Study">项目地址</a>

# `MyBatis`(一)

## 解决的问题

`MyBatis`解决了持久层重复代码多的问题，简化了持久层的开发，减少了持久层的代码量。

## 简单的CURD

```java
package com.bbxx.dao;

import com.bbxx.pojo.UserVO;

import java.util.List;

public interface IUserDao {

    // 查询所有
    List<UserVO> findAll();

    // 条件查询
    List<UserVO> findCondition(UserVO vo);

    // 删除
    Integer delete(Integer id);

    // 修改
    Boolean update(UserVO vo);

    // 新增
    Boolean insert(UserVO vo);

    // 模糊查询
    List<UserVO> findByName(String username);

    // 聚合函数
    Long findTotal();
}
```

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper
        PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.bbxx.dao.IUserDao">
    <!-- 配置查询所有操作 -->
    <select id="findAll" resultType="com.bbxx.pojo.UserVO">
        select *
        from users
    </select>

    <select id="findCondition" resultType="UserVO">
        select * from users where 1 = 1
        <if test="id!=null">
            and id=#{id}
        </if>
        <if test="username!=null">
            and username=#{username}
        </if>
        <if test="birthday!=null">
            and birthday=#{birthday}
        </if>
    </select>

    <delete id="delete">
        delete
        from users
        where id = #{value}
    </delete>

    <update id="update" parameterType="UserVO">
        update users
        <set>
            <if test="username!=null">
                username = #{username}
            </if>
        </set>
        where id=#{id}
    </update>

    <insert id="insert">
        insert into users(username, birthday, address)
        values (#{username}, #{birthday}, #{address})
    </insert>

    <select id="findByName" resultType="UserVO">
        select *
        from users
        where username like concat("%", #{value}, "%")
    </select>

    <select id="findTotal" resultType="long">
        select count(1)
        from users
    </select>
</mapper>
```

## #和$

`#{}`表示一个占位符号

通过`#{}`可以实现 `preparedStatement` 向占位符中设置值，自动进行 `java` 类型和 `jdbc` 类型转换，

`#{}`可以有效防止 `sql` 注入。 `#{}`可以接收简单类型值或 `pojo` 属性值。 如果 `parameterType` 传输单个简单类
型值，`#{}`括号中可以是 value 或其它名称。

`${}`表示拼接 `sql` 串

通过`${}`可以将 `parameterType` 传入的内容拼接在 `sql` 中且不进行 `jdbc` 类型转换， `${}`可以接收简
单类型值或 `pojo` 属性值，如果 `parameterType` 传输单个简单类型值，`${}`括号中只能是 value。

> 源码级别解析

```java
class A{
    @Override
    public String handleToken(String content) {
      Object parameter = context.getBindings().get("_parameter");
      if (parameter == null) {
        context.getBindings().put("value", null);
      } else if (SimpleTypeRegistry.isSimpleType(parameter.getClass())) {
        context.getBindings().put("value", parameter);
      }
      Object value = OgnlCache.getValue(content, context.getBindings());
      String srtValue = (value == null ? "" : String.valueOf(value)); // issue #274 return "" instead of "null"
      checkInjection(srtValue);
      return srtValue;
    }
}
```

>读取的 key 的名字就是”value”，所以我们在绑定参数时就只能叫 value 的名字

## 深入理解

- `MyBatis`可自己写`Dao`实现类也可不写实现类。推荐不写实现类。
- 不写实现类采用的是基于代理的CRUD操作。
- `MyBatis`用到了`OGNL`表达式
  - `Object Graphic Navigation Language`
    	对象	图	导航	   语言
  - 它是通过对象的取值方法来获取数据。在写法上把get给省略了。
    比如：我们获取用户的名称
    	类中的写法：`user.getUsername();`
    	`OGNL`表达式写法：`user.username`
    `mybatis`中为什么能直接写`username`,而不用user.呢：
    	因为在`parameterType`中已经提供了属性所属的类，所以此时不需要写对象名

## 简单实现`MyBatis`

先空着

# `MyBatis`(二)

## 引言

**主要介绍**

- `mybatis`中的连接池及事务控制
  - 连接池使用及分析
  - 事务控制分析
- `xml`的动态`SQL`
- 多表操作
  - 一对一
  - 一对多
  - 多对多（不常用，不会搞这么复杂的）
    - 多对多需要一个中间表

## 连接池及事务控制

### 连接池介绍

相当于实现分配好很多数据库连接在容器中，需要用时从容器中拿连接，不需要用时把连接归还到容器中，可避免频繁的打开关闭，节约资源（打开关闭连接很消耗资源）。

优点：减少获取连接所消耗的时间

缺点：初始化连接时速度慢

### `MyBatis`中的连接池

**提供三种方式**

- 配置的位置，主配置文件（我命名为`SqlConfig.xml`）中的`dataSource`标签，type表示采用何种连接。
- type取值
  - POOLED： 采用传统的`javax.sql.DataSource`规范中的连接池，`mybatis`中有针对规范的实现。我们可以用其他连接池替代，如`Druid`，`type="我们的druid"`，因为`druid`是遵循规范的，所以把类全名加上就行了。
  - `UNPOOLED`：采用传统的获取连接的方式，虽然也实现`Javax.sql.DataSource`接口，但是并没有使用池的思想。
  - `JNDI`：采用服务器提供的`JNDI`技术实现，来获取`DataSource`对象，不同的服务器所能拿到`DataSource`是不一样。
    		   注意：如果不是web或者`maven`的`war`工程，是不能使用的。
      		  使用`tomcat`服务器的话，采用连接池就是`dbcp`连接池。

### `MyBatis`中的事务

事务的四大特性ACID
	不考虑隔离性会产生的3个问题
	解决办法：四种隔离级别

它是通过`sqlsession`对象的commit方法和rollback方法实现事务的提交和回滚

## 动态`SQL`

### if

满足条件就会拼接`SQL`，不满足就不拼接

### choose(when,otherwise)



### trim(where,set)

where多用于多条件查询，where标签会去除多余的and，<span style="color:red">**注意**and要写在最开始！</span>

set常用于update操作，set标签会去除多余的逗号，<span style="color:red">**注意**逗号要写在最后面！！！！</span>

### `foreach`

循环操作。`foreach`的collection的取值有list，array。

Map的话，查文档得知 index是key item是value

> **官方原文，具体还得看源码**

`*foreach*` 元素的功能非常强大，它允许你指定一个集合，声明可以在元素体内使用的集合项（`item`）和索引（`index`）变量。它也允许你指定开头与结尾的字符串以及集合项迭代之间的分隔符。这个元素也不会错误地添加多余的分隔符，看它多智能！

**提示** 你可以将任何可迭代对象（如 `List`、`Set` 等）、`Map` 对象或者数组对象作为集合参数传递给 `*foreach*`。当使用可迭代对象或者数组时，`index` 是当前迭代的序号，item 的值是本次迭代获取到的元素。当使用 `Map` 对象（或者 `Map.Entry` 对象的集合）时，`index` 是键，`item` 是值。

### 动态`SQL`全部代码

#### `java`代码

```java
public interface IUserDao {

    // 删除 -- 测试事务
    Integer delete(Integer id);

    // 查询所有 -- 查看事务是否成功提交
    List<UserVO> findAll();

    // 条件查询 -- 动态SQL 之if
    List<UserVO> findCondition(UserVO vo);

    // 条件查询 -- 动态SQL 之where
    List<UserVO> findCondition2(UserVO vo);

    // 新增 -- 动态SQL 之 set
    boolean update(UserVO vo);

    // 循环新增 -- 动态SQL 之 foreach
    Long insertBatch(List<UserVO> vos);

    // 循环删除 -- 动态SQL之 foreach 数组
    Long deleteBatch(Integer[] ids);

    // 循环删除 -- 动态SQL之 foreach 集合
    Long deleteBatch(List<Integer> lists);
}
```

#### `xml`文件

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper
        PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
        "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.bbxx.dao.IUserDao">
    <!-- 配置查询所有操作 -->
    <select id="findAll" resultType="UserVO">
        select *
        from users
    </select>

    <!--动态SQL if-->
    <select id="findCondition" resultType="UserVO">
        select * from users where 1 = 1
        <if test="id!=null">
            and id=#{id}
        </if>
        <if test="username!=null">
            and username like concat("%",#{username},"%")
        </if>
        <if test="birthday!=null">
            and birthday=#{birthday}
        </if>
    </select>
    <!--动态SQL where 去除前面多余的and-->
    <select id="findCondition2" resultType="UserVO">
        select * from users
        <where>
            <if test="id!=null">
                and id=#{id}
            </if>
            <if test="username!=null">
                and username like concat("%",#{username},"%")
            </if>
            <if test="birthday!=null">
                and birthday=#{birthday}
            </if>
        </where>
    </select>
    <!--测试事务-->
    <delete id="delete">
        delete
        from users
        where id = #{value}
    </delete>

    <!--动态SQL测试set 去除后面多余的逗号-->
    <update id="update" parameterType="UserVO">
        update users
        <set>
            <if test="username!=null">username=#{username},</if>
            <if test="birthday!=null">,birthday=#{birthday}</if>
        </set>
        where id=#{id}
    </update>

    <!-- 循环新增 ==> 动态SQL 之 foreach 使用ArrayList集合，collection中写的是参数的类型！这里是list集合 -->
    <insert id="insertBatch" parameterType="UserVO">
        insert into users(username,birthday,address)
        values
        <foreach collection="list" item="data" separator=",">
            (#{data.username},#{data.birthday},#{data.address})
        </foreach>
    </insert>

    <!--
    循环删除 ==> 动态SQL 之 动态SQL之 foreach 数组
    Map的话，查文档得知 index是key item是value
    -->
    <insert id="deleteBatch">
        delete from users where id in
        <foreach collection="array" item="data" open="(" separator="," close=")">
            #{data}
        </foreach>
    </insert>
</mapper>
```

#### 单元测试代码

```java
public class Demo {
    InputStream in = null;
    SqlSessionFactoryBuilder builder = null;
    SqlSessionFactory factory = null;
    SqlSession sqlSession = null;
    IUserDao mapper = null;

    @Before
    public void init() throws IOException {
        in = Resources.getResourceAsStream("SqlConfig.xml");
        builder = new SqlSessionFactoryBuilder();
        factory = builder.build(in);
        sqlSession = factory.openSession();
        mapper = sqlSession.getMapper(IUserDao.class);
    }

    @Test
    public void findAll() {
        List<UserVO> all = mapper.findAll();
    }

    /**
     * 测试事务
     */
    @Test
    public void affairs() {
        sqlSession.commit(false);
        Integer delete = mapper.delete(5);
        System.out.println(delete);
        sqlSession.commit();// 提交事务后数据才会真的删除。 // 对比数据库中的信息查看即可。
    }

    /**
     * 测试动态SQL ==> if
     */
    @Test
    public void testIf() {
        // 查出四条数据
        List<UserVO> condition = mapper.findCondition(new UserVO(null, null, null));
        // 查出三条数据
        List<UserVO> o = mapper.findCondition(new UserVO(null, "o", null));
    }

    /**
     * 测试where，会去掉前面多余的and
     */
    @Test
    public void testWhere() {
        // 查出四条数据
        List<UserVO> condition = mapper.findCondition2(new UserVO(null, null, null));
        // 查出三条数据
        List<UserVO> o = mapper.findCondition2(new UserVO(null, "o", null));
    }

    /**
     * 测试set，会去掉末尾多余的 逗号(,)
     */
    @Test
    public void testSet() {
        boolean kkx = mapper.update(new UserVO(2, "kkx", null));
        sqlSession.commit();
    }

    /**
     * 批量新增
     */
    @Test
    public void testForeach1() {
        ArrayList<UserVO> userVOS = new ArrayList<>();
        userVOS.add(new UserVO(null, "001", "1988-11-11"));
        userVOS.add(new UserVO(null, "002", "1988-02-01"));
        userVOS.add(new UserVO(null, "003", "1999-11-11"));
        userVOS.add(new UserVO(null, "004", "1995-02-21"));
        Long aLong = mapper.insertBatch(userVOS);
        System.out.println(aLong);
        sqlSession.commit();
        findAll();
    }

    /**
     * 循环删除
     */
    @Test
    public void testForeach2() {
        Integer[] ids = {10, 11, 12, 13};
        Long aLong = mapper.deleteBatch(ids);
        System.out.println(aLong);
        sqlSession.commit();
    }

    @After
    public void destroy() throws IOException {
        sqlSession.close();
        in.close();
    }
}
```



## 多表操作

如果POJO字段的名称和数据库的名称不对应则采用

```xml
<resultMap type="类型 如xx类" id="标识符">
	<id column="数据库字段名" property="代码中的字段名"></id> // 主键
    <result column="数据库字段名" property="代码中的字段名"></result> // 普通字段
</resultMap>
```

如果是一对一采用

```xml
<resultMap type="类型 如xx类" id="标识符">
	<id column="数据库字段名" property="代码中的字段名"></id> // 主键
    <result column="数据库字段名" property="代码中的字段名"></result> // 普通字段
    <association property="代码字段名" javaType="POJO属性的类型">
        <id column="数据库字段名" property="代码中的字段名"></id> 
    	<result column="数据库字段名" property="代码中的字段名"></result>
    </association>
</resultMap>
```

如果是一对多采用

```xml
<resultMap type="类型 如xx类" id="标识符">
	<id column="数据库字段名" property="代码中的字段名"></id> // 主键
    <result column="数据库字段名" property="代码中的字段名"></result> // 普通字段
    <collection property="代码字段名" ofType="指定的是映射到list集合属性中pojo的类型。">
        <id column="数据库字段名" property="代码中的字段名"></id> 
    	<result column="数据库字段名" property="代码中的字段名"></result>
    </collection>
</resultMap>
```

### `javaType`和`ofType`

`JavaType`和`ofType`都是用来指定对象类型的，但是`JavaType`是用来指定`pojo`中属性的类型，而`ofType`指定的是映射到list集合属性中`pojo`的类型。

### 一对一

**一对一的`POJO`如下**

```java
public class User{
    private Integer id;
    private String userName;
    private Role role;
}
/**
 * 一个用户 对应 一个 角色。 一对一
*/
```

**`MyBatis`中一对一采用**

```xml
<mapper namespace="com.itheima.dao.IAccountDao">
<!-- 建立对应关系 -->
    <resultMap type="User" id="userMap">
        <id column="id" property="id"/>
        <result column="user_name" property="userName"/>
    	<!-- 它是用于指定从表方的引用实体属性的 -->
        <association property="role" javaType="Role">
            <id column="id" property="id"/>
            <result column="role_name" property="roleName"/>
            <result column="role_desc" property="roleDesc"/>
        </association>
    </resultMap>
	<select id="findAll" resultMap="userMap">
		SQL语句
    </select>
</mapper>
```

PS：简单，所以没有例子。

### 一对多

**一对多的`POJO`**

```java
public class RoleVO implements Serializable {
    private Integer id;

    private String roleName;

    private String roleDesc;
    
    private List<UserVO> userList;
}
```

**`xml`**

```xml
<mapper namespace="com.bbxx.dao.IRoleDao">

    <resultMap id="roleMap" type="RoleVO">
        <id property="id" column="id"/>
        <result property="roleName" column="role_name"/>
        <result property="roleDesc" column="role_desc"/>
        <collection property="userList" ofType="UserVO" >
            <!-- 有两个id 确保不出错，不这样会出错的，查询的数据会不对。 -->
            <id column="u.id" property="id"></id>
            <result column="username" property="username"/>
            <result column="birthday" property="birthday"/>
            <result column="address" property="address"/>
        </collection>
    </resultMap>

    <select id="findByPrimaryKey" resultMap="roleMap">
        select r.id, r.role_desc, r.role_name, u.id, u.username, u.address, u.birthday
        from users as u,
             role as r,
             middle as m
        where r.id = #{value}
          and m.u_id = u.id
          and m.r_id = r.id
    </select>

</mapper>
```

### 多对多

一个用户对应多种角色

一个角色对应多个用户

用户与角色间多对多关系。

多对多关系需要中间表（middle），数据库系统概论中学过的。

不会，数据库方面有欠缺，先鸽了，看完书后再补，不过代码里我给了一个我认为正确的简单的多对多。

# `MyBatis`(三)

## 引言

- 延迟加载
  - 需要用的时候再加载数据
- 一级缓存
  - 一级缓存是 `SqlSession` 范围的缓存，当调用 `SqlSession` 的修改，添加，删除，`commit()，close()`等方法时，就会清空一级缓存。
- 二级缓存
  - 二级缓存是 `mapper` 映射级别的缓存，多个 `SqlSession` 去操作同一个 `Mapper` 映射的 `sql` 语句，多个 `SqlSession` 可以共用二级缓存，二级缓存是跨 `SqlSession` 的

## 延迟加载

### 一对一的延迟加载

#### 概要

举例：用户和账户之间是 一个用户对应多个账户。一个账户对应一个用户。所以账户和用户是一对一的关系。

我们对用户信息进行懒加载。

- `proerty`是`java`字段的名称
- `javaType`是查询出来的数据类型
- select是要调用的查询方法，通过这个查询方法把懒数据查询出来
- column是查询的条件，即where xx = column的值。这个column取自`resultMap`。

```xml
<association property="user" javaType="User" select="com.bbxx.dao.lazy.IUserDao.findOne" column="uid">
</association>
```

#### 具体代码

```java
public interface IAccountDao {
    // 懒加载案例，只查账号不查用户信息 一对一
    List<Account> findAll();
}

public interface IUserDao {
    // User是一对一查询 懒加载中的那个懒数据
    User findOne(Integer id);
}
```

```xml
<mapper namespace="com.bbxx.dao.lazy.IAccountDao">
	<resultMap id="accountMap" type="Account">
        <id column="id" property="id"/>
        <result column="uid" property="uid"/>
        <result column="money" property="money"/>
        <association property="user" javaType="User" select="com.bbxx.dao.lazy.IUserDao.findOne" column="uid">
        </association>
    </resultMap>

    <select id="findAll" resultMap="accountMap">
        select *
        from account
    </select>
</mapper>

<mapper namespace="com.bbxx.dao.lazy.IUserDao">
    <resultMap id="userMap" type="User">
        <id column="id" property="id"/>
        <result column="username" property="username"/>
        <result column="address" property="address"/>
        <result column="sex" property="sex"/>
        <result column="birthday" property="birthday"/>
    </resultMap>

    <select id="findOne" resultMap="userMap">
        select * from user where id = #{uid}
    </select>
</mapper>
```

### 一对多延迟加载

#### 概要

用户和账户是一对多查询。我们对“多”进行懒加载，要用时在查询。

对多的查询采用

- `proerty`是数据对应的`java`字段的名称
- `ofType`是查询出来集合中存储的数据类型
- `select`是要调用的查询方法，通过这个查询方法把懒数据查询出来
- `column`是查询的条件，即`where xx = column`的值。这个`column`取自`resultMap`。

```xml
<collection property="accounts" ofType="Account" select="com.bbxx.dao.lazy.IAccountDao.findById" column="id">
</collection>
```
#### 具体代码

```java
public interface IUserDao {
    // 懒加载 一对多查询 查询每个用户的所有账户信息
    List<User> findAll();
}

public interface IAccountDao {
	// 懒加载  对“多”的懒加载
    List<Account> findById(Integer id);
}
```

```xml
IUserDao的mapper文件
<mapper namespace="com.bbxx.dao.lazy.IUserDao">
    <resultMap id="userMap" type="User">
        <id column="id" property="id"/>
        <result column="username" property="username"/>
        <result column="address" property="address"/>
        <result column="sex" property="sex"/>
        <result column="birthday" property="birthday"/>
        <collection property="accounts" ofType="Account" select="com.bbxx.dao.lazy.IAccountDao.findById" column="id">

        </collection>
    </resultMap>

    <select id="findAll" resultMap="userMap">
        select * from user
    </select>
</mapper>

IAccountDao的mapper文件
<mapper namespace="com.bbxx.dao.lazy.IAccountDao">
<!--  一对一的查询  -->
    <resultMap id="accountMap" type="Account">
        <id column="id" property="id"/>
        <result column="uid" property="uid"/>
        <result column="money" property="money"/>
    </resultMap>

    <select id="findById" resultMap="accountMap">
        select * from account where uid = #{id}
    </select>
</mapper>
```

## 一级缓存

一级缓存是 `SqlSession` 级别的缓存，只要 `SqlSession` 没有 flush 或 close，它就存在！

```xml
<mapper namespace="com.itheima.dao.IUserDao">
<!-- 根据 id 查询 -->
<select id="findById" resultType="UsEr" parameterType="int" useCache="true">
select * from user where id = #{uid}
</select>
</mapper>
```

请自行编码验证！

## 二级缓存

好像基本不用，所有我不学，要的时候再看文档！

# `MyBatis`(四)

使用注解还是`xml`，看公司的使用习惯，目前看到的是用`xml`更多一些（看过四家公司，看过其中三家公司的代码，都是用的`xml`，没看到用注解的，只看到过个人开发用注解！）。

## 引言

- 注解配置
- 注解开发
- 注解动态`SQL`