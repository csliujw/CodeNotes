# 概览

## 常见面试题

- 什么是事务,以及事务的四大特性? 
- 事务的隔离级别有哪些,MySQL默认是哪个? 
- 内连接与左外连接的区别是什么? 
- 常用的存储引擎？InnoDB与MyISAM的区别？ 
- MySQL默认InnoDB引擎的索引是什么数据结构? 
- 如何查看MySQL的执行计划? 
- 索引失效的情况有哪些? 
- 什么是回表查询? 
- 什么是MVCC? 
- MySQL主从复制的原理是什么? 
- 主从复制之后的读写分离如何实现? 
- 数据库的分库分表如何实现?

## 规划

### 基础复习

- MySQL 概述
- SQL-DDL
- SQL-DML
- SQL-DQL
- SQL-DCL
- MySQL函数
- 约束
- 多表查询
- 事务

### 原理进阶

- MySQL体系结构
- MySQL存储引擎
- 索引
- SQL优化
- 视图
- 存储过程\函数
- 触发器
- 锁
- InnoDB存储引擎
- MySQL管理

### 运维

- MySQL日志
- MySQL复制
- MyCat概述
- 分片相关概念
- 分片配置
- 分配规则
- 读写分离

# 基础复习

## MySQL概述

> 数据库：Database，简称DB。按照一定的数据结构来组织、存储和管理数据的仓库。

> 数据库管理系统：Database Management System，一种操纵和管理数据库的大型软件，用于创建、使用和维护数据库，简称DBMS。

### 数据库相关概念

- [x] 数据库：存储数据的仓库

- [x] 数据库管理系统：Database Management System，操纵和管理数据库的大型软件，简称 DBMS。

- 关系型数据库（RDBMS）

    - 概念： 关系型数据库，是建立在关系模型基础上，由多张相互连接的二维表组成的数据库。 

    - 特点： 
        1. 使用表存储数据，格式统一，便于维护 ；

        2. 使用SQL语句操作，标准统一，使用方便； 
        3. 数据存储在磁盘中，安全

- 非关系型数据库(NoSQL)

    - 概念：Not-Only SQL，泛指非关系型数据库，是对关系型数据库的补充。 
    - 特点： 1. 数据结构灵活；2. 伸缩性强

- [x] SQL：操作关系型数据库的编程语言，是一套标准。

### MySQL数据库

> 版本

MySQL官方提供了两种不同的版本: 

- 社区版（MySQL Community Server） 免费，MySQL不提供任何技术支持
- 商业版（MySQL Enterprise Edition） 收费，可以试用30天，官方提供技术支持

> 安装

[MySQL安装指南_技术交流_牛客网 (nowcoder.com)](https://www.nowcoder.com/discuss/825171?type=all&order=recall&pos=&page=1&ncTraceId=&channel=-1&source_id=search_all_nctrack&gio_id=649BB42AA30733488E2E468CA0F7721F-1642732433430)

> 启动与停止

```shell
# 启动
net start mysql80
# 停止
net stop mysql80
```

> 客户端连接

- 用 MySQL 提供的 MySQL 8.0 Command Line Client
- 用系统自带的命令窗口，执行指令 `mysql -h 127.0.0.1 -P 3306 -u root -p password`

## SQL

### SQL通用语法

1. SQL语句可以单行或多行书写，以分号结尾。 

2. SQL语句可以使用空格/缩进来增强语句的可读性。 

3. MySQL数据库的SQL语句不区分大小写，关键字建议使用大写。 

4. 注释： •

    单行注释：-- 注释内容 或 # 注释内容(MySQL特有) 

    多行注释： /* 注释内容 */

### SQL分类

| 分类 | 全称                       | 说明                                                   |
| ---- | -------------------------- | ------------------------------------------------------ |
| DDL  | Data Definition Language   | 数据定义语言，用来定义数据库对象(数据库，表，字段)     |
| DML  | Data Manipulation Language | 数据操作语言，用来对数据库表中的数据进行增删改         |
| DQL  | Data Query Language        | 数据查询语言，用来查询数据库中表的记录                 |
| DCL  | Data Control Language      | 数据控制语言，用来创建数据库用户、控制数据库的访问权限 |

### DDL

- 数据库操作
    - show databases;
    - create database 数据库名
    - use 数据库名;
    - select databases();
    - drop databases 数据库名
- 表操作
    - show tables;
    - create table 表名(字段 字段类型, 字段 字段类型);
    - desc 表名;
    - show create table 表名;
    - alter table 表名 add/modify/drop/rename;
    - drop table 表名;

#### 数据库操作

> 查询

查询所有数据库 `show databases;`
查询当前数据库 `select database();`

> 创建

```sql
create database [IF NOT EXISTS] 数据库名 [DEFAULT CHARSET 字符集] [COLLATE 排序规则];
# mysql数据库中 utf8 字符集存储的长度是三个字节，有些特殊字符占四个字节 
# mb4则支持四个字节的（语言表达有问题），默认好像就是utf8mb4
create database if not exists demo default charset utf8mb4;
```

> 删除

```sql
drop database 数据库名;
drop database demo;
```

> 使用

```sql
use 数据库名;
use demo;
```

#### 表操作-查询

查询当前数据库所有表 `show tables`

查询表结构 `desc 表名`

查询指定表的建表语句 `show create table 表名`

![image-20220131155715763](img\image-20220131155715763.png)

#### 表操作-创建

```sql
create table 表名(
	字段1 字段类型 [comment 字段1注释],
	字段2 字段类型 [comment 字段2注释],
)[comment 表注释];

drop table if esists tb_user; # 存在则先删除

create table tb_user(
	id int primary key,
	name varchar(30) not null
)
```

#### 数据类型

MySQL中的数据类型有很多，主要分为三类：数值类型、字符串类型、日期时间类型。

<div style="text-align:center;"><h3>数值类型</h3></div>

| 分类         | 类型     | 大小                                                  | 有符号(SIGNED)范围                                         | 无符号(UNSIGNED)范围 | 描述               |
| ------------ | -------- | ----------------------------------------------------- | ---------------------------------------------------------- | -------------------- | ------------------ |
| 数值类型     | TINYINT  | 1  byte                                               | (-128，127)                                                | (0，255)             | 小整数值           |
| SMALLINT     | 2  bytes | (-32768，32767)                                       | (0，65535)                                                 | 大整数值             | 大整数值           |
| MEDIUMINT    | 3  bytes | (-8388608，8388607)                                   | (0，16777215)                                              | 大整数值             | 大整数值           |
| INT或INTEGER | 4  bytes | (-2147483648，2147483647)                             | (0，4294967295)                                            | 大整数值             | 大整数值           |
| BIGINT       | 8  bytes | (-2^63，2^63-1)                                       | (0，2^64-1)                                                | 极大整数值           | 极大整数值         |
| FLOAT        | 4  bytes | (-3.402823466 E+38，3.402823466351 E+38)              | 0 和 (1.175494351  E-38，3.402823466 E+38)                 | 单精度浮点数值       | 单精度浮点数值     |
| DOUBLE       | 8  bytes | (-1.7976931348623157 E+308，1.7976931348623157 E+308) | 0 和  (2.2250738585072014 E-308，1.7976931348623157 E+308) | 双精度浮点数值       | 双精度浮点数值     |
| DECIMAL      |          | 依赖于M(精度)和D(标度)的值                            | 依赖于M(精度)和D(标度)的值                                 | 小数值(精确定点数)   | 小数值(精确定点数) |

<div style="text-align:center;"><h3>字符串类型</h3></div>

| 分类       | 类型                  | 大小                         | 描述                         |
| ---------- | --------------------- | ---------------------------- | ---------------------------- |
| 字符串类型 | CHAR                  | 0-255 bytes                  | 定长字符串                   |
| VARCHAR    | 0-65535 bytes         | 变长字符串                   | 变长字符串                   |
| TINYBLOB   | 0-255 bytes           | 不超过255个字符的二进制数据  | 不超过255个字符的二进制数据  |
| TINYTEXT   | 0-255 bytes           | 短文本字符串                 | 短文本字符串                 |
| BLOB       | 0-65 535 bytes        | 二进制形式的长文本数据       | 二进制形式的长文本数据       |
| TEXT       | 0-65 535 bytes        | 长文本数据                   | 长文本数据                   |
| MEDIUMBLOB | 0-16 777 215 bytes    | 二进制形式的中等长度文本数据 | 二进制形式的中等长度文本数据 |
| MEDIUMTEXT | 0-16 777 215 bytes    | 中等长度文本数据             | 中等长度文本数据             |
| LONGBLOB   | 0-4 294 967 295 bytes | 二进制形式的极大文本数据     | 二进制形式的极大文本数据     |
| LONGTEXT   | 0-4 294 967 295 bytes | 极大文本数据                 | 极大文本数据                 |

char性能较好，varchar性能较差；varchar需要数一下字符串多长，但是相比char更节省空间，varchar是字符串是多大就用多大的空间，char是不管字符串多长，都用设定好的大小去存储一个字符串。 

<div style="text-align:center;"><h3>时间类型</h3></div>

| 分类      | 类型 | 大小                                       | 范围                      | 格式                     | 描述                     |
| --------- | ---- | ------------------------------------------ | ------------------------- | ------------------------ | ------------------------ |
| 日期类型  | DATE | 3                                          | 1000-01-01 至  9999-12-31 | YYYY-MM-DD               | 日期值                   |
| TIME      | 3    | -838:59:59 至  838:59:59                   | HH:MM:SS                  | 时间值或持续时间         | 时间值或持续时间         |
| YEAR      | 1    | 1901 至 2155                               | YYYY                      | 年份值                   | 年份值                   |
| DATETIME  | 8    | 1000-01-01 00:00:00 至 9999-12-31 23:59:59 | YYYY-MM-DD HH:MM:SS       | 混合日期和时间值         | 混合日期和时间值         |
| TIMESTAMP | 4    | 1970-01-01 00:00:01 至 2038-01-19 03:14:07 | YYYY-MM-DD HH:MM:SS       | 混合日期和时间值，时间戳 | 混合日期和时间值，时间戳 |

#### 习题

根据需求创建表(设计合理的数据类型、长度) 设计一张员工信息表，要求如下： 

1. 编号（纯数字） 
2. 员工工号 (字符串类型，长度不超过10位) 
3. 员工姓名（字符串类型，长度不超过10位） 
4. 性别（男/女，存储一个汉字） 
5. 年龄（正常人年龄，不可能存储负数） 
6. 身份证号（二代身份证号均为18位，身份证中有X这样的字符） 
7. 入职时间（取值年月日即可）

```sql
create table tb_emp(
	id int primary key, -- 主键索引
    number varchar(10),
    name varchar(10),
    sex char(1),
    age tinyint unsigned, -- 无符号整数，范围正常。
    id_card char(18),
    emyloyee_time date
) engine=INNODB charset utf8mb4;
```

#### 表操作-修改

添加字段：`alter table 表名 add 字段名 类型(长度) [comment 注释] [约束];`
`alter table tb_emp add nickname varchar(20) comment '昵称';` 

![image-20220131162848796](img\image-20220131162848796.png)

修改数据类型：`alter table 表名 modify 字段名 新数据类型(长度);`
`alter table tb_emp modify nickname int;`
![image-20220131163101591](img\image-20220131163101591.png)

修改字段名和字段类型：`alter table 表名 change 旧字段名 新字段名 类型(长度) [comment 注释] [约束]`
`alter table tb_emp change nickname username varchar(30) comment '用户名' not null;`
![image-20220131163452393](img\image-20220131163452393.png)

删除字段：`alter table 表名 drop 字段名`;
`alter table tb_emp drop username;`

修改表名：`alter table 表名 rename to 新表名;`
`alter table tb_emp rename emp;`

#### 表操作-删除

删除表  `drop table [if exists] 表名`

删除指定表，并重新创建该表 `truncate table 表名;`

表删除的效率比较

- drop：删除内容和定义，释放空间。简单来说就是**把整个表去掉**。以后要新增数据是不可能的,除非新增一个表
- truncate：只是清空表数据，会保留表的数据结构。
- delete：是删除整个表的数据,但是是系统一行一行地删，效率比truncate低。delete 语句每次删除一行，会在事务日志中为所删除的每行记录一项。所以可以对delete操作进行roll back，效率也低。

### DML

Data Manipulation Language(数据操作语言)，用来对数据库中表的数据记录进行增删改操作。

- 添加数据：`insert into 表名 (字段名1, 字段名2) values(值1, 值2), (值1, 值2), (值1,值2);`
- 修改数据：`update 表名 set 字段名1=值1, 字段名2=值2,.... [where 条件];`
- 删除数据：`delete from 表名 [where 条件]`

添加数据（INSERT）；修改数据（UPDATE）；删除数据（DELETE）

> 添加数据

给指定字段添加数据
insert into table_name (字段1, 字段2, ...) values(值1, 值2);

给全部字段添加数据
insert to 表名 values(值1,值2,...)

批量添加
insert into 表名 (字段名1, 字段名2) values(值1, 值2), (值1, 值2), (值1,值2);
insert into 表名 (字段名1, 字段名2,...) values(值1, 值2,...), (值1, 值2,...), (值1,值2,...);

<span style="color:red">**注意：**</span>插入数据时，指定的字段顺序需要与值的顺序是一一对应的。字符串和日期型数据应该包含在引号中。插入的数据大小，应该在字段的规定范围内

> 修改数据

`update 表名 set 字段名1=值1, 字段名2=值2,.... [where 条件];`

<span style="color:red">**注意：**</span>修改语句的条件可以有，也可以没有，如果没有条件，则会修改整张表的所有数据。

> 删除数据

`delete from 表名 [where 条件]`

<span style="color:red">**注意：**</span>DELETE 语句的条件可以有，也可以没有，如果没有条件，则会删除整张表的所有数据。 DELETE 语句不能删除某一个字段的值(可以使用UPDATE)。

### DQL

Data Query Language(数据查询语言)，数据查询语言，用来查询数据库中表的记录。

查询的基本语法如下：

```sql
select
	字段列表
from
	表名列表
where
	条件列表
group by
	分组字段列表
having
	分组后条件列表
order by
	排序字段列表
limit
	分页参数
```

- 基本查询
- 条件查询（where）
- 聚合函数（count、max、min、avg、sum）
- 分组查询（group by）
- 排序查询（order by）
- 分页查询（limit)

#### 基本查询

```sql
select * from tb_emp;
select name from tb_emp;
```

#### 条件查询

```sql
select 字段列表 from 表名 where 条件列表;
```

<div style="text-align:center;"><h3>条件</h3></div>

![image-20220131170613284](img\image-20220131170613284.png)

![image-20220131170637473](img\image-20220131170637473.png)

> 示例

```sql
# 表数据
create table dept(
    id   int auto_increment comment 'ID' primary key,
    name varchar(50) not null comment '部门名称'
)comment '部门表';
INSERT INTO dept (id, name) VALUES (1, '研发部'), (2, '市场部'),(3, '财务部'), (4, '销售部');


create table emp(
    id  int auto_increment comment 'ID' primary key,
    name varchar(50) not null comment '姓名',
    age  int comment '年龄',
    idcard char(18) default '120235681203256878',
    dept_id int comment '部门ID'
)comment '员工表';
INSERT INTO emp (id, name, age, dept_id) VALUES (1, '张无忌', 20, 1),(2, '杨逍', 33, 1),(3, '赵敏', 18, 2), (4, '常遇春', 43, 2),(5, '小昭', 19, 3),(6, '韦一笑', 48, 3);
```

```sql
-- 1. 查询年龄 =88 的员工信息
select * from emp where age = 88;

-- 2. 查询年龄 <20 的员工信息
select * from emp where age < 20;

-- 3. 查询年龄 <=20 的员工信息
select * from emp where age <= 20;

-- 4. 查询没有身份证号的员工信息
select * from emp where idcard is null;

-- 5. 查询有身份证号的员工信息
select * from emp where idcard is not null;

-- 6. 查询年龄不等于 88 的员工信息
select * from emp where age != 88;
select * from emp where age <> 88;

-- 7. 查询年龄在 15<=age<=20 的员工信息
select * from emp where age between 15 and 20;
select * from emp where age>=15 && age<=20;
select * from emp where age>=15 and age<=20;


-- 8. 查询性别为 女 且年龄小于25的员工信息
select * from emp where sex='女' and age<25;

-- 9. 查询年龄 为18 或 20 或 40 的员工信息
select * from emp where age=18 or age=20 or age=40;
select * from emp where age in (18,20,40);

-- 10. 查询姓名为两个字的员工信息
select * from emp where name like '__';

-- 11. 查询身份证号最后一位是X的员工信息
select * from emp where name like '_________________X';
select * from emp where name like '%X';
```

#### 聚合函数

将一列数据作为一个整体，进行纵向计算 。常见聚合函数如下：

| 函数  | 功能     |
| ----- | -------- |
| count | 统计数量 |
| max   | 最大值   |
| min   | 最小值   |
| avg   | 平均值   |
| sum   | 求和     |

语法：`select 聚合函数(字段列表) from 表名;`

<span style="color:red">**注意：**</span>null 值不参与聚合函数运算。

**count函数**

- COUNT(字段) 会统计该字段在表中出现的次数，忽略字段为 null 的情况。即不统计字段为 null 的记录。 
- COUNT(\*) 则不同，它执行时返回检索到的行数的计数，不管这些行是否包含null值，*
- COUNT(1)跟COUNT(\*)类似，不将任何列是否null列入统计标准，仅用1代表代码行，所以在统计结果的时候，不会忽略列值为NULL的行。
- COUNT(1)和 COUNT(\*)表示的是直接查询符合条件的数据库表的行数。而COUNT(字段)表示的是查询符合条件的列的值，并判断不为NULL的行数的累计，效率自然会低一点。从效率层面说，COUNT(\*) ≈ COUNT(1) > COUNT(字段)，又因为 COUNT(\*)是SQL92定义的标准统计数的语法，**我们建议使用 COUNT(\*)。**

#### 分组查询

`select 字段列表 from 表名 [where 条件] group by 分组字段名 [having 分组后过滤条件]`

<span style="color:red">**where和having的区别：**</span>

- 执行时机不同：where是分组之前进行过滤，不满足where条件，不参与分组；而having是分组之后对结果进行过滤。
- 判断条件不同：where不能对聚合函数进行判断，而having可以。

```sql
-- 分组查询
-- 1.根据性别分组，统计男性员工和女性员工的数量
select sex,count(*) from emp group by sex;
-- 2.根据年龄分组，统计男性员工和女性员工的平均年龄
select sex,avg(age) from emp group by sex;
-- 3.查询年龄小于45的员工，并根据工作地址分组，获取员工数量大于等于3的工作地址
select workaddress,count(*) as address_count from emp where age<45 group by workaddress having address_count>=3;
```

**注意：**

-  执行顺序: where > 聚合函数 > having 。 
- 分组之后，查询的字段一般为聚合函数和分组字段，查询其他字段无任何意义。

#### 排序查询

`select 字段列表 from 表名 order by 字段1 排序方式1，字段2 排序方式2;`

- 排序方式
    - ASC：升序（默认值）
    - DESC：降序
- 注意：如果是多字段排序，当第一个字段值相同时，才会根据第二个字段进行排序。

#### 分页查询

`select 字段列表 from 表名 limit 起始索引,查询记录数;`

```sql
select * from emp limit 0,5;
```

<span style="color:red">**注意：**</span>

- 起始索引从0开始，起始索引 = （查询页码 - 1）* 每页显示记录数。
- 分页查询是数据库的方言，不同的数据库有不同的实现，MySQL中是LIMIT。
- 如果查询的是第一页数据，起始索引可以省略，直接简写为 limit 10。

#### 习题

按照需求完成如下DQL语句编写 

1. 查询年龄为20,21,22,23岁的员工信息。 
2. 查询性别为男 ，并且年龄在 20-40 岁(含)以内的姓名为三个字的员工。 
3. 统计员工表中, 年龄小于60岁的 , 男性员工和女性员工的人数。 
4. 查询所有年龄小于等于35岁员工的姓名和年龄，并对查询结果按年龄升序排序，如果年龄相同按入职时间降序排序。 
5. 查询性别为男，且年龄在20-40 岁(含)以内的前5个员工信息，对查询的结果按年龄升序排序，年龄相同按入职时间升序排序。

```sql
select * from emp where age in(20,21,22,23);

select sex,age,name from emp where sex='男' and (age between 20 and 40) and name like '___';

select sex,count(*) from emp where age<60 group by sex;

select name,age from emp where age<=35 order by age asc,entrydate desc;

select * from emp where sex='男' and age between 20 and 40 order by age asc,entrydate desc limit 0,5;
```

#### 执行顺序

![image-20220201004522563](img\image-20220201004522563.png)

### DCL

DCL英文全称是Data Control Language (数据控制语言)，用来管理数据库用户 、控制数据库的访问权限。

- 用户管理
    - `create user '用户'@'主机名' identified by '密码';`
    - `alter user '用户'@'主机名' identified  with mysql_native_password by '密码';`
    - `drop user '用户'@'主机名';`
- 权限控制
    - `grant 权限列表 on 数据库名.表名 to '用户'@'主机名';`
    - `revoke 权限列表 on 数据库名.表名 from '用户'@'主机名';`

#### 管理用户

查询用户：

```sql
use mysql;
select * from user;
```

创建用户：

```sql
create user '用户名'@'主机名' identified by '密码';
# 虽然创建了用户 user2 但是没有对数据库的访问权限，只能登录
create user 'user2'@'localhost' identified by 'root';
# 任意主机都可以登录。 % 是通配符，表示任意
create user 'user2'@'%' identified by 'root';
```

修改用户密码

```sql
alter user '用户名'@'主机名' identified with mysql_native_password by '新密码';

 alter user 'user2'@'localhost' identified with mysql_native_password by '1234'
```

删除用户

```sql
drop user 'user2'@'localhost';
```

**注意：** 

- 主机名可以使用 % 通配。
- 这类SQL开发人员操作的比较少，主要是DBA（ Database Administrator 数据库管理员）使用。

#### 权限控制

MySQL中定义了很多种权限，<a href="https://dev.mysql.com/doc/refman/8.0/en/privileges-provided.html#priv_all">官方文档</a> ，但是常用的就以下几种：

| 权限                | 说明               |
| ------------------- | ------------------ |
| ALL, ALL PRIVILEGES | 所有权限           |
| SELECT              | 查询数据           |
| INSERT              | 插入数据           |
| UPDATE              | 修改数据           |
| DELETE              | 删除数据           |
| ALTER               | 修改表             |
| DROP                | 删除数据库/表/视图 |
| CREATE              | 创建数据库/表      |

查询权限：

```sql
show grants for '用户名'@'主机';
```

授予权限

```sql
grant 权限列表 on 数据库名.表名 to '用户名'@'主机';
# 授予 user2 用户demo数据库所有表的所有权限。
grant all on demo.* to 'user2'@'%'; 
```

撤销权限

```sql
revoke 权限列表 on 数据库名.表名 to '用户名'@'主机';

revoke all on demo.* to 'user2'@'%';
```

## 函数

- 字符串函数
    - `concat,lower,upper,lpad,rpad,trim,substring`
- 数值函数
    - `ceil,floor,mod,rand,round`
- 日期函数
    - `curdate,curtime,now,year,month,day,date_add,datediff`
- 流程函数
    - `if,ifnull,case[...] when... then... else... end`

### 字符串函数

常用的字符串函数如下：

| 函数                     | 功能                                                         |
| ------------------------ | ------------------------------------------------------------ |
| concat(s1,s2,...sn)      | 字符串拼接，将 s1,s2,...sn 拼接成一个字符串                  |
| lower(str)               | 将字符串str全部转为小写                                      |
| upper(str)               | 将字符串str全部转为大写                                      |
| LPAD(str,n,pad)          | 左填充，用字符串 pad 对 str 的左边进行填充，达到 n 个字符串长度 |
| RPAD(str,n,pad)          | 右填充，用字符串 pad 对 str 的右边进行填充，达到 个字符串长度 |
| TRIM(str)                | 去掉字符串头部和尾部的空格（LTRIM、RTRIM）                   |
| SUBSTRING(str,start,len) | 返回从字符串 str 从 start 位置起的 len 个长度的字符串        |

```sql
mysql> select concat('s1','asf','demo');
+---------------------------+
| concat('s1','asf','demo') |
+---------------------------+
| s1asfdemo                 |
+---------------------------+
1 row in set (0.00 sec)

mysql> select lower("BIG");
+--------------+
| lower("BIG") |
+--------------+
| big          |
+--------------+
1 row in set (0.01 sec)

mysql> select upper("small");
+----------------+
| upper("small") |
+----------------+
| SMALL          |
+----------------+
1 row in set (0.00 sec)

mysql> select LPAD('00101',32,'0');
+----------------------------------+
| LPAD('00101',32,'0')             |
+----------------------------------+
| 00000000000000000000000000000101 |
+----------------------------------+
1 row in set (0.01 sec)

mysql> select RPAD('320.0',6,'0');
+---------------------+
| RPAD('320.0',6,'0') |
+---------------------+
| 320.00              |
+---------------------+
1 row in set (0.07 sec)

mysql> select trim('   hello world  ');
+--------------------------+
| trim('   hello world  ') |
+--------------------------+
| hello world              |
+--------------------------+
1 row in set (0.01 sec)

mysql> select substring('hello world java','2','3');
+---------------------------------------+
| substring('hello world java','2','3') |
+---------------------------------------+
| ell                                   |
+---------------------------------------+
1 row in set (0.00 sec)


# 由于业务需求变更，企业员工的工号，统一为5位数，目前不足5位数的全部在前面补0。比如： 1号员
# 工的工号应该为00001
update emp set workernumber=lpad(workernumber,5,'0');
```

### 数值函数

| 函数       | 功能                               |
| ---------- | ---------------------------------- |
| CELL(x)    | 向上取整                           |
| FLOOR(x)   | 向下取整                           |
| MOD(x,y)   | 返回x/y的模                        |
| RAND()     | 返回0~1内的随机数                  |
| ROUND(x,y) | 求参数x的四舍五入的值，保留y位小数 |

```mysql
-- 通过数据库的函数，生成一个六位数的随机验证码
mysql> select LPAD(ROUND(rand()*1000000),6,'0');
+-----------------------------------+
| LPAD(ROUND(rand()*1000000),6,'0') |
+-----------------------------------+
| 429655                            |
+-----------------------------------+
1 row in set (0.00 sec)

mysql> select LPAD(ROUND(rand()*1000000),6,'0');
+-----------------------------------+
| LPAD(ROUND(rand()*1000000),6,'0') |
+-----------------------------------+
| 063800                            |
+-----------------------------------+
1 row in set (0.00 sec)

mysql> select LPAD(ROUND(rand()*1000000),6,'0');
+-----------------------------------+
| LPAD(ROUND(rand()*1000000),6,'0') |
+-----------------------------------+
| 030035                            |
+-----------------------------------+
1 row in set (0.00 sec)
```

### 日期函数

| 函数                              | 功能                                                |
| --------------------------------- | --------------------------------------------------- |
| CURDATE()                         | 返回当前日期                                        |
| CURTIME()                         | 返回当前时间                                        |
| NOW()                             | 返回当前日期和时间                                  |
| YEAR(date)                        | 获取指定 date 的年份                                |
| MONTH(date)                       | 获取指定 date 的月份                                |
| DAY(date)                         | 获取指定 date 的日期                                |
| DATE_ADD(date,INTERVAL expr type) | 返回一个日期 时间值加上一个时间间隔 expr 后的时间值 |
| DATEDIFF(date1,date2)             | 返回起始时间 date1 和 结束时间 data2 之间的天数     |

```SQL
mysql> select curdate();
+------------+
| curdate()  |
+------------+
| 2022-02-01 |
+------------+
1 row in set (0.01 sec)

mysql> select curtime();
+-----------+
| curtime() |
+-----------+
| 14:41:15  |
+-----------+
1 row in set (0.01 sec)

mysql> select now();
+---------------------+
| now()               |
+---------------------+
| 2022-02-01 14:41:28 |
+---------------------+
1 row in set (0.00 sec)

mysql> select year(now());
+-------------+
| year(now()) |
+-------------+
|        2022 |
+-------------+
1 row in set (0.00 sec)

mysql> select year('2021-11-11 12:22');
+--------------------------+
| year('2021-11-11 12:22') |
+--------------------------+
|                     2021 |
+--------------------------+
1 row in set (0.00 sec)

mysql> select date_add(now(),interval 70 day);
+---------------------------------+
| date_add(now(),interval 70 day) |
+---------------------------------+
| 2022-04-12 14:43:23             |
+---------------------------------+
1 row in set (0.00 sec)

mysql> select date_add(now(),interval 70 month);
+-----------------------------------+
| date_add(now(),interval 70 month) |
+-----------------------------------+
| 2027-12-01 14:43:29               |
+-----------------------------------+
1 row in set (0.00 sec)

mysql> select date_add(now(),interval 70 hour);
+----------------------------------+
| date_add(now(),interval 70 hour) |
+----------------------------------+
| 2022-02-04 12:43:36              |
+----------------------------------+
1 row in set (0.00 sec)

mysql> select datediff('2022-2-1','2021-1-1');
+---------------------------------+
| datediff('2022-2-1','2021-1-1') |
+---------------------------------+
|                             396 |
+---------------------------------+
1 row in set (0.01 sec)

mysql> select datediff('2022-2-1','2023-1-1');
+---------------------------------+
| datediff('2022-2-1','2023-1-1') |
+---------------------------------+
|                            -334 |
+---------------------------------+
1 row in set (0.00 sec)
```

```sql
-- 查询所有员工的入职天数，并根据入职天数倒序排序。
select name, datediff(curdate(),entrydate) ddiff from emp order by ddiff desc;
```

### 流程函数

在SQL语句中实现条件筛选，从而提高语句的效率。

| 函数                                                       | 功能                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| IF(value, t, f)                                            | 如果 value 为 true，则返回 t，否则返回 f                     |
| IFNULL(value1, value2)                                     | 如果 value 不为空，返回 value1，否则返 value2                |
| CASE WHEN [val1] THEN [res1] ... ELSE [default] END        | 如果 val1 为 true，返回 res1，... 否则返回 default 默认值    |
| CASE [expr] WHEN [val1] THEN [res1] ... ELSE [default] END | 如果 expr 的值等于 val，返回 res1 ， 否则返回 default 默认值 |

```sql
mysql> select ifnull('','Default');
+----------------------+
| ifnull('','Default') |
+----------------------+
|                      |
+----------------------+
1 row in set (0.00 sec)

mysql> select ifnull(null,'Default');
+------------------------+
| ifnull(null,'Default') |
+------------------------+
| Default                |
+------------------------+
1 row in set (0.00 sec)
-- case when then else end
-- 查询emp表的员工，姓名和工作地址(北京/上海 ---> 一线城市，其他 ---> 二线城市)
select name,
	(case workaddress when '北京' then '一线城市'
	when '上海' then '一线城市'
	else '二线城市' end) as 工作地址
from emp;
```

```sql
- 统计班级各个学员的成绩，展示的规则如下：
- • >= 85，展示优秀
- • >= 60，展示及格
- • 否则，展示不及格
create table score(
	id int comment 'ID',
    name varchar(20) comment '姓名',
    math int comment '数学',
    english int comment '英语',
    chinese int comment '语文'
);
insert into score(id,name,math,english,chinese) 
values(1,'tom',67,88,95),
(2,'rose',23,66,90),
(3,'jack',56,98,76);

select id,
name,
(case when math>=85 then '优秀' when math>=60 then '及格' else '不及格' end) '数学',
(case when english>=85 then '优秀' when english>=60 then '及格' else '不及格' end) '英语',
(case when chinese>=85 then '优秀' when chinese>=60 then '及格' else '不及格' end) '语文'
from score;
```

## 约束

- 非空约束： NOT NULL 
- 唯一约束： UNIQUE 
- 主键约束： PRIMARY KEY (自增: AUTO_INCREMENT)
-  默认约束： DEFAULT 
- 检查约束： CHECK 
- 外键约束：FOREIGN KEY

### 概念

- 概念：约束是作用于表中字段上的规则，用于限制存储在表中的数据。
- 目的：保证数据库中数据的正确、有效性和完整性。
- 分类：

| 约束                     | 描述                                                     | 关键字      |
| ------------------------ | -------------------------------------------------------- | ----------- |
| 非空约束                 | 限制该字段的数据不能为 NULL                              | NOT NULL    |
| 唯一约束                 | 保证该字段的所有数据都是唯一、不重复的                   | UNIQUE      |
| 主键约束                 | 主键是一行数据的唯一标识，要求非空且唯一                 | PRIMARY KEY |
| 默认约束                 | 保存数据时，如果未指定该字段的值，则采用默认值           | DEFAULT     |
| 检测约束(8.0.16版本之后) | 保证字段值满足某一个条件                                 | CHECK       |
| 外键约束                 | 用来让两张表的数据之间建立连接，保证数据的一致性和完整性 | FOREIGN KEY |

<span style="color:red">**注意：约束是作用于表中字段上的，可以在创建表/修改表的时候添加约束。**</span>

### 演示

根据需求，完成表结构的创建

| 字段名 | 字段含义    | 字段类型    | 约束条件                  |
| ------ | ----------- | ----------- | ------------------------- |
| id     | ID 唯一标识 | int         | 主键，且自动增长          |
| name   | 姓名        | varchar(10) | 不为空，且唯一            |
| age    | 年龄        | int         | 大于0，并且小于等于120    |
| status | 状态        | char(1)     | 如果没有指定该值，默认为1 |
| gender | 性别        | char(2)     | 无                        |

```sql
create table tb_user(
id int AUTO_OMCRE,EMT PRIMARY KEY COMMENT '',
name varchar(10) NOT NULL UNIQUE COMMENT '',
age int check(age>0 && age<=120) COMMENT '',
status char(1) default '1' COMMENT '',
gender char(1) COMMENT ''
);
```

### 外键约束

外键用来让两张表的数据之间建立连接，从而保证数据的一致性和完整性。

![image-20220201153732702](img\image-20220201153732702.png)

注意：目前上述的两张表，在数据库层面，并未建立外键关联，所以是无法保证数据的一致性和完整性的。

语法：

```sql
create table 表名(
	字段名 数据类型,
    ...
    [constraint] [外键名] foreign key (外键字段名) references 主表(主表列名)
);
alter table 表名 add constraint 外键名称 foreign key(外键字段名) references 主表(主表列名);

-- 删除外键
alter table 表名 drop foreign key 外键名称;
```

![image-20220201154122979](img\image-20220201154122979.png)

```sql
alter table 表名 add constraint 外键名称 foreign key(外键字段名) references 主表(主表字段名)
on update cascade on delete cascade;
```

### 演示

```mysql
-- ------------------------------------------------------------------- 约束演示 ----------------------------------------------
create table user(
    id int primary key auto_increment comment '主键',
    name varchar(10) not null unique comment '姓名',
    age int check ( age > 0 && age <= 120 ) comment '年龄',
    status char(1) default '1' comment '状态',
    gender char(1) comment '性别'
) comment '用户表';

-- 插入数据
insert into user(name,age,status,gender) values ('Tom1',19,'1','男'),('Tom2',25,'0','男');
insert into user(name,age,status,gender) values ('Tom3',19,'1','男');

insert into user(name,age,status,gender) values (null,19,'1','男');
insert into user(name,age,status,gender) values ('Tom3',19,'1','男');

insert into user(name,age,status,gender) values ('Tom4',80,'1','男');
insert into user(name,age,status,gender) values ('Tom5',-1,'1','男');
insert into user(name,age,status,gender) values ('Tom5',121,'1','男');

insert into user(name,age,gender) values ('Tom5',120,'男');


-- --------------------------------------------- 约束 (外键) -------------------------------------
-- 准备数据
create table dept(
    id   int auto_increment comment 'ID' primary key,
    name varchar(50) not null comment '部门名称'
)comment '部门表';
INSERT INTO dept (id, name) VALUES (1, '研发部'), (2, '市场部'),(3, '财务部'), (4, '销售部'), (5, '总经办');


create table emp(
    id  int auto_increment comment 'ID' primary key,
    name varchar(50) not null comment '姓名',
    age  int comment '年龄',
    job varchar(20) comment '职位',
    salary int comment '薪资',
    entrydate date comment '入职时间',
    managerid int comment '直属领导ID',
    dept_id int comment '部门ID'
)comment '员工表';

INSERT INTO emp (id, name, age, job,salary, entrydate, managerid, dept_id) VALUES
            (1, '金庸', 66, '总裁',20000, '2000-01-01', null,5),(2, '张无忌', 20, '项目经理',12500, '2005-12-05', 1,1),
            (3, '杨逍', 33, '开发', 8400,'2000-11-03', 2,1),(4, '韦一笑', 48, '开发',11000, '2002-02-05', 2,1),
            (5, '常遇春', 43, '开发',10500, '2004-09-07', 3,1),(6, '小昭', 19, '程序员鼓励师',6600, '2004-10-12', 2,1);

-- 添加外键
alter table emp add constraint fk_emp_dept_id foreign key (dept_id) references dept(id);

-- 删除外键
alter table emp drop foreign key fk_emp_dept_id;

-- 外键的删除和更新行为
alter table emp add constraint fk_emp_dept_id foreign key (dept_id) references dept(id) on update cascade on delete cascade ;

alter table emp add constraint fk_emp_dept_id foreign key (dept_id) references dept(id) on update set null on delete set null ;



-- -------------------------------- 多表关系 演示 ---------------------------------------------

-- 多对多 ----------------
create table student(
    id int auto_increment primary key comment '主键ID',
    name varchar(10) comment '姓名',
    no varchar(10) comment '学号'
) comment '学生表';
insert into student values (null, '黛绮丝', '2000100101'),(null, '谢逊', '2000100102'),(null, '殷天正', '2000100103'),(null, '韦一笑', '2000100104');


create table course(
    id int auto_increment primary key comment '主键ID',
    name varchar(10) comment '课程名称'
) comment '课程表';
insert into course values (null, 'Java'), (null, 'PHP'), (null , 'MySQL') , (null, 'Hadoop');


create table student_course(
    id int auto_increment comment '主键' primary key,
    studentid int not null comment '学生ID',
    courseid  int not null comment '课程ID',
    constraint fk_courseid foreign key (courseid) references course (id),
    constraint fk_studentid foreign key (studentid) references student (id)
)comment '学生课程中间表';

insert into student_course values (null,1,1),(null,1,2),(null,1,3),(null,2,2),(null,2,3),(null,3,4);


-- --------------------------------- 一对一 ---------------------------
create table tb_user(
    id int auto_increment primary key comment '主键ID',
    name varchar(10) comment '姓名',
    age int comment '年龄',
    gender char(1) comment '1: 男 , 2: 女',
    phone char(11) comment '手机号'
) comment '用户基本信息表';

create table tb_user_edu(
    id int auto_increment primary key comment '主键ID',
    degree varchar(20) comment '学历',
    major varchar(50) comment '专业',
    primaryschool varchar(50) comment '小学',
    middleschool varchar(50) comment '中学',
    university varchar(50) comment '大学',
    userid int unique comment '用户ID',
    constraint fk_userid foreign key (userid) references tb_user(id)
) comment '用户教育信息表';


insert into tb_user(id, name, age, gender, phone) values
        (null,'黄渤',45,'1','18800001111'),
        (null,'冰冰',35,'2','18800002222'),
        (null,'码云',55,'1','18800008888'),
        (null,'李彦宏',50,'1','18800009999');

insert into tb_user_edu(id, degree, major, primaryschool, middleschool, university, userid) values
        (null,'本科','舞蹈','静安区第一小学','静安区第一中学','北京舞蹈学院',1),
        (null,'硕士','表演','朝阳区第一小学','朝阳区第一中学','北京电影学院',2),
        (null,'本科','英语','杭州市第一小学','杭州市第一中学','杭州师范大学',3),
        (null,'本科','应用数学','阳泉第一小学','阳泉区第一中学','清华大学',4);
```



## 多表查询

- 多表关系：一对一、一对多、多对多
- 多表查询
    - 笛卡儿积：两个集合的所有的组合情况。
    - 连接查询：
        - 左外连接：查询左表所有数据，以及两张表交集部分数据
        - 右外连接：查询右表所有数据，以及两张表交集部分数据
        - 自连接：当前表与自身的连接查询，自连接必须使用表别名
        - 内连接：相当于查询A、B交集部分数据
- 子查询：标量子查询（子查询结果为单个值）；列子查询(子查询结果为一列)；行子查询(子查询结果为一行)；表子查询(子查询结果为多行多列)

 ### 多表关系

项目开发中，在进行数据库表结构设计时，会根据业务需求及业务模块之间的关系，分析并设计表结构，由于业务之间相互关联，所 以各个表结构之间也存在着各种联系，基本上分为三种：

- 一对多(多对一) 
- 多对多 
- 一对一

> 一对多（多对一）

➢ 案例: 部门与员工的关系 
➢ 关系: 一个部门对应多个员工，一个员工对应一个部门 
➢ 实现: 在多的一方建立外键，指向一的一方的主键
![image-20220201162055837](img\image-20220201162055837.png)

> 多对多

案例: 学生与课程的关系 
➢ 关系: 一个学生可以选修多门课程，一门课程也可以供多个学生选择 
➢ 实现: 建立第三张中间表，中间表至少包含两个外键，分别关联两方主键

![image-20220201162206455](img\image-20220201162206455.png)

> 一对一

➢ 案例: 用户与用户详情的关系 
➢ 关系: 一对一关系，多用于单表拆分，将一张表的基础字段放在一张表中，其他详情字段放在另一张表中，以提升操作效率 
➢ 实现: 在任意一方加入外键，关联另外一方的主键，并且设置外键为唯一的(UNIQUE)
![image-20220201162313709](img\image-20220201162313709.png)

### 多表查询概述

多表查询是指: 指从多张表中查询数据 

笛卡尔积: 笛卡尔乘积是指在数学中，两个集合A集合和B集合的所有组合情况。(在多表查询时，需要消除无效的笛卡尔积)

![image-20220201162441949](img\image-20220201162441949.png)

### 内连接 

**内连接查询的是两张表交集的部分。**

- 隐式内连接

```sql
select 字段列表 
from table1,table2 
where 条件...;
```

- 显示外连接

```sql
select 字段列表 
from table1 [INNER] JOIN table2
ON 连接条件...;
```

### 外连接 

- 左外连接

```sql
select 字段列表 
from table1 LEFT [INNER] JOIN table2
ON 连接条件...;
```

<span style="color:red">**相当于查询表1(左表)的所有数据 包含 表1和表2交集部分的数据**</span>

- 右外连接

```sql
select 字段列表 
from table1 RIGHT [INNER] JOIN table2
ON 连接条件...;
```

<span style="color:red">**相当于查询表2(右表)的所有数据 包含 表1和表2交集部分的数据**</span>

### 自连接

```sql
select 字段列表
from tableA 别名A
JOIN tableA 别名B
ON 条件...;
```

<span style="color:red">**自连接查询，可以是内连接查询，也可以是外连接查询。**</span>

### 联合查询-union,union all

对于union查询，就是把多次查询的结果合并起来，形成一个新的查询结果集。

```sql
select 字段列表 from tableA...;
UNION [ALL]
select 字段列表 from tableB...;
```

对于联合查询的多张表的列数必须保持一致，字段类型也需要保持一致。 
union all 会将全部的数据直接合并在一起；union 会对合并之后的数据去重。

### 子查询 

概念：SQL语句中嵌套SELECT语句，称为嵌套查询，又称子查询。

```sql
select * from t1 
where column1 = (select column1 from t2);
```

子查询外部的语句可以是INSERT / UPDATE / DELETE / SELECT 的任何一个。

根据子查询结果不同，分为： 

- 标量子查询（子查询结果为单个值）
- 列子查询(子查询结果为一列)
- 行子查询(子查询结果为一行)
- 表子查询(子查询结果为多行多列)

根据子查询位置，分为：WHERE之后 、FROM之后、SELECT 之后。

#### 标量子查询

子查询返回的结果是单个值（数字、字符串、日期等），最简单的形式，这种子查询成为标量子查询。 
常用的操作符：= <> > >= < <=

#### 列子查询

子查询返回的结果是一列（可以是多行），这种子查询称为列子查询。 
常用的操作符：IN 、NOT IN 、 ANY 、SOME 、 ALL

| 操作符 | 描述                                 |
| ------ | ------------------------------------ |
| IN     | 在指定的集合范围之内，多选一         |
| NOT IN | 在指定的集合范围之内，多选一         |
| ANY    | 子查询返回列表中，有任意一个满足即可 |
| SOME   | 与ANY等同，使用 的地方都可以使用ANY  |
| ALL    | 子查询返回列表的所有值都必须满足     |

#### 行子查询

子查询返回的结果是一行（可以是多列），这种子查询称为行子查询。 
常用的操作符：= 、<> 、IN 、NOT IN

#### 表子查询

子查询返回的结果是多行多列，这种子查询称为表子查询。 
常用的操作符：IN

### 多表查询案例

1 .查询员工的姓名、年龄、职位、部门信息。 
2 .查询年龄小于30岁的员工姓名、年龄、职位、部门信息。
3 .查询拥有员工的部门ID、部门名称。
4 .查询所有年龄大于40岁的员工, 及其归属的部门名称; 如果员工没有分配部门, 也需要展示出来。
5 .查询所有员工的工资等级。
6 .查询 "研发部" 所有员工的信息及工资等级。
7 .查询 "研发部" 员工的平均工资。
8 .查询工资比 "灭绝" 高的员工信息。
9 .查询比平均薪资高的员工信息。
10.查询低于本部门平均工资的员工信息。
11.查询所有的部门信息, 并统计部门的员工人数。
12.查询所有学生的选课情况, 展示出学生名称, 学号, 课程名称

```sql
-- 准备数据
create database mydb1;
use mydb1;
create table dept(
    id   int auto_increment comment 'ID' primary key,
    name varchar(50) not null comment '部门名称'
)comment '部门表';

create table emp(
    id  int auto_increment comment 'ID' primary key,
    name varchar(50) not null comment '姓名',
    age  int comment '年龄',
    job varchar(20) comment '职位',
    salary int comment '薪资',
    entrydate date comment '入职时间',
    managerid int comment '直属领导ID',
    dept_id int comment '部门ID'
)comment '员工表';

create table salgrade(
    grade int,
    losal int,
    hisal int
) comment '薪资等级表';
```

```sql
-- 插入数据
insert into salgrade values (1,0,3000);
insert into salgrade values (2,3001,5000);
insert into salgrade values (3,5001,8000);
insert into salgrade values (4,8001,10000);
insert into salgrade values (5,10001,15000);
insert into salgrade values (6,15001,20000);
insert into salgrade values (7,20001,25000);
insert into salgrade values (8,25001,30000);

INSERT INTO dept (id, name) VALUES (1, '研发部'), 
(2, '市场部'),
(3, '财务部'), 
(4, '销售部'),
(5, '总经办'), 
(6, '人事部');
INSERT INTO emp (id, name, age, job,salary, entrydate, managerid, dept_id) 
VALUES(1, '金庸', 66, '总裁',20000, '2000-01-01', null,5),
(2, '张无忌', 20, '项目经理',12500, '2005-12-05', 1,1),
(3, '杨逍', 33, '开发', 8400,'2000-11-03', 2,1),
(4, '韦一笑', 48, '开发',11000, '2002-02-05', 2,1),
(5, '常遇春', 43, '开发',10500, '2004-09-07', 3,1),
(6, '小昭', 19, '程序员鼓励师',6600, '2004-10-12', 2,1),

(7, '灭绝', 60, '财务总监',8500, '2002-09-12', 1,3),
 (8, '周芷若', 19, '会计',48000, '2006-06-02', 7,3),
(9, '丁敏君', 23, '出纳',5250, '2009-05-13', 7,3),

(10, '赵敏', 20, '市场部总监',12500, '2004-10-12', 1,2),
(11, '鹿杖客', 56, '职员',3750, '2006-10-03', 10,2),
(12, '鹤笔翁', 19, '职员',3750, '2007-05-09', 10,2),
(13, '方东白', 19, '职员',5500, '2009-02-12', 10,2),

(14, '张三丰', 88, '销售总监',14000, '2004-10-12', 1,4),
(15, '俞莲舟', 38, '销售',4600, '2004-10-12', 14,4),
(16, '宋远桥', 40, '销售',4600, '2004-10-12', 14,4),
(17, '陈友谅', 42, null,2000, '2011-10-12', 1,null);
```

```sql
-- 1 .查询员工的姓名、年龄、职位、部门信息。（隐式内连接）
-- 表: emp , dept
-- 连接条件: emp.dept_id = dept.id
-- 这种会漏掉没有部门信息的人
select e.name, e.age, e.job, d.name
from emp e,
     dept d
where e.dept_id = d.id;
-- 这种不会
select e.name, e.age, e.job, d.name
from emp e left join dept d on d.id = e.dept_id;


-- 2 .查询年龄小于30岁的员工姓名、年龄、职位、部门信息。
-- 表: emp , dept
-- 连接条件: emp.dept_id = dept.id
select e.name, e.age, e.job, d.name
from emp e
         left join dept d on d.id = e.dept_id where e.age<30;

-- 3 .查询拥有员工的部门ID、部门名称。
-- 表: emp , dept
-- 连接条件: emp.dept_id = dept.id
select distinct d.id , d.name from emp e , dept d where e.dept_id = d.id;
select d.* from dept d where d.id in (select emp.dept_id from emp group by emp.dept_id);


-- 4 .查询所有年龄大于40岁的员工, 及其归属的部门名称; 如果员工没有分配部门, 也需要展示出来。
-- 表: emp , dept
-- 连接条件: emp.dept_id = dept.id
-- 外连接
select e.*,d.name from emp e left join dept d on d.id = e.dept_id where e.age>40;


-- 5 .查询所有员工的工资等级。
-- 表: emp , salgrade
-- 连接条件 : emp.salary >= salgrade.losal and emp.salary <= salgrade.hisal
select e.name,e.salary,s.grade from emp e,salgrade s where e.salary between s.losal and s.hisal;


-- 6 .查询 "研发部" 所有员工的信息及工资等级。
-- 表: emp , salgrade , dept
-- 连接条件 : emp.salary between salgrade.losal and salgrade.hisal , emp.dept_id = dept.id
-- 查询条件 : dept.name = '研发部'
select e.name,d.name,e.salary,s.grade from  emp e,salgrade s,dept d where e.dept_id=1 and e.dept_id=d.id and (e.salary between s.losal and s.hisal);


-- 7. 查询 "研发部" 员工的平均工资
-- 表: emp , dept
-- 连接条件 :  emp.dept_id = dept.id
select avg(e.salary) from emp e, dept d where e.dept_id = d.id and d.name = '研发部';


-- 8. 查询工资比 "灭绝" 高的员工信息。
-- a. 查询 "灭绝" 的薪资
select salary from emp where name = '灭绝';

-- b. 查询比她工资高的员工数据
select * from emp where salary > ( select salary from emp where name = '灭绝' );


-- 9. 查询比平均薪资高的员工信息
-- a. 查询员工的平均薪资
select avg(salary) from emp;

-- b. 查询比平均薪资高的员工信息
select * from emp where salary > ( select avg(salary) from emp );


-- 10. 查询低于本部门平均工资的员工信息

-- a. 查询指定部门平均薪资  1
select avg(e1.salary) from emp e1 where e1.dept_id = 1;
select avg(e1.salary) from emp e1 where e1.dept_id = 2;

-- b. 查询低于本部门平均工资的员工信息
select * from emp e2 where e2.salary < ( select avg(e1.salary) from emp e1 where e1.dept_id = e2.dept_id );


-- 11. 查询所有的部门信息, 并统计部门的员工人数
select d.id, d.name , ( select count(*) from emp e where e.dept_id = d.id ) '人数' from dept d;

select count(*) from emp where dept_id = 1;
```



## 事务

- 简介：事务是一组操作的集合，这组操作，要么全部执行成功，要么全部执行失败。
- 操作：
    - `start transaction; -- 开启事务`
    - `commit / rollback;-- 提交/回滚事务`
- 四大特性：原子性、一致性、隔离性、持久性
- 并发事务问题：脏读、不可重复读、幻读
- 隔离级别：READ INCOMMITED、READ COMMITED、REPEATABLE READ、SERIALIZABLE。

### 事务简介

<span style="color:red">**事务**</span> 是一组操作的集合，它是一个不可分割的工作单位，事务会把所有的操作作为一个整体一起向系统提交或撤销操作 请求，即这些操作<span style="color:red">**要么同时成功，要么同时失败。**</span>

![image-20220201163223983](img\image-20220201163223983.png)

<span style="color:red">**默认MySQL的事务是自动提交的，也就是说，当执行一条DML语句，MySQL会立即隐式的提交事务。**</span>

### 事务操作

查看/设置事务提交方式

```sql
select @@autocommit; -- 1 表示事务默认自动提交。
+--------------+
| @@autocommit |
+--------------+
|            1 |
+--------------+
1 row in set (0.00 sec)

set @@autocommit = 0;
```

提交事务

```sql
commit
```

回滚事务

```mysql
ROLLBAK
```

开启事务

```sql
start TRANSACTION 或 BEGIN
```

案例

```sql
create table account(
	id int auto_increment primary key,
    name varchar(10),
    momeny int
);
insert into account(id,name,momeny) value(null,'张三',2000),(null,'李四',2000);

-- 转账操作 张三给李四转账1000
-- 1.查询张三的用户

-- 2. 张三用户-1000
-- 程序抛出异常，事务回滚。
-- 3. 李四用户+1000
```



### 事务四大特性(ACID)

- 原子性（Atomicity）：事务是不可分割的最小操作单元，要么全部成功，要么全部失败。 
- 一致性（Consistency）：事务完成时，必须使所有的数据都保持一致状态。 
- 隔离性（Isolation）：数据库系统提供的隔离机制，保证事务在不受外部并发操作影响的独立环境下运行。 
- 持久性（Durability）：事务一旦提交或回滚，它对数据库中的数据的改变就是永久的。（一般，事务提交后，会把数据持久化到磁盘里，这个持久化时机是可以设置的）

### 并发事务问题

| 问题       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| 脏读       | 一个事务读到另一个事务还没提交的数据                         |
| 不可重复读 | 事务 A 多次读取同一数据，事务B在事务A多次读取的过程中，对数据作了**更新并提交**，**导致事务A多次读取同一数据时，结果不一致**<br />（一个事务先后读取同一条记录，但两次读取的数据不同，称之为不可重复读。） |
| 幻读       | 一个事务先根据某些条件查询出一些记录，之后另一个事务又向表中插入了符合这些条件的记录，原先的事务再次按照该条件查询时，能把另一个事务插入的记录也读出来。（幻读在读未提交、读已提交、可重复读隔离级别都可能会出现） |

![image-20220201164120794](img\image-20220201164120794.png)

![image-20220201164153913](img\image-20220201164153913.png)

![image-20220201164216281](img\image-20220201164216281.png)

**总结**：不可重复读的和幻读很容易混淆，**不可重复读侧重于修改，幻读侧重于新增或删除**

### 事务隔离级别

- 读未提交 read uncommitted
- 读已提交 read committed
- 可以重复读 repeatable read
- 串行化 serializable

[彻底搞懂 MySQL 事务的隔离级别-阿里云开发者社区 (aliyun.com)](https://developer.aliyun.com/article/743691)

| 5隔离级别                  | 脏读 | 不可重复读 | 幻读 |
| -------------------------- | ---- | ---------- | ---- |
| Read uncommitted           | √    | √          | √    |
| Read committed(Oracle默认) | ×    | √          | √    |
| Repeatable Read(MySQL默认) | ×    | ×          | √    |
| Serializable               | ×    | ×          | ×    |

```sql
-- 查看事务隔离级别
select @@TRANSACTION_ISOLATION;

-- 设置事务隔离级别
SET [SESSION|GLOBAL] TRANSATION ISOLATION LEVEL [READ UNCOMMITED | READ COMMITED | SERIALIZABLE]
```

注意：事务隔离级别越高，数据越安全，但是性能越低。

### 演示

```mysql
-- ---------------------------- 事务操作 ----------------------------
-- 数据准备
create table account(
    id int auto_increment primary key comment '主键ID',
    name varchar(10) comment '姓名',
    money int comment '余额'
) comment '账户表';
insert into account(id, name, money) VALUES (null,'张三',2000),(null,'李四',2000);


-- 恢复数据
update account set money = 2000 where name = '张三' or name = '李四';


select @@autocommit;

set @@autocommit = 0; -- 设置为手动提交

-- 转账操作 (张三给李四转账1000)
-- 1. 查询张三账户余额
select * from account where name = '张三';

-- 2. 将张三账户余额-1000
update account set money = money - 1000 where name = '张三';

程序执行报错 ...

-- 3. 将李四账户余额+1000
update account set money = money + 1000 where name = '李四';


-- 提交事务
commit;

-- 回滚事务
rollback ;


-- 方式二
-- 转账操作 (张三给李四转账1000)
start transaction ;

-- 1. 查询张三账户余额
select * from account where name = '张三';

-- 2. 将张三账户余额-1000
update account set money = money - 1000 where name = '张三';

程序执行报错 ...

-- 3. 将李四账户余额+1000
update account set money = money + 1000 where name = '李四';


-- 提交事务
commit;

-- 回滚事务
rollback;


-- 查看事务隔离级别
select @@transaction_isolation;

-- 设置事务隔离级别
set session transaction isolation level read uncommitted ;

set session transaction isolation level repeatable read ;
```



# 进阶

## MySQL8.0.26-Linux版安装

WSL-Ubuntu 安装 [怎样在 Ubuntu Linux 上安装 MySQL - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/64080934)

### 1. 准备一台Linux服务器

云服务器或者虚拟机都可以; 

Linux的版本为 CentOS7;

### 2. 下载Linux版MySQL安装包

https://downloads.mysql.com/archives/community/

![image-20211031230239760](img/image-20211031230239760.png) 

### 3. 上传MySQL安装包

![image-20211031231930205](img/image-20211031231930205.png) 

### 4. 创建目录,并解压

```
mkdir mysql

tar -xvf mysql-8.0.26-1.el7.x86_64.rpm-bundle.tar -C mysql
```

### 5. 安装mysql的安装包

```
cd mysql

rpm -ivh mysql-community-common-8.0.26-1.el7.x86_64.rpm 

rpm -ivh mysql-community-client-plugins-8.0.26-1.el7.x86_64.rpm 

rpm -ivh mysql-community-libs-8.0.26-1.el7.x86_64.rpm 

rpm -ivh mysql-community-libs-compat-8.0.26-1.el7.x86_64.rpm

yum install openssl-devel

rpm -ivh  mysql-community-devel-8.0.26-1.el7.x86_64.rpm

rpm -ivh mysql-community-client-8.0.26-1.el7.x86_64.rpm

rpm -ivh  mysql-community-server-8.0.26-1.el7.x86_64.rpm

```

### 6. 启动MySQL服务

```
systemctl start mysqld
```

```
systemctl restart mysqld
```

```
systemctl stop mysqld
```

### 7. 查询自动生成的root用户密码

```
grep 'temporary password' /var/log/mysqld.log
```

命令行执行指令 :

```
mysql -u root -p
```

然后输入上述查询到的自动生成的密码, 完成登录 .

### 8. 修改root用户密码

登录到MySQL之后，需要将自动生成的不便记忆的密码修改了，修改成自己熟悉的便于记忆的密码。

```
ALTER  USER  'root'@'localhost'  IDENTIFIED BY '1234';
```

执行上述的SQL会报错，原因是因为设置的密码太简单，密码复杂度不够。我们可以设置密码的复杂度为简单类型，密码长度为4。

```
set global validate_password.policy = 0;
set global validate_password.length = 4;
```

降低密码的校验规则之后，再次执行上述修改密码的指令。

### 9. 创建用户

默认的root用户只能当前节点localhost访问，是无法远程访问的，我们还需要创建一个root账户，用户远程访问

```
create user 'root'@'%' IDENTIFIED WITH mysql_native_password BY '1234';
```

### 10. 并给root用户分配权限

```
grant all on *.* to 'root'@'%';
```

### 11. 重新连接MySQL

```
mysql -u root -p
```

然后输入密码

### 12. 通过DataGrip远程连接MySQL

## 存储引擎

- MySQL体系结构：连接层、服务层、引擎层、存储层
- 存储引擎介绍
- 存储引擎的特点：INNODB与MyISAM相比，INNODB支持事务，外键和行级锁。
- 存储引擎选择
    - INNODB：存储业务系统中对于事务、数据完整性要求较高的核心数据。
    - MyISAM：存储业务系统的非核心事务。


### MySQL体系结构

![image-20220122173547178](img\image-20220122173547178.png)

**连接层**：最上层是一些客户端和链接服务，主要完成一些类似于连接处理、授权认证、及相关的安全方案。服务器也会为安全接入的每个客户 端验证它所具有的操作权限。

**服务层**：第二层架构主要完成大多数的核心服务功能，如SQL接口，并完成缓存的查询，SQL的分析和优化，部分内置函数的执行。所有跨存 储引擎的功能也在这一层实现，如 过程、函数等。 

**引擎层**：存储引擎真正的负责了MySQL中数据的存储和提取，服务器通过API和存储引擎进行通信。不同的存储引擎具有不同的功能，这样我 们可以根据自己的需要，来选取合适的存储引擎。 

**存储层**：主要是将数据存储在文件系统之上，并完成与存储引擎的交互。

### 存储引擎简介

存储引擎是MySQL的核心；存储引擎就是存储数据、建立索引、更新/查询数据等技术的实现方式 。**存储引擎是基于表的，而不是基于库的**，所以存储引擎也可被称为表类型。

- 在创建表时，指定存储引擎

```mysql
create table tb_user(
	id int,
    name varchar(80)
) ENGINE = INNODB;
```

- 查看当前数据库支持的存储引擎

```mysql
show ENGINES;
```

### 存储引擎特点

> **InnoDB**

- 介绍：InnoDB是一种兼顾高可靠性和高性能的通用存储引擎，在 MySQL 5.5 之后，InnoDB是默认的 MySQL 存储引擎。
- 特点：
    - DML操作遵循ACID模型，支持事务
    - 行级锁，提高并发访问性能
    - 支持 FOREIGN KEY约束，保证数据的完整性和正确性；
- 文件：xxx.ibd：xxx代表的是表名，innoDB引擎的每张表都会对应这样一个表空间文件，存储该表的表结构（frm、sdi）、数据和索引。 参数：innodb_file_per_table

![image-20220122175230096](img\image-20220122175230096.png)

> MyISAM

介绍：MyISAM是MySQL早期的默认存储引擎。 

特点：不支持事务，不支持外键，支持表锁，不支持行锁，访问速度快 

文件：xxx.sdi中存储表结构信息；xxx.MYD中存储数据；xxx.MYI中存储索引

> Memory

介绍：Memory引擎的表数据时存储在内存中的，由于受到硬件问题、或断电问题的影响，只能将这些表作为临时表或缓存使用。

特点：内存存放；默认使用hash索引

文件 xxx.sdi中存储表结构信息

![image-20220122175842514](img\image-20220122175842514.png)

InnoDB 与 MyISAM 直接的区别：InnoDB支持事务、锁机制为行级锁、支持外键。

### 存储引擎选择

在选择存储引擎时，应该根据应用系统的特点选择合适的存储引擎。对于复杂的应用系统，还可以根据实际情况选择多种存储引擎进行组合。 

➢ InnoDB: 是Mysql的默认存储引擎，支持事务、外键。如果应用对事务的完整性有比较高的要求，在并发条件下要求数据的一致 性，数据操作除了插入和查询之外，还包含很多的更新、删除操作，那么InnoDB存储引擎是比较合适的选择。 

➢ MyISAM ： 如果应用是以读操作和插入操作为主，只有很少的更新和删除操作，并且对事务的完整性、并发性要求不是很高，那么选择这个存储引擎是非常合适的。 【一般会选MongoDB】

➢ MEMORY：将所有数据保存在内存中，访问速度快，通常用于临时表及缓存。MEMORY的缺陷就是对表的大小有限制，太大的表无法缓存在内存中，而且无法保障数据的安全性。 【一般会选Redis】

### 总结

- [x] 体系结构
- 连接层、服务层、引擎层、存储层
- [x] 存储引擎简介
- [x] 存储引擎特点
- INNODB 与 MyISAM：事务、外键、行级锁
- [x] 存储引擎应用
- INNODB 存储业务系统中对于事务、数据完整性要求较高的核心数据
- MyISAM 存储业务系统的非核心事务。

## 索引

- 索引概述：高效获取数据的数据结构

- 索引结构：B+Tree、Hash

- 索引分类 ：主键索引、唯一索引、常规索引、全文索引、聚集索引、二级索引。

- 索引语法 

    ```sql
    create [unique] index xxx on xxx(xxx);
    show index from xxx;
    drop index xxx on xxxx;
    ```

- SQL性能分析：执行频次、慢查询日志、profile、explain

- 索引使用：联合索引、索引失效、SQL提示、覆盖索引、前缀索引、单列/联合索引。

- 索引设计原则：表、字段、索引。

### 概述

**索引（index）是帮助MySQL 高效获取数据的数据结构（有序）**。在数据之外，数据库系统还维护着满足特定查找算法的数据结构，**这些数据结构以某种方式引用（指向）数据**， 这样就**可以在这些数据结构上实现高级查找算法**，这种数据结构就是索引。

索引一般采用 B+Tree / Hash 这种数据结构。

| 优点                                                         | 缺点                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 提高数据检索的效率，降低数据库的IO成本                       | 索引列也是要占用空间的。                                     |
| 通过索引列对数据进行排序，降低数据排序的成本，降低CPU的 消耗。 | 索引大大提高了查询效率，同时却也降低更新表的速度，如对表进 行INSERT、UPDATE、DELETE时，效率降低。 |

### 索引结构

| 索引       | InnoDB          | MyISAM | Memory |
| ---------- | --------------- | ------ | ------ |
| B+tree索引 | 支持            | 支持   | 支持   |
| Hash索引   | 不支持          | 不支持 | 支持   |
| R-tree索引 | 不支持          | 支持   | 不支持 |
| Full-text  | 5.6版本之后支持 | 支持   | 不支持 |

<span style="color:red">**我们平常所说的索引，如果没有特别指明，都是指B+树结构组织的索引。**</span>

> 剖析各种平衡树结构做索引的优缺点：

![image-20220202102720134](img\image-20220202102720134.png)

二叉树缺点：顺序插入时，会形成一个链表，查询性能大大降低。 大数据量情况下，层级较深，检索速度慢。 

红黑树：大数据量情况下，层级较深，检索速度慢。

- B-Tree（多路平衡查找树）：以一颗最大度数（max-degree）为5(5阶)的b-tree为例(每个节点最多存储4个key，5个指针)：

![image-20220202102904364](img\image-20220202102904364.png)

具体动态变化的过程可以参考网站: https://www.cs.usfca.edu/~galles/visualization/BTree.html

- B+Tree：以一颗最大度数（max-degree）为4（4阶）的b+tree为例：

![image-20220202103024493](img\image-20220202103024493.png)

B-Tree 和 B+Tree 的区别：

- B+Tree 所有的数据都会出现在叶子节点
- 叶子节点形成一个单向链表

MySQL索引数据结构对经典的B+Tree进行了优化。在原B+Tree的基础上，增加一个指向相邻叶子节点的链表指针，就形成了带有顺 序指针的B+Tree，提高区间访问的性能。
![image-20220202103738086](img\image-20220202103738086.png)

- 哈希索引

哈希索引，哈希索引就是采用一定的hash算法，将键值换算成新的hash值，映射到对应的槽位上，然后存储在hash表中。 如果两个(或多个)键值，映射到一个相同的槽位上，他们就产生了hash冲突（也称为hash碰撞），可以通过链表来解决。
![image-20220202103833246](img\image-20220202103833246.png)

哈希索引特点

- Hash索引只能用于对等比较(=，in)，不支持范围查询（between，>，< ，...）
- 无法利用索引完成排序操作
- 查询效率高，通常只需要一次检索就可以了，效率通常要高于B+tree索引

存储引擎支持

在MySQL中，支持hash索引的是Memory引擎，而InnoDB中具有自适应hash功能，hash索引是存储引擎根据B+Tree索引在指定条件下自动构建的。

### 问题辨析

为什么InnoDB存储引擎选择使用B+tree索引结构？

➢ 相对于二叉树，层级更少，搜索效率高； 

➢ 对于B-tree，无论是叶子节点还是非叶子节点，都会保存数据，这样导致一 页中存储的键值减少，指针跟着减少，要同样保存大量数据，只能增加树的 高度，导致性能降低； 

➢ 相对Hash索引，B+tree支持范围匹配及排序操作；

### 索引分类

| 分类     | 含义                                                 | 特点                     | 关键字   |
| -------- | ---------------------------------------------------- | ------------------------ | -------- |
| 主键索引 | 针对于表中主键创建的索引                             | 默认自动创建, 只能有一个 | PRIMARY  |
| 唯一索引 | 避免同一个表中某数据列中的值重复                     | 可以有多个               | UNIQUE   |
| 常规索引 | 快速定位特定数据                                     | 可以有多个               |          |
| 全文索引 | 全文索引查找的是文本中的关键词，而不是比较索引中的值 | 可以有多个               | FULLTEXT |

在InnoDB存储引擎中，根据索引的存储形式，又可以分为以下两种：

| 分类                      | 含义                                                       | 特点                |
| ------------------------- | ---------------------------------------------------------- | ------------------- |
| 聚集索引(Clustered Index) | 将数据存储与索引放到了一块，索引结构的叶子节点保存了行数据 | 必须有,而且只有一个 |
| 二级索引(Secondary Index) | 将数据与索引分开存储，索引结构的叶子节点关联的是对应的主键 | 可以存在多个        |

聚集索引选取规则: 

➢ 如果存在主键，主键索引就是聚集索引。 

➢ 如果不存在主键，将使用第一个唯一（UNIQUE）索引作为聚集索引。 

➢ 如果表没有主键，或没有合适的唯一索引，则InnoDB会自动生成一个rowid作为隐藏的聚集索引。

![image-20220202104706091](img\image-20220202104706091.png)

二级索引可能会存在一个回表查询。二级索引中存储的是二级索引和主键（此处是name和id），如果需要查询的数据不止name和id，则会触发一次回表查询，根据主键id，再查询一次数据，拿到需要的字段值。

<span style="color:red">**思考题**</span>

InnoDB主键索引的B+tree高度为多高？
假设：一行数据大小为1k，一页中可以存储16行这样的数据。InnoDB的指针用6个字节的空间，主键即使为bigint，占用字节数为8。

一页的大小：16*1k=16k。一页可以存多少个索引项和指针？
5个索引项需要6个指针。故计算方式如下：

高度为2：
$n*8+(n+1)*6=16*1024$，n大约是 1170。
$1171*16=18736$
高度为3:
$1171*1171*16=21939856$

### 索引语法

- 创建、查看、删除索引

```sql
-- 创建索引
create [unique][fulltext] index index_name on table_name (index_col_name,...);
-- 查看索引
show index from table_name;
-- 删除索引
drop index index_name on table_name;
```

- 按照如下需求，完成索引的创建
    - name字段为姓名字段，该字段的值可能会重复，为该字段创建索引。
    - phone手机号字段的值，是非空，且唯一的，为该字段创建唯一索引。
    - 为profession、age、status创建联合索引。
    - 为email建立合适的索引来提升查询效率。

```sql
-- 表结构
create table tb_user(
	id int primary key auto_increment comment '主键',
	name varchar(50) not null comment '用户名',
	phone varchar(11) not null comment '手机号',
	email varchar(100) comment '邮箱',
	profession varchar(11) comment '专业',
	age tinyint unsigned comment '年龄',
	gender char(1) comment '性别 , 1: 男, 2: 女',
	status char(1) comment '状态',
	createtime datetime comment '创建时间'
) comment '系统用户表';


INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('吕布', '17799990000', 'lvbu666@163.com', '软件工程', 23, '1', '6', '2001-02-02 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('曹操', '17799990001', 'caocao666@qq.com', '通讯工程', 33, '1', '0', '2001-03-05 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('赵云', '17799990002', '17799990@139.com', '英语', 34, '1', '2', '2002-03-02 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('孙悟空', '17799990003', '17799990@sina.com', '工程造价', 54, '1', '0', '2001-07-02 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('花木兰', '17799990004', '19980729@sina.com', '软件工程', 23, '2', '1', '2001-04-22 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('大乔', '17799990005', 'daqiao666@sina.com', '舞蹈', 22, '2', '0', '2001-02-07 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('露娜', '17799990006', 'luna_love@sina.com', '应用数学', 24, '2', '0', '2001-02-08 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('程咬金', '17799990007', 'chengyaojin@163.com', '化工', 38, '1', '5', '2001-05-23 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('项羽', '17799990008', 'xiaoyu666@qq.com', '金属材料', 43, '1', '0', '2001-09-18 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('白起', '17799990009', 'baiqi666@sina.com', '机械工程及其自动化', 27, '1', '2', '2001-08-16 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('韩信', '17799990010', 'hanxin520@163.com', '无机非金属材料工程', 27, '1', '0', '2001-06-12 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('荆轲', '17799990011', 'jingke123@163.com', '会计', 29, '1', '0', '2001-05-11 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('兰陵王', '17799990012', 'lanlinwang666@126.com', '工程造价', 44, '1', '1', '2001-04-09 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('狂铁', '17799990013', 'kuangtie@sina.com', '应用数学', 43, '1', '2', '2001-04-10 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('貂蝉', '17799990014', '84958948374@qq.com', '软件工程', 40, '2', '3', '2001-02-12 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('妲己', '17799990015', '2783238293@qq.com', '软件工程', 31, '2', '0', '2001-01-30 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('芈月', '17799990016', 'xiaomin2001@sina.com', '工业经济', 35, '2', '0', '2000-05-03 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('嬴政', '17799990017', '8839434342@qq.com', '化工', 38, '1', '1', '2001-08-08 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('狄仁杰', '17799990018', 'jujiamlm8166@163.com', '国际贸易', 30, '1', '0', '2007-03-12 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('安琪拉', '17799990019', 'jdodm1h@126.com', '城市规划', 51, '2', '0', '2001-08-15 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('典韦', '17799990020', 'ycaunanjian@163.com', '城市规划', 52, '1', '2', '2000-04-12 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('廉颇', '17799990021', 'lianpo321@126.com', '土木工程', 19, '1', '3', '2002-07-18 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('后羿', '17799990022', 'altycj2000@139.com', '城市园林', 20, '1', '0', '2002-03-10 00:00:00');
INSERT INTO tb_user (name, phone, email, profession, age, gender, status, createtime) VALUES ('姜子牙', '17799990023', '37483844@qq.com', '工程造价', 29, '1', '4', '2003-05-26 00:00:00');
```

创建索引

```sql
create index idx_user_name on tb_user(name);

create unique index idx_user_phone on tb_user(phone);

create index idx_p_a_s on tb_user(profession,age,status);

create index idx_email on tb_user(email);
```

### SQL性能分析



## SQL优化

## 视图/存储过程/触发器

## 锁

## InnoDB引擎

MySQL事务的隔离级别默认是可重复读。

MySQL的事务是通过

redo log 日志保证持久性。

update\delete 等操作先回到buffer pool里，然后把变化写入到redolog buffer，redolog buffer再写入redolog里。

为什么不直接buffer pool将脏页刷新到磁盘里？因为update、delete这些的数据极大可能是随机写入磁盘的，效率低，效率低的话，出现问题中断它写入的可能性就大。而redolog里的则是顺序写入，一条一条写进去写入速度很快，出现问题中断它的几率小很多。
如果buffer pool写入.ibd 文件时出错了，可以通过redolog里的数据进行恢复。

## MySQL管理

