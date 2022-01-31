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

### DCL



## 函数

## 约束

## 多表查询

## 事务

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

- MySQL体系结构
- 存储引擎介绍
- 存储引擎的特点
- 存储引擎选择

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

特点：不支持事务，不支持外键 支持表锁，不支持行锁，访问速度快 

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

- 索引概述
- 索引结构
- 索引分类 
- 索引语法 
- SQL性能分析
- 索引使用
- 索引设计原则

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

