# 基本术语

- 数据库：保存有组织的数据的容器。

- 数据库软件（DBMS）：数据库管理系统。

- 表：某种特定类型数据的结构化清单。

- 模式：关于数据库和表的布局及特性的信息。

- 列（column）：表中的一个字段。

- 数据类型（datatype）：所容许的数据的类型。

- 行（row）：表中的一个记录。【行和记录可以互相替代，但从术语上来说，行才是正确的。】

- 主键（primary key）：唯一区分表中的每行，

  规则（MySQL）

  - 任意两行都不具有相同的主键值。
  - 每行都必须具有一个主键值。

- SQL（Structured Query Language）：结构化查询语言

----

# `MySQL简介`

## `MySQL版本`

- 4---InnoDB引擎，增加事务处理、改进全文本搜索。
- 4.1---对函数库，子查询，集成帮助等的重要增加
- 5----存储过程、触发器、游标、视图

----

# 使用`MySQL`

## 连接

> 开启`MySQL`服务

`net start mysql` windows下

- net start 服务名称

`service mysqld start` Linux下

- `service mysqld start` 5.0版本是`mysqld`
- `service mysql start`  5.57版本是`mysql`

> 关闭`MySQL`服务

`net stop mysql` windows下

`service mysql stop` Linux下，版本问题同开启一致

> 连接本地数据库

`mysql -uroot -proot` 

- -u用户名【用户名为root】
- -p密码【用户名为root】

> 连接服务器端的数据库

`mysql -h 192.168.1.1 -P 3306 -uroot -proot`

- -h 主机地址
- -P 端口地址
- -u 用户名
- -p 密码

## 选择数据库

> 查看数据库

`show databases`

> 选择数据库

`use databaseName`

> 查看表

`show tables`

> 显示表的列

`show columns from tableName`

`desc tableName` 也可

> show的作用

查看表结构

- `show columns from tableName`
- `MySQL 5.0提供了 desc tableName作为快捷方式`
- `show status`, 显示服务器状态信息
- `show create database databaseName` 查看创建数据库的`MySQL`语句
- `show create table tableName`, 查看数据库表的`MySQL`语句
- `show grants for username` 显示授予用户的安全权限
- `show errors / show warnings` 显示服务器错误或警告消息
- 查看使用帮助：在`mysql`模式下输入 `help show`

> `mysql`清屏

- `system clear` Linux系统下
- `system cls` `mysql8.0`+ windows下

# 检索数据

## Select基本用法

`sql`不区分大小写。习惯上关键字大写，列名小写。

> 查询单列

`select prod_name from products`

> 查询多列

`select prod_id, prod_name, prod_price from products;` 列与列用逗号分割。

> 查询所有

`select * from products`   尽量别使用通配符查询所有字段，会拉低查询效率。

>查询不同的行（去重）

`select distinct vend_id from products`  使用关键字 distinct

> 限制结果 【限制查询的结果数目】

`select prod_name from products limit 5`  返回不多于5行

`select prod_name from products limit 5,5`  从行5开始，查5条数据。 会查到 5 6 7 8 9。

----

# 排序检索数据

使用order by子句。

> 排序数据

`select prod_name from products order by prod_name`  按照prod_name排序

- 默认是升序
- order by也可以用不显示的那些列作为排序条件。如`select prod_name from products order by prod_id`

> 按多个排序

`select prod_id, prod_price, prod_name from products order by prod_price, prod_name`  按prod_price和prod_name排序。先按价格，价格一样再按prod_name

> 指定排序方向

`select prod_id, prod_price, prod_name from products order by prod_price desc`    

- `desc=descend`    降序
- `asc=ascend`     升序

- `desc`只作用于最前面那列
  - `eg`： `select prod_id,prod_price,prod_name from products order by prod_price desc,prod_name`
  - prod_price是降序
  - prod_name是升序

> 查询价格最高的【order + limit】

`select * from products order by prod_price desc limit 1`

----

# 过滤数据

使用where子句指定搜索条件，where子句在from子句之后给出。

在同时使用order by 和 where子句时，应该让order by位于where之后。

> 基本用法

`select prod_name,prod_price from products where prod_price = 2.5`

> `SQL`过滤与应用过滤

`SQL`过滤，数据库进行了一定的优化。要传给客户机应用（或开发语言）处理的数据更少。

应用过滤，将所有的数据发送给客户机应用（或开发语言），传递数据的开销大，且要处理的数据更多。

> where子句操作符

| 操作符  | 说明               |
| ------- | ------------------ |
| =       | 等于               |
| <>      | 不等于             |
| !=      | 不等于             |
| <       | 小于               |
| <=      | 小于等于           |
| >       | 大于               |
| >=      | 大于等于           |
| between | 在指定的两个值之间 |

> 范围值检查 [between示例]

`select prod_name, prod_price from products where prod_price between 5 and 10`

> 空值检查

`select prod_name from products where prod_price is null`  查询价格为空的数据

# 数据过滤

组合where子句、not和in操作符

> 使用and多条件过滤

`select prod_id, prod_price, prod_name from products where vend_id=1003 and prod_price<=10;`  

>使用or进行任意匹配

`select prod_name, prod_price from products where vend_id = 1002 or vend_id = 1003;`

> 条件计算次序

or操作符优先于and操作符。需要使用圆括号明确地分组相应的操作符。

` select prod_name, prod_price ` 

`from products`
 `where (vend_id = 1002 or vend_id = 1003) and prod_price >=10;`

> in操作符

in指定条件范围

`select prod_name, prod_price from products where vend_id in (1002,1003) order by prod_name;`

- in语法清晰直观【对比or】
- in的计算次序更容易管理
- in操作符比一般or操作符执行更快
- in可以包含其他的select子句，使得可以动态地建立where子句。

> not操作符

否定条件

`select * from products where vend_id not in(1002,1003);`

在复杂`sql`中很有用。

`MySQL中的NOT 支持对IN BETWEEN和EXISTS子句取反`

----

# 用通配符进行条件过滤

使用Like和通配符进行通配搜索。

- 通配符：用来匹配值的一部分的特殊字符。
- 搜索模式：由字面值、通配符或两者组合构成的搜索条件。

> **百分号（%）通配符**

%可以匹配0个、1个或多个字符。<span style="color:red">%不可匹配NULL！</span>

jet开头的

`select * from products where prod_name like 'jet%';`

s开头，e结尾的

`select * from products where prod_name like 's%e';`

> **下划线（ _ ）通配符**

只匹配当个字符

`select prod_id, prod_name from products where prod_name like '_ ton anvil';`

> **通配符的使用技巧**

通配符搜索处理一般比前面的其他搜索花的时间更长。

- 不要过度使用通配符
- 确实需要时才使用。除非必要，否在不要把它用在搜索模式的开始处。
- 仔细注意通配符的位置。若放错地方，可能不会返回想要的数据。

----

# 使用正则表达式

用来匹配文本的特殊的串（字符集合），如从文本中提取电话号码。

`MySQL`的where子句对正则表达式提供了初步的支持。

> **匹配包含文本1000的所有行。**

`select prod_name from products where prod_name REGEXP '1000' order by prod_name;`

> **匹配一个任意字符**

`select prod_name from products where prod_name REGEXP '.000' order by prod_name`

得到结果

- JetPack 1000
- JetPack 2000

匹配到任意一个字符开头且后缀为000的数据。

> **Like与Regexp的区别**

- Like匹配整列。
  - 如 like ‘1000’   得整列的值都是1000才行。

- Regexp在列值内进行匹配。
  - 如 Regexp '1000' 列值内含有1000就行。

> **Regexp匹配整列**

使用^和$（anchor）

> **Regexp中的or匹配**

`select * from products where prod_name REGEXP '1000|2000' order by prod_name;`

> **Regexp中匹配几个字符之一**

用了 | 试了下，发现没成功

`select * from xx where prod_name REGEXP '1|2 Ton'` 不行，失败了

这个SQL的意思是 你要筛选 1 或 2 Ton 所以筛选结果不对。

这个可以

`select * from products where prod_name REGEXP [123] ton `  筛选出含有

- 1 ton 或 2 ton 或 3 ton 的数据

> **排除xx字符**·

`select * from products where prod_name REGEXP '[^123] ton';`

排除了含有 1 ton  ，2 ton ，3 ton的数据。

> **范围匹配**

`select * from products where prod_name REGEXP '[1-5]';`

匹配含有1-5的数据

也可[a-z]

> **特殊字符的匹配**

`select * from products where prod_name REGEXP '\\.';`

```sql
\\. 表示匹配 .
```

> **匹配字符类**

字符类列表

| 类        | 说明                                |
| --------- | ----------------------------------- |
| [:alnum:] | 任意字母和数字(同[a-zA-Z0-9])       |
| [:alpha:] | 任意字符 (同[a-zA-Z])               |
| [:blank:] | 空格和制表符(`同 [\\t]` )           |
| [:cntrl:] | ASCII控制字符  (ASCII 0 到31 和127) |
|           |                                     |
|           |                                     |
|           |                                     |
|           |                                     |
|           |                                     |

![image-20200913031529100](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20200913031529100.png)

> **匹配多个实例**

重复元字符表

| 元字符    | 说明                       |
| --------- | -------------------------- |
| *         | 0或多个                    |
| +         | 1或多个                    |
| ？        | 0或1个                     |
| {n}       | 指定数目的匹配             |
| {n , }    | 不少于指定数目的匹配       |
| { n , m } | 匹配数目的范围  m不超过255 |

举例子

- `select prod_name from products where prod_name REGEXP \\([0-9] sticks?\\)`

```sql
\\( 匹配(
sticks?  中的s? 表示为s可出现一次或0次
```

- `select * from products where prod_name REGEXP '[0-9]{4}' ` 匹配包含连续出现四次的数据

> **定位符**

| 元字符      | 说明       |
| ----------- | ---------- |
| ^           | 文本的开始 |
| $           | 文本的结尾 |
| [ [ :<: ] ] | 词的开始   |
| [ [:>:] ]   | 词的结尾   |

词的开始 词的结尾不会用。

> **注意**

高版本`MySQL`不区分大小写（3.23.4以后的版本）。

要区分大小写的话用<span style="color:red">BINARY</span>关键字

----

# 计算字段

对字段进行操作，如拼接 大小写转换 格式化等等。

> **拼接字段**

```mysq
select concat(vend_name, '(' , vend_contry , ')') from vendors;
```

其他DBMS用的可能是 + ||

> **删除空格**

- RTrim(字段)
- LTrim(字段)
- Trim(字段)

> **算术运算**

直接对字段【可计算的字段】 + - * /即可

# 数据处理函数

不同DBMS的函数不一样，使用时记得写上注释。

> **支持的函数类型**

- 用于处理文本串（如删除或填充值，转换值为大写或小写）的文本函数。
- 用于在数值数据上进行算术操作（如返回绝对值，进行代数运算）的数值函数。
-  用于处理日期和时间值并从这些值中提取特定成分（例如，返回两个日期之差，检查日期有效性等）的日期和时间函数。
-  返回DBMS正使用的特殊信息（如返回用户登录信息，检查版本细节）的系统函数

> 