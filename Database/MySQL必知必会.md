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

# MySQL简介

## MySQL版本

- 4---InnoDB 引擎，增加事务处理、改进全文本搜索。
- 4.1---对函数库，子查询，集成帮助等的重要增加
- 5----存储过程、触发器、游标、视图

# 使用MySQL

## 连接

> 开启 `MySQL` 服务

`net start mysql` windows 下

- net start 服务名称

`service mysqld start` Linux 下

- `service mysqld start` 5.0版本是 `mysqld`
- `service mysql start`  5.57版本是 `mysql`

> 关闭 `MySQL` 服务

`net stop mysql` windows 下

`service mysql stop` Linux 下，版本问题同开启一致

> 连接本地数据库

`mysql -uroot -proot` 

- -u用户名【用户名为 root】
- -p密码【用户名为 root】

> 连接服务器端的数据库

`mysql -h 192.168.1.1 -P 3306 -uroot -proot`

- -h 主机地址
- -P 端口地址
- -u 用户名
- -p 密码

## 选择数据库

> 查看数据库：`show databases`
>
> 选择数据库：`use databaseName`
>
> 查看表：`show tables`
>
> 显示表的列：`show columns from tableName` 或 `desc tableName`

> show的作用

查看表结构

- `show columns from tableName`
- `MySQL 5.0 提供了 desc tableName 作为快捷方式`
- `show status`, 显示服务器状态信息
- `show create database databaseName` 查看创建数据库的 `MySQL` 语句
- `show create table tableName`, 查看数据库表的 `MySQL` 语句
- `show grants for username` 显示授予用户的安全权限
- `show errors / show warnings` 显示服务器错误或警告消息
- 查看使用帮助：在 `mysql` 模式下输入 `help show`

> `mysql` 清屏

- `system clear` Linux 系统下
- `system cls` `mysql8.0`+ windows 下

# 检索数据

## Select基本用法

`sql` 不区分大小写。习惯上关键字大写，列名小写。

- 查询单列 `select prod_name from products`
- 查询多列 `select prod_id, prod_name, prod_price from products;` 列与列用逗号分割。
- 查询所有 `select * from products`  尽量别使用通配符查询所有字段，会拉低查询效率。
- 查询不同的行（去重）`select distinct vend_id from products`  使用关键字 distinct
- 限制结果 【限制查询的结果数目】
    - `select prod_name from products limit 5`  返回不多于 5 行
    - `select prod_name from products limit 5,5`  从行 5 开始，查 5 条数据。 会查到 5 6 7 8 9。

# 排序检索数据

使用 order by 子句。

> 排序数据

`select prod_name from products order by prod_name`  按照 prod_name 排序

- 默认是升序
- order by 也可以用不显示的那些列作为排序条件。如 `select prod_name from products order by prod_id`

> 按多个排序

`select prod_id, prod_price, prod_name from products order by prod_price, prod_name`  按 prod_price 和 prod_name 排序。先按价格，价格一样再按 prod_name

> 指定排序方向

`select prod_id, prod_price, prod_name from products order by prod_price desc`    

- `desc=descend` 降序
- `asc=ascend` 升序

- `desc` 只作用于最前面那列
  - eg：`select prod_id,prod_price,prod_name from products order by prod_price desc,prod_name`
  - prod_price 是降序
  - prod_name 是升序

> 查询价格最高的【order + limit】

`select * from products order by prod_price desc limit 1`

# 过滤数据

使用 where 子句指定搜索条件，where 子句在 from 子句之后给出。

在同时使用 order by 和 where 子句时，应该让 order by 位于 where 之后。

> 基本用法

`select prod_name,prod_price from products where prod_price=2.5`

> `SQL` 过滤与应用过滤

`SQL` 过滤，数据库进行了一定的优化。要传给客户机应用（或开发语言）处理的数据更少。

应用过滤，将所有的数据发送给客户机应用（或开发语言），传递数据的开销大，且要处理的数据更多。

> where 子句操作符

| 操作符  | 说明               |
| :------ | :----------------- |
| =       | 等于               |
| <>      | 不等于             |
| !=      | 不等于             |
| <       | 小于               |
| <=      | 小于等于           |
| >       | 大于               |
| >=      | 大于等于           |
| between | 在指定的两个值之间 |

> 范围值检查 [between 示例]

`select prod_name, prod_price from products where prod_price between 5 and 10`

> 空值检查

`select prod_name from products where prod_price is null`  查询价格为空的数据

# 数据过滤

组合 where 子句、not 和 in 操作符

> 使用 and 多条件过滤

`select prod_id, prod_price, prod_name from products where vend_id=1003 and prod_price<=10;`  

>使用 or 进行任意匹配

`select prod_name, prod_price from products where vend_id = 1002 or vend_id = 1003;`

> 条件计算次序

or 操作符优先于 and 操作符。需要使用圆括号明确地分组相应的操作符。

```sql
select prod_name, prod_price
from products
where (vend_id = 1002 or vend_id = 1003) and prod_price >=10;
```

> in 操作符

in 指定条件范围

`select prod_name, prod_price from products where vend_id in (1002,1003) order by prod_name;`

- in 语法清晰直观【对比 or】
- in 的计算次序更容易管理
- in 操作符比一般 or 操作符执行更快
- in 可以包含其他的 select 子句，使得可以动态地建立 where 子句。

> not 操作符

否定条件

`select * from products where vend_id not in(1002,1003);`

在复杂 `sql` 中很有用。

MySQL 中的 NOT 支持对 IN BETWEEN 和 EXISTS 子句取反

# 用通配符进行条件过滤

使用 Like 和通配符进行通配搜索。

- 通配符：用来匹配值的一部分的特殊字符。
- 搜索模式：由字面值、通配符或两者组合构成的搜索条件。

> <b>百分号（%）通配符</b>

%可以匹配0个、1个或多个字符。<span style="color:red">%不可匹配NULL！</span>

jet 开头的

`select * from products where prod_name like 'jet%';`

s 开头，e 结尾的

`select * from products where prod_name like 's%e';`

> <b>下划线（ _ ）通配符</b>

只匹配单个字符

`select prod_id, prod_name from products where prod_name like '_ ton anvil';`

> <b>通配符的使用技巧</b>

通配符搜索处理一般比前面的其他搜索花的时间更长。

- 不要过度使用通配符
- 确实需要时才使用。除非必要，否在不要把它用在搜索模式的开始处。
- 仔细注意通配符的位置。若放错地方，可能不会返回想要的数据。

# 使用正则表达式

用来匹配文本的特殊的串（字符集合），如从文本中提取电话号码。

`MySQL`的where子句对正则表达式提供了初步的支持。

> <b>匹配包含文本 1000 的所有行。</b>

`select prod_name from products where prod_name REGEXP '1000' order by prod_name;`

> <b>匹配一个任意字符</b>

`select prod_name from products where prod_name REGEXP '.000' order by prod_name`

得到结果

- JetPack 1000
- JetPack 2000

匹配到任意一个字符开头且后缀为 000 的数据。

> <b>Like 与 Regexp 的区别</b>

- Like 匹配整列。
  - 如 like ‘1000’   得整列的值都是 1000 才行。

- Regexp 在列值内进行匹配。
  - 如 Regexp '1000' 列值内含有 1000 就行。

> <b>Regexp 匹配整列</b>

使用 ^ 和 $（anchor）

> <b>Regexp 中的 or 匹配</b>

`select * from products where prod_name REGEXP '1000|2000' order by prod_name;`

> <b>Regexp 中匹配几个字符之一</b>

用了 | 试了下，发现没成功

`select * from xx where prod_name REGEXP '1|2 Ton'` 不行，失败了

这个 SQL 的意思是 你要筛选 1 或 2 Ton 所以筛选结果不对。

这个可以

`select * from products where prod_name REGEXP [123] ton `  筛选出含有

- 1 ton 或 2 ton 或 3 ton 的数据

> <b>排除 xx 字符</b>

`select * from products where prod_name REGEXP '[^123] ton';`

排除了含有 1 ton  ，2 ton ，3 ton的数据。

> <b>范围匹配</b>

`select * from products where prod_name REGEXP '[1-5]';`

匹配含有 1-5 的数据

也可 [a-z]

> <b>特殊字符的匹配</b>

`select * from products where prod_name REGEXP '\\.';`

```sql
\\. 表示匹配 .
```

> <b>匹配字符类</b>

字符类列表

| 类        | 说明                                  |
| --------- | ------------------------------------- |
| [:alnum:] | 任意字母和数字(同 [a-zA-Z0-9])        |
| [:alpha:] | 任意字符 (同 [a-zA-Z])                |
| [:blank:] | 空格和制表符(`同 [\\t]` )             |
| [:cntrl:] | ASCII 控制字符  (ASCII 0 到 31 和127) |

> <b>匹配多个实例</b>

重复元字符表

| 元字符    | 说明                         |
| --------- | ---------------------------- |
| *         | 0 或多个                     |
| +         | 1 或多个                     |
| ？        | 0 或 1 个                    |
| {n}       | 指定数目的匹配               |
| {n , }    | 不少于指定数目的匹配         |
| { n , m } | 匹配数目的范围  m 不超过 255 |

举例子

- `select prod_name from products where prod_name REGEXP \\([0-9] sticks?\\)`

```sql
\\( 匹配(
sticks?  中的s? 表示为s可出现一次或0次
```

- `select * from products where prod_name REGEXP '[0-9]{4}' ` 匹配包含连续出现四次的数据

> <b>定位符</b>

| 元字符      | 说明       |
| ----------- | ---------- |
| ^           | 文本的开始 |
| $           | 文本的结尾 |
| [ [ :<: ] ] | 词的开始   |
| [ [:>:] ]   | 词的结尾   |

词的开始 词的结尾不会用。

> <b>注意</b>

高版本 `MySQL` 不区分大小写（3.23.4 以后的版本）。

要区分大小写的话用 <span style="color:red">BINARY</span> 关键字

# 计算字段

对字段进行操作，如拼接、大小写转换、格式化等等。

> <b>拼接字段</b>

```mysq
select concat(vend_name, '(' , vend_contry , ')') from vendors;
```

其他 DBMS 用的可能是 + ||

> <b>删除空格</b>

- RTrim (字段)
- LTrim (字段)
- Trim (字段)

> <b>算术运算</b>

直接对字段【可计算的字段】 + - * /即可

# 数据处理函数

不同 DBMS 的函数不一样，使用时记得写上注释。

> <b>支持的函数类型</b>

- 用于处理文本串（如删除或填充值，转换值为大写或小写）的文本函数。
- 用于在数值数据上进行算术操作（如返回绝对值，进行代数运算）的数值函数。
-  用于处理日期和时间值并从这些值中提取特定成分（例如，返回两个日期之差，检查日期有效性等）的日期和时间函数。
-  返回 DBMS 正使用的特殊信息（如返回用户登录信息，检查版本细节）的系统函数

## 文本处理函数

长度、大小写转换、去空串、字符串截取，发音相近。

> <b>发音相近</b>

```sql
select * from customers where soundex(cust_contact) = soundex('Y Lie');
```

## 日期和时间处理函数

这块的内容比较重要，要好好学学，用的挺频繁。

| 函数          | 说明                                                         |
| :------------ | :----------------------------------------------------------- |
| AddDate()     | 增加一个日期（天 、周）AddDate (字段, INTERVAL 1 WEEK/YEAR/DAY) |
| AddTime()     | 增加一个时间（时、分）类似上面                               |
| CurDate()     | 返回当前日期【年月日】                                       |
| CurTime()     | 返回当前时间【时分秒】select CurTime(); 查询当前时间         |
| Date()        | 返回日期时间的日期部分 Date(xxx) xxx 字段的日期部分          |
| DateDiff()    | 计算两个日期之差                                             |
| Date_Add()    | 高度灵活的日期运算函数<a href="https://www.w3school.com.cn/sql/func_date_add.asp">具体用法</a> |
| Date_Format() | 返回格式化的日期或串<a href="https://www.w3school.com.cn/sql/func_date_format.asp">具体用法</a> |
| DayOfWeek()   | 返回日期对应的星期几 DayOfWeek(日期)                         |
| Time()        | 返回时间部分。 时分秒。                                      |

利用 MySQL 进行时间部分的匹配时，只匹配需要的那一部分字段。比如只要年月日就只比年月日。

反例：`WHERE order_date = '2005-09-01’` 可能含有 `00：00：00`

# 数据汇总/聚集函数

<span style="color:red">聚集函数，用的很频繁</span>

## 典型场景

- 确定表中行数。【统计 null 吗？】
- 获得表中行组的和
- 找出表列的 Max Min Avg

## SQL聚集函数

| 函数  | 说明                                               |
| :---- | :------------------------------------------------- |
| avg   | 忽略为 null 的行                                   |
| count | count(*) 空，非空都统计；count(column) 不统计 null |
| max   | 忽略为 null 的行，可用于数值，非数值。如最大日期。 |
| min   | 说明同 max                                         |
| sum   | 忽略为 null 的行                                   |

如果不允许计算重复的值，则可以指定 distinct 参数

```sql
# 17.多
select avg(distinct prod_price) as avg_price from products;
# 16.多
select avg(prod_price) as avg_price from products;
# 对于count 只能用于 count() 不能count(*)
# 个人看法 count(distinct *)逻辑上也说不过去~~ 一般都有primary，不会同。
select count(distinct  prod_price) from products;
```

同时使用多个聚集函数。

```sql
 select count(*) as num_items,
    -> max(prod_price) as max_price,
    -> min(prod_price) as min_price,
    -> avg(prod_price) as avg_price
    -> from products;
```

# 分组数据

group by & having。【弱项】

## 典型场景

需要把数据分为多个逻辑组，对每个逻辑组进行聚集计算。比如，我们查询 `pet_shop` 表中，按动物的品种作为分组，查询每个品种中最高的价格。

```mysql
SELECT species, MAX(price) AS price
FROM pet_shop
GROUP BY species;
```

```shell
+---------+-------+
| species | price |
+---------+-------+
| cat     |   200 |
| dog     |   600 |
| rabbit  |    50 |
+---------+-------+
```

## 创建分组

根据 vend_id 分组，统计每组的 `num_prods` 数目。group by 后面

```mysql
 select vend_id,count(*) as num_prods 
 from products 
 group by vend_id;
 
 # 下面这条语句 没有显示 vend_id
 select count(*) as num_prods 
 from products 
 group by vend_id;
```

## group by

- GROUP BY 子句可以包含任意数目的列。这使得能对分组进行嵌套，为数据分组提供更细致的控制。
- 如果在 GROUP BY 子句中嵌套了分组，数据将在最后规定的分组上进行汇总。换句话说，在建立分组时，指定的所有列都一起计算（所以不能从个别的列取回数据）。
- GROUP BY 子句中列出的每个列都必须是检索列或有效的表达式（<span style="color:red">但不能是聚集函数</span>）。如果在 SELECT 中使用表达式，则必须在 GROUP BY 子句中指定相同的表达式。不能使用别名。
-  除聚集计算语句外，SELECT 语句中的每个列都必须在 GROUP BY 子句中给出。【前面试了，发现不必给出~】
-  如果分组列中具有 NULL 值，则 NULL 将作为一个分组返回。如果列中有多行 NULL 值，它们将分为一组。
- GROUP BY 子句必须出现在 WHERE 子句之后，ORDER BY 子句之前。

## 过滤分组

包括那些分组，排除那些分组。

### having与where

无法使用 where。where 过滤指定的行而非分组。where 没有分组的概念。我们需要使用 having！

having 在分组中的用法与 where 类似。

<b>另一种解释</b>

<span style="color:orange">WHERE 在数据分组前进行过滤，HAVING 在数据分组后进行过滤。</span>这是一个重要的区别，WHERE 排除的行不包括在分组中。这可能会改变计算值，从而影响 HAVING 子句中基于这些值过滤掉的分组。

```mysql
# 筛选出 数目大于2的
select cust_id,count(*) as orders 
from orders 
group by cust_id 
having count(*)>=2;
```

### having与where一起使用

使它返回过去 12 个月内具有两个以上订单的顾客。

```mysql
select vend_id, count(*) as num_prods 
from products
where prod_price >= 10
group by vend_id 
having count(*) >=2 ;

# where 先过滤了，having在把where过滤后的数据 再进行过滤。
```

### select子句顺序

```mysql
select
from
where   行级别过滤
group by  分组说明
having  组级别过滤
order by  排序 【默认升序 asc[上升] desc[下降]】
limit
```

# 用户变量

为了便于使用，我们可以定义一个特定的变量来保存该值，这样的变量就是 MySQL 的用户变量了。在 `SELECT` 语句中设置临时变量，需要使用 `@` 开头，然后需要使用 `:=` 来进行赋值。

比如，我们把在 `pet_shop` 表中的最大价格赋值给用户变量 `max_price`，把最低价格赋值给用户变量 `min_price`。

```sql
select @max_price := MAX(price), @min_price := MIN(price) 
from pet_shop;
```

最高的价格和最低的价格都已经保存到用户变量中了，当我们需要时，直接查询这两个用户变量即可。例如，我们查询一下最高价格的动物和最低价格的动物信息。

```mysql
select @max_price; # 查询最高价格

# 查询 pet_shop 中的最高价格和最低价格
select * from pet_shop
where price in(@max_price, @min_price);
```

# 过滤重复数据

## BIT_COUNT 和 BIT_OR

- `BIT_OR` 是用来对两个二进制进行或运算。
- `BIT_COUNT` 是用来计算二进制中包含 1 的个数。

`BIT_OR` 是进行按位或运算，就是只有当两个数全为 0 时，结果才为 0，用它可以获得一个二进制的数值，但是返回给我们的结果会把该二进制转换成十进制。

```mysql
select bit_or(1<<4)
select 1<<4 # 16

select bit_count(3) # 2，3 的二进制是 11，1 的个数是 2
```

BIT_COUNT 和 BIT_OR 会自动把十进制转换为二进制。

# 子查询

嵌套在其他查询中的查询。

## 利用子查询进行过滤

select 套 select

我感觉没啥好记得，就是嵌套查询嘛。

给一个例子即可。

```mysql
select cust_id 
from orders 
where order_num in (select order_num 
                    from orderitems 
                    where prod_id = 'TNT2');
```

嵌套子查询效率较低，不建议大量使用。

## 作为计算字段使用子查询

经典案例，供参考。效率较差，不是很推荐。

```mysql
select cust_name,cust_state,
	(select count(*) 
     from orders 
     where orders.cust_id = customers.cust_id) as orders from customers 
order by cust_name;
```

# 联结表

这块还是看王姗的数据库系统概论。

## 联结（join）

超过三个表不推荐使用 join，不清晰。推荐用 where。【阿里巴巴开发手册】

联结是一种机制，用来在一条 SELECT 语句中关联表。【用 where 不香吗，清晰，效率高】

where 写法，不加条件的话会产生<span style="color:red">笛卡儿积</span>。

- where 是等值联结

```mysql
select vend_name, prod_name, prod_price 
from vendors, products 
where vendors.vend_id = products.vend_id 
order by vend_name, prod_name;
```

- 联结的表越多，越耗性能。

# 高级联结

表别名，解决二义性。

## 自联结

有时候自联结比子查询快！

```mysql
select p1.prod_id, p1.prod_name 
from products as p1, products as p2 
where p1.vend_id = p2.vend_id
and p2.prod_id = 'DTNTR';
```

用自联结而不用子查询 自联结通常作为外部语句用来替代从相同表中检索数据时使用的子查询语句。虽然最终的结果是相同的，但有时候处理联结远比处理子查询快得多。

## 自然联结

## 外部联结

两张表之间的关联。

```mysql
# 内连接 可以不写inner关键字
select customers.cust_id, orders.order_num 
from customers 
inner join orders on customers.cust_id = orders.cust_id;

# 左外连接，
# LEFT JOIN 关键字会从左表 (table_name1) 那里返回所有的行，即使在右表 (table_name2) 中没有匹配的行
# 可以不写outer关键字
select customers.cust_id, orders.order_num 
from customers 
left outer join orders on customers.cust_id = orders.cust_id;

```

# 组合查询

## 推荐&效率

使用 UNION，UNION ALL。看王姗的书。

- 在单个查询中从不同的表返回类似结构的数据；
- 对单个表执行多个查询，按单个查询返回数据；
- 组合查询和 where 到达那个效率高未知，需要我们进行测试；

## 使用UNION

UNION 会取消重复行！！！UNION ALL，不取消重复行。

> <b>案例</b>

把两个查询结果集并起来了

```mysql
select * from products 
where prod_price<=5
	union
select * from products 
where vend_id in(1001,1002)
# 相当于
select * from products 
where prod_price<=5 or vend_id in(1001,1002)
```

> <b>使用规则</b>

- UNION 必须由两条或两条以上的 SELECT 语句组成，语句之间用关键字 UNION 分隔（因此，如果组合 4 条 SELECT 语句，将要使用 3 个 UNION 关键字）。
-  UNION 中的每个查询必须包含相同的列、表达式或聚集函数（不过各个列不需要以相同的次序列出）。
- 列数据类型必须兼容：类型不必完全相同，但必须是 DBMS 可以隐含地转换的类型（例如，不同的数值类型或不同的日期类型）。

## 对组合结果进行排序

末尾加个 order by 即可

```mysql
select * from products 
where prod_price<=5
	union
select * from products 
where vend_id in(1001,1002)
order by vend_id
```

# 全文本搜索

并非所有的搜索引擎都支持全文本搜索。

- `MyISAM` 支持全文本搜索。

- `InnoDB` 不支持全文本搜索。

# 索引

索引是一种与表有关的结构，它的作用相当于书的目录，可以根据目录中的页码快速找到所需的内容。

当表中有大量记录时，若要对表进行查询，没有索引的情况是全表搜索：将所有记录一一取出，和查询条件进行对比，然后返回满足条件的记录。这样做会执行大量磁盘 I/O 操作，并花费大量数据库系统时间。

而如果在表中已建立索引，在索引中找到符合查询条件的索引值，通过索引值就可以快速找到表中的数据，可以<b>大大加快查询速度</b>。

对一张表中的某个列建立索引，有以下两种语句格式：

```sql
ALTER TABLE 表名字 ADD INDEX 索引名 (列名);

CREATE INDEX 索引名 ON 表名字 (列名);
```

我们用这两种语句分别建立索引：

```sql
ALTER TABLE employee ADD INDEX idx_id (id);  #在employee表的id列上建立名为idx_id的索引

CREATE INDEX idx_name ON employee (name);   #在employee表的name列上建立名为idx_name的索引
```

索引的效果是加快查询速度，当表中数据不够多的时候是感受不出它的效果的。可以使用命令 <b>SHOW INDEX FROM 表名字;</b> 查看刚才新建的索引。

在使用 SELECT 语句查询的时候，语句中 WHERE 里面的条件，会<b>自动判断有没有可用的索引</b>。

比如有一个用户表，它拥有用户名(username)和个人签名(note)两个字段。其中用户名具有唯一性，并且格式具有较强的限制，我们给用户名加上一个唯一索引；个性签名格式多变，而且允许不同用户使用重复的签名，不加任何索引。

这时候，如果你要查找某一用户，使用语句 `select * from user where username=?` 和 `select * from user where note=?` 性能是有很大差距的，对<b>建立了索引的用户名</b>进行条件查询会比<b>没有索引的个性签名</b>条件查询快几倍，在数据量大的时候，这个差距只会更大。

一些字段不适合创建索引，比如性别，这个字段存在大量的重复记录无法享受索引带来的速度加成，甚至会拖累数据库，导致数据冗余和额外的 CPU 开销。

# 视图

视图是从一个或多个表中导出来的表，是一种<b>虚拟存在的表</b>。它就像一个窗口，通过这个窗口可以看到系统专门提供的数据，这样，用户可以不用看到整个数据库中的数据，而只关心对自己有用的数据。

注意理解视图是虚拟的表：

- 数据库中只存放了视图的定义，而没有存放视图中的数据，这些数据存放在原来的表中；
- 使用视图查询数据时，数据库系统会从原来的表中取出对应的数据；
- 视图中的数据依赖于原来表中的数据，一旦表中数据发生改变，显示在视图中的数据也会发生改变；
- 在使用视图的时候，可以把它当作一张表。

创建视图的语句格式为：

```sql
CREATE VIEW 视图名(列a,列b,列c) AS SELECT 列1,列2,列3 FROM 表名字;
```

可见创建视图的语句，后半句是一个 SELECT 查询语句，所以<b>视图也可以建立在多张表上</b>，只需在 SELECT 语句中使用<b>子查询</b>或<b>连接查询</b>。

创建一个简单的视图，名为 <b>v_emp</b>，包含 <b>v_name，v_age，v_phone</b> 三个列：

```sql
CREATE VIEW v_emp (v_name,v_age,v_phone) AS SELECT name,age,phone FROM employee;
```







