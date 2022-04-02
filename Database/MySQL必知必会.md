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

# `MySQL简介`

## `MySQL版本`

- 4---InnoDB引擎，增加事务处理、改进全文本搜索。
- 4.1---对函数库，子查询，集成帮助等的重要增加
- 5----存储过程、触发器、游标、视图

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

> 查询单列 `select prod_name from products`

> 查询多列 `select prod_id, prod_name, prod_price from products;` 列与列用逗号分割。

> 查询所有 `select * from products`   尽量别使用通配符查询所有字段，会拉低查询效率。

>查询不同的行（去重）`select distinct vend_id from products`  使用关键字 distinct

> 限制结果 【限制查询的结果数目】
>
> - `select prod_name from products limit 5`  返回不多于5行
> - `select prod_name from products limit 5,5`  从行5开始，查5条数据。 会查到 5 6 7 8 9。

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

# 过滤数据

使用where子句指定搜索条件，where子句在from子句之后给出。

在同时使用order by 和 where子句时，应该让order by位于where之后。

> 基本用法

`select prod_name,prod_price from products where prod_price = 2.5`

> `SQL`过滤与应用过滤

`SQL`过滤，数据库进行了一定的优化。要传给客户机应用（或开发语言）处理的数据更少。

应用过滤，将所有的数据发送给客户机应用（或开发语言），传递数据的开销大，且要处理的数据更多。

> where子句操作符

| 操作符  |        说明        |
| :-----: | :----------------: |
|    =    |        等于        |
|   <>    |       不等于       |
|   !=    |       不等于       |
|    <    |        小于        |
|   <=    |      小于等于      |
|    >    |        大于        |
|   >=    |      大于等于      |
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

## 文本处理函数

长度、大小写转换、去空串、字符串截取，发音相近。

> **发音相近**

```sql
select * from customers where soundex(cust_contact) = soundex('Y Lie');
```

## 日期和时间处理函数

这块的内容比较重要，要好好学学，用的挺频繁。

|     函数      |                             说明                             |
| :-----------: | :----------------------------------------------------------: |
|   AddDate()   | 增加一个日期（天 、周）AddDate(字段, INTERVAL 1 WEEK/YEAR/DAY) |
|   AddTime()   |                增加一个时间（时、分）类似上面                |
|   CurDate()   |                    返回当前日期【年月日】                    |
|   CurTime()   |     返回当前时间【时分秒】select CurTime(); 查询当前时间     |
|    Date()     |      返回日期时间的日期部分 Date(xxx) xxx字段的日期部分      |
|  DateDiff()   |                       计算两个日期之差                       |
|  Date_Add()   | 高度灵活的日期运算函数<a href="https://www.w3school.com.cn/sql/func_date_add.asp">具体用法</a> |
| Date_Format() | 返回格式化的日期或串<a href="https://www.w3school.com.cn/sql/func_date_format.asp">具体用法</a> |
|  DayOfWeek()  |             返回日期对应的星期几 DayOfWeek(日期)             |
|    Time()     |                   返回时间部分。 时分秒。                    |

利用MySQL进行时间部分的匹配时，只匹配需要的那一部分字段。比如只要年月日就只比年月日。

反例：`WHERE order_date = '2005-09-01’` 可能含有 `00：00：00`

# 数据汇总/聚集函数

<span style="color:red">聚集函数，用的很频繁</span>

## 典型场景

- 确定表中行数。【统计null吗？】
- 获得表中行组的和
- 找出表列的Max Min Avg

## SQL聚集函数

| 函数  |                       说明                       |
| :---: | :----------------------------------------------: |
|  avg  |                  忽略为null的行                  |
| count | count(*) 空，非空都统计；count(column)不统计null |
|  max  | 忽略为null的行，可用于数值，非数值。如最大日期。 |
|  min  |                    说明同max                     |
|  sum  |                  忽略为null的行                  |

如果不允许计算重复的值，则可以指定distinct参数

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

需要把数据分为多个逻辑组，对每个逻辑组进行聚集计算。

## 创建分组

根据vend_id 分组，统计每组的`num_prods`数目。group by后面

```mysql
 select vend_id,count(*) as num_prods 
 from products 
 group by vend_id;
 
 # 下面这条语句 没有显示 vend_id
 select count(*) as num_prods 
 from products 
 group by vend_id;
```

## group by！

- GROUP BY子句可以包含任意数目的列。这使得能对分组进行嵌套，为数据分组提供更细致的控制。
- 如果在GROUP BY子句中嵌套了分组，数据将在最后规定的分组上进行汇总。换句话说，在建立分组时，指定的所有列都一起计算（所以不能从个别的列取回数据）。
- GROUP BY子句中列出的每个列都必须是检索列或有效的表达式（<span style="color:red">但不能是聚集函数</span>）。如果在SELECT中使用表达式，则必须在GROUP BY子句中指定相同的表达式。不能使用别名。
-  除聚集计算语句外，SELECT语句中的每个列都必须在GROUP BY子句中给出。【前面试了，发现不必给出~】
-  如果分组列中具有NULL值，则NULL将作为一个分组返回。如果列中有多行NULL值，它们将分为一组。
- GROUP BY子句必须出现在WHERE子句之后，ORDER BY子句之前。

## 过滤分组

包括那些分组，排除那些分组。

### having与where

无法使用where。where过滤指定的行而非分组。where没有分组的概念。我们需要使用having！

having在分组中的用法与where类似。

**另一种解释**

<span style="color:orange">WHERE在数据分组前进行过滤，HAVING在数据分组后进行过滤。</span>这是一个重要的区别，WHERE排除的行不包括在分组中。这可能会改变计算值，从而影响HAVING子句中基于这些值过滤掉的分组。

```mysql
# 筛选出 数目大于2的
select cust_id,count(*) as orders 
from orders 
group by cust_id 
having count(*)>=2;
```

### having与where一起使用

使它返回过去12个月内具有两个以上订单的顾客。

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

超过三个表不推荐使用join，不清晰。推荐用where。【阿里巴巴开发手册】

联结是一种机制，用来在一条SELECT语句中关联表。【用Where不香吗，清晰，效率高】

where写法，不加条件的话会产生<span style="color:red">笛卡儿积</span>。

- where是等值联结

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

使用UNION，UNION ALL。看王姗的书。

- 在单个查询中从不同的表返回类似结构的数据；
- 对单个表执行多个查询，按单个查询返回数据；
- 组合查询和where到达那个效率高未知，需要我们进行测试；

## 使用UNION

UNION会取消重复行！！！UNION ALL，不取消重复行。

> **案例**

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

> **使用规则**

- UNION必须由两条或两条以上的SELECT语句组成，语句之间用关键字UNION分隔（因此，如果组合4条SELECT语句，将要使用3个UNION关键字）。
-  UNION中的每个查询必须包含相同的列、表达式或聚集函数（不过各个列不需要以相同的次序列出）。
- 列数据类型必须兼容：类型不必完全相同，但必须是DBMS可以隐含地转换的类型（例如，不同的数值类型或不同的日期类型）。

## 对组合结果进行排序

末尾加个order by即可

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

- `MyISAM`支持全文本搜索。

- `InnoDB`不支持全文本搜索。







