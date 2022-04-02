# Windows 上安装 MySQL

安装版：[MySQL安装指南_技术交流_牛客网 (nowcoder.com)](https://www.nowcoder.com/discuss/825171?type=all&order=recall&pos=&page=1&ncTraceId=&channel=-1&source_id=search_all_nctrack&gio_id=649BB42AA30733488E2E468CA0F7721F-1642732433430)

Windows 上安装 MySQL 相对来说会较为简单，最新版本可以在 [MySQL 下载](http://dev.mysql.com/downloads/mysql) 中下载中查看。

![图片说明](https://uploadfiles.nowcoder.com/images/20191011/334190970_1570776156000_6466136D1077A4900F42C47E312D4188)

点击 **Download** 按钮进入下载页面，点击下图中的 **No thanks, just start my download.** 就可立即下载：

![图片说明](https://uploadfiles.nowcoder.com/images/20191011/334190970_1570776164052_48B5E330CE1ABB26B5756AD0ADE1DD24)

下载完后，我们将 zip 包解压到相应的目录，这里我将解压后的文件夹放在 **C:\web\mysql-8.0.11** 下。

**接下来我们需要配置下 MySQL 的配置文件**

打开刚刚解压的文件夹 **C:\web\mysql-8.0.11** ，在该文件夹下创建 **my.ini** 配置文件，编辑 **my.ini** 配置以下基本信息：

```
[client]
# 设置mysql客户端默认字符集
default-character-set=utf8

[mysqld]
# 设置3306端口
port = 3306
# 设置mysql的安装目录
basedir=C:\\web\\mysql-8.0.11
# 设置 mysql数据库的数据的存放目录，MySQL 8+ 不需要以下配置，系统自己生成即可，否则有可能报错
# datadir=C:\\web\\sqldata
# 允许最大连接数
max_connections=20
# 服务端使用的字符集默认为8比特编码的latin1字符集
character-set-server=utf8
# 创建新表时将使用的默认存储引擎
default-storage-engine=INNODB
```

**接下来我们来启动下 MySQL 数据库：**

以管理员身份打开 cmd 命令行工具，切换目录：

```
cd C:\web\mysql-8.0.11\bin
```

初始化数据库：

```
mysqld --initialize --console
```

执行完成后，会输出 root 用户的初始默认密码，如：

```
...
2018-04-20T02:35:05.464644Z 5 [Note] [MY-010454] [Server] A temporary password is generated for root@localhost: APWCY5ws&hjQ
...
```

**APWCY5ws&hjQ** 就是初始密码，后续登录需要用到，你也可以在登陆后修改密码。[ikOws<oyi7Gg]

输入以下安装命令：

```
mysqld install
```

启动输入以下命令即可：

```
net start mysql
```

**注意**: 在 5.7 需要初始化 data 目录：

```
cd C:\web\mysql-8.0.11\bin 
mysqld --initialize-insecure 
```

初始化后再运行 net start mysql 即可启动 mysql。

# 登录 MySQL

当 MySQL 服务已经运行时, 我们可以通过 MySQL 自带的客户端工具登录到 MySQL 数据库中, 首先打开命令提示符, 输入以下格式的命名:

```shell
mysql -h 主机名 -u 用户名 -p
```

参数说明：

- **-h** : 指定客户端所要登录的 MySQL 主机名, 登录本机(localhost 或 127.0.0.1)该参数可以省略;
- **-u** : 登录的用户名;
- **-p** : 告诉服务器将会使用一个密码来登录, 如果所要登录的用户名密码为空, 可以忽略此选项。

如果我们要登录本机的 MySQL 数据库，只需要输入以下命令即可：

```
mysql -u root -p
```

按回车确认, 如果安装正确且 MySQL 正在运行, 会得到以下响应:

```
Enter password:
```

若密码存在, 输入密码登录, 不存在则直接按回车登录。登录成功后你将会看到 Welcome to the MySQL monitor... 的提示语。

然后命令提示符会一直以 **mysq>** 加一个闪烁的光标等待命令的输入, 输入 **exit** 或 **quit** 退出登录。

# 修改MySQL登陆密码

故障现场：登陆到MySQL服务器,不管你执行什么命令都报这个错

```shell
mysql> show databases;
ERROR 1820 (HY000): You must reset your password using ALTER USER statement before executing this statement.
mysql> use test;
ERROR 1820 (HY000): You must reset your password using ALTER USER statement before executing this statement.
...
```

原因分析：这个主要是由一个参数控制的 default_password_lifetime,看看官方的解释

仔细看哈，Note信息有时候比上面的信息有用（英文的note我一般都是忽略的，有可能你忽略掉的那部分对性能也有帮助哦）

![](https://img-blog.csdn.net/20170324093931578?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaGo3amF5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

问题解决：在开源的世界里，我们不怕报错，有报错才有解决思路。下面来看下自己环境里的

```shell
mysql> select version();
+--------------+
| version()    |
+--------------+
| 5.7.10-3-log |
+--------------+
1 row in set (0.01 sec)

mysql> show variables like 'default_password_lifetime';
+---------------------------+-------+
| Variable_name             | Value |
+---------------------------+-------+
| default_password_lifetime | 360     |
+---------------------------+-------+
1 row in set (0.00 sec)

# 原来如此。那么就修改密码呗

alter user user() identified by "123456";
```

