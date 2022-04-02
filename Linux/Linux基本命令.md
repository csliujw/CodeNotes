# Linux基本命令

## 后台运行程序

如果想让一个程序在后台运行，只要在执行命令的末尾加上一个&符号就可以了。但是这种方式不是很保险，有些程序当你登出终端后它就会停止。

目前发现了三个命令可以在后台指向程序，关闭终端后程序也不会停止运行。

- screen -- 可以用，但是有时候会挂掉。
- nohub -- 可以用，开发中似乎常用这个命令。
- tmux -- 可以用，我一般用它在后台训练深度学习模型。

### screen

创建：screen -S 名称

查看会话：screen -ls

进入会话：screen -r

- 进入会话后，执行 python app.py，这样退出终端后代码也不会终止

删除 kill -9 194746  `There is a screen on"194746.name` 在 screen -wipe 就可以杀死进程了。

### tmux

- [x] 安装

```shell
# Ubuntu 或 Debian
$ sudo apt-get install tmux

# CentOS 或 Fedora
$ sudo yum install tmux

# Mac
$ brew install tmux
```

- [x] 启动&推出

```shell
tmux # 启动
exit # 退出 或 快捷键 ctrl + d
```

- [x] 创建&杀死会话

```shell
tmux new -s session-name # 创建会话
tmux kill-session # 命令用于杀死某个会话
# 使用会话编号
$ tmux kill-session -t 0

# 使用会话名称
$ tmux kill-session -t <session-name>
```

- [x] 接入会话 重新连接某个已经存在的会话

```shell
# 使用会话编号
$ tmux attach -t 0

# 使用会话名称
$ tmux attach -t <session-name>
```

### nohup

**nohup** 英文全称 no hang up（不挂起），用于在系统后台不挂断地运行命令，退出终端不会影响程序的运行。

安装 nohup 命令： `sudo apt install coreutils`

```java
// 事先准备的 Java 代码
import java.util.concurrent.TimeUnit;
public class Demo{
    public static void main(String[]args) throws Exception{
        new Thread(()->{
            while(true){
                try{
                    TimeUnit.SECONDS.sleep(1);
                    System.out.println("````");
                }catch(Exception e){
                    // pass
                }
            }
        }).start();
    }
}
```

编译运行这段 Java 代码

```shell
javac Demo.java
java Demo
# 关闭终端，在打开另一个终端，jps 命令查看，发现 java 进程被终止了。
```

使用 nohup 命令，后台运行进程，关闭终端不中断进程。

```bash
nohup java Demo # 在后台运行 demo 这个 Java 代码  代码中的输出语句，默认会被重定向到 /home/用户名/nohup.out 文件中

sudo nohup java Demo > ./out.txt # 修改重定向的位置。把输出的内容重定向到 当前目录的 out.txt 文件中。（文件不用我们自己创建，命令自己会创建文件）

nohup java Demo & > ./out.txt # 加上 & 直接在后台运行，不在终端显示。
```

- jobs -l：列出当前在后台执行的命令
- fg N：将命令进程号码为N的命令进程放到前台执行，同%N
- bg N：将命令进程号码为N的命令进程放到后台执行，同%N，%N是通过jobs命令查到的后台正在执行的命令的序号，不是pid

## 文件和目录操作

### 进入/查找

- ls

- cd

- pwd

- find：查找文件

    - `find . -name "*.c"`   将当前目录及其子目录下所有文件后缀为 **.c** 的文件列出来
    - `find . -ctime -20`   将当前目录及其子目录下所有最近 20 天内更新过的文件列出
    - `find /home -name "*.txt" -ctime -20`  查找 home 目录下 名称为`*.txt` 最近 20 天内更新过的文件列出来
    - `find . -type f`    将目前目录其其下子目录中所有一般文件列出

- file/stat：查找文件类型或文件属性信息

    - `file a.txt`  列出文件a.txt的属性

    - `file -b a.txt`  列出文件a.txt的属性,忽略name

    - `stat a.txt`  查看 a.txt文件的inode内容内容

        ```shell
          File: a.txt
          Size: 29              Blocks: 0          IO Block: 4096   regular file
        Device: 2h/2d   Inode: 6192449487700424  Links: 1
        Access: (0666/-rw-rw-rw-)  Uid: ( 1000/liujiawei)   Gid: ( 1000/liujiawei)
        Access: 2021-07-11 18:28:59.417365700 +0800
        Modify: 2021-07-11 18:28:59.417919200 +0800
        Change: 2021-07-11 18:31:42.793002500 +0800
         Birth: -
        ```

- echo：把内容重定向到指定的文件中，有则打开，无则创建。

- 管道命令 |

    - 将前面的结果给后面的命令，例如：`ls -la | wc`，将ls的结果交由`wc`命令来统计字数。

- 重定向 

    - `>`是覆盖模式
    - `>>`是追加模式
    - 如：`echo hello world  > demo.txt` 把左边的输出放到右边的文件中去。

### 创建/拷贝

- mkdir
- rmdir
- touch
- cp：复制文件或目录

    - cp 源文件 目标文件
    - `cp   demo.txt   copy.txt` `把demo.txt 复制 到 copy.txt`
- mv：移动文件或目录、文件或目录改名

    - `mv a.txt copy/` `把 a.txt 移动到 copy/ 目录下`
- rm：删除文件
- ln：创建软链接

### 查看文件

- cat	查看文本文件内容
- more	可分页查看
    - `more -20 demo.txt`   显示前20行
    - `more +20 demo.txt`   前20行不显示
- less    可分页，可搜索，回翻
- `tail -10`    查看文件的尾部的10行
- `head -20`    查看文件的头部20行

## 打包和压缩文件

`gzip`只能用来压缩文件

- `gzip filename` 不保留源文件   `gzip -c filename > newfilename.gz` 保留源文件
    - `gzip demo.txt` `压缩 demo.txt 文件`
    - `gzip *` `压缩当前目录下的所有文件`
    - `gzip -d demo.txt.gz` `解压demo.txt.gz文件`
- `bzip2 filename` 和 `gzip` 的用法一样
- `tar -czvf filename`  压缩文件
- `tar -xzvf filename.tar.gz`  解压文件

## 上传下载

### SZ&RZ

[Linux rz/sz命令 在终端直接上传下载文件_随波一落叶-CSDN博客](https://blog.csdn.net/BobYuan888/article/details/86603749?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1.pc_relevant_default&utm_relevant_index=2)

### SSH 下载文件

在 windows terminal 上直接执行命令，将 /root/Manipulator.cpp 文件拷贝到 D:// 目录下。

```bash
scp root@114.132.43.225:/root/Manipulator.cpp D://
```

### SSH 上传文件

同上，只是命令不一样

```bash
scp D:\test.bmp root@114.132.43.225:/root/
```

## 网络管理

### 网络接口相关

`ifconfig`：查看网络接口信息

`ifup/ifdown`：开启或关闭接口

`netstat`：查看网络状态

`tcpdump`：显示本机网络流量的状态

`traceroute`：检测到目的主机所经过的路由器

`host/dig/nslookup`：检测DNS解析

## 安装软件

`wget` : `wget   www.baidu.com`  帮我把百度的首页下载过来了。

## 进程管理

### 查看进程&杀死

#### 查看进程信息

```bash
# 查看进程
ps # 查看进程的基本信息

man ps # 查看 ps 的帮助文档
EXAMPLES
To see every process on the system using standard syntax: # 标准语法
    ps -e
    ps -ef
    ps -eF
    ps -ely

To see every process on the system using BSD syntax: # BSD 语法。信息更多，可以看到 CPU 和 MEM 使用率
    ps ax
    ps axu

To print a process tree:
    ps -ejH
    ps axjf

To get info about threads:
    ps -eLf
    ps axms

To get security info:
    ps -eo euser,ruser,suser,fuser,f,comm,label
    ps axZ
    ps -eM

To see every process running as root (real & effective ID) in user format:
    ps -U root -u root u

To see every process with a user-defined format:
    ps -eo pid,tid,class,rtprio,ni,pri,psr,pcpu,stat,wchan:14,comm
    ps axo stat,euid,ruid,tty,tpgid,sess,pgrp,ppid,pid,pcpu,comm
    ps -Ao pid,tt,user,fname,tmout,f,wchan

Print only the process IDs of syslogd:
    ps -C syslogd -o pid=

Print only the name of PID 42:
    ps -q 42 -o comm=
```

#### 结合管道命令查看

```bash
man grep # 查看 grep 的帮助手册

ps -ef | grep fzz # ps -ef 查询出信息，然后找出信息中包含 fzz 的进程。更多细致用法请查看帮助手册。
```

#### 杀死进程

- kill：kill 命令杀死指定进程 PID 的进程

```bash
# 查找一个top进程，并杀死
www@www:/$ ps -ef| grep top
www+   277   186  0 16:16 pts/1    00:00:00 top
www+   278   240  0 16:16 pts/2    00:00:00 top
www+   279   257  0 16:16 pts/3    00:00:00 top
www+   290   154  0 16:17 pts/0    00:00:00 grep --color=auto top
www@www:/$ kill -9 277
www@www:/$ ps -ef| grep top
www+   278   240  0 16:16 pts/2    00:00:00 top
www+   279   257  0 16:16 pts/3    00:00:00 top
www+   292   154  0 16:17 pts/0    00:00:00 grep --color=auto top
www@www:/$
```

- killall：killall 命令用于杀死指定名字的进程（kill processes by name）
    - killall -9 top 杀死所有 top 进程。
- pkill：pkill 和 killall 差不多，也是用于杀掉指定名称的进程

- `ps：查找进程的信息`
    - `ps`
- `nice和renice：调整进程的优先级`
    - `nice -n 1 ls`  将 ls 的优先序加 1 并执行
    - `nice ls`  将 ls 的优先序加 10 并执行
- `kill：杀死进程`
    - `kill pid` 杀死进程
    - `kill -KILL 12346`  强制杀死进程
- `free：查看内存使用情况`
    - `free -m` 以mb为单位查看内存情况
    - `free -t`   以总和的形式查询内存的使用信息
    - `free -s 10`  每10s 执行一次命令
- `top：查看实时刷新的系统进程信息`
- 作业管理
    - jobs：列举额作业号码和名称
    - bg：在后台恢复运行
    - fg：在前台恢复运行
    - ctr+z：暂时停止某个进程
- 自动任务：
    - at
    - cron
- 管理守护进程
    - chkconfig
    - service
    - ntsysv

## 管理用户

### 创建用户

```bash
sudo useradd -m qq # 创建用户 qq  加上参数 -m 会帮助我们自动创建用户的 home 目录
sudo passwd qq # 设置用户 qq 的密码
Enter new UNIX password:
Retype new UNIX password:
passwd: password updated successfully

su qq # 且换到用户 qq

sudo userdel qq # 删除用户 qq
```

