# Linux基本命令

## screen

创建：screen -S 名称

查看会话：screen -ls

进入会话：screen -r

- 进入会话后，执行 python app.py，这样退出终端后代码也不会终止

删除 kill -9 194746  `There is a screen on"194746.name` 在 screen -wipe 就可以杀死进程了。

## 文件和目录操作

- ls

- cd

- pwd

- mkdir

- rmdir

- touch

- cp：复制文件或目录

    - cp 源文件 目标文件
    - `cp   demo.txt   copy.txt` `把demo.txt 复制 到 copy.txt`

- mv：移动文件或目录、文件或目录改名

    - `mv a.txt copy/` `把 a.txt 移动到 copy/ 目录下`

- rm：删除文件

- ln

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

## 查看文件

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