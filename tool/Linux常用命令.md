# 介绍

没啥好介绍的，服务器一般就用Linux。

# Linux发行版本

## 分类

知道这些，方便选择Linux。

- Linux内核
  - Debian
    - Ubuntu【Linux Mint】
  - Fedora
    - RHEL【CentOS / Oracle Linux】
  - SUSE
    - SLES【openSUSE】
  - 其他发行版本

## Linux版本推荐

- **CentOS-7.0-x86_64-DVD-1503-01.iso** : 标准安装版，一般下载这个就可以了（推荐）
- **CentOS-7.0-x86_64-NetInstall-1503-01.iso** : 网络安装镜像（从网络安装或者救援系统）
- **CentOS-7.0-x86_64-Everything-1503-01.iso**: 对完整版安装盘的软件进行补充，集成所有软件。（包含centos7的一套完整的软件包，可以用来安装系统或者填充本地镜像）
- **CentOS-7.0-x86_64-GnomeLive-1503-01.iso**: GNOME桌面版
- **CentOS-7.0-x86_64-KdeLive-1503-01.iso**: KDE桌面版
- **CentOS-7.0-x86_64-livecd-1503-01.iso** : 光盘上运行的系统，类拟于winpe
- **CentOS-7.0-x86_64-minimal-1503-01.iso** : 精简版，自带的软件最少

# Linux系统启动过程

- 内核引导
  - 打开电源后 BIOS开机自检，安装BIOS中的设置启动设备。
- 运行init
  - init进程是系统所有进程的起点，可把他比作系统所有进程的父亲。
  - init需要读取配置文件 /etc/inittab
  - 然后再确定运行级别
- 系统初始化
- 建立终端
- 用户登录系统

CentOS7配置文件 `/usr/lib/systemd/system`   `/etc/systemd/system`

# Linux入门命令

## 正确的关机

- 图形模型与文字模式的切换
  - Linux预设提供了六个命令窗口终端机让我们来登录，默认我们登录的就是第一个窗口，也就是tty1，这个六个窗口分别为tty1,tty2 … tty6，你可以按下Ctrl + Alt + F1 ~ F6 来切换它们。当你进入命令窗口界面后再返回图形界面只要按下Ctrl + Alt + F7 就回来了。如果你用的vmware 虚拟机，命令窗口切换的快捷键为 Alt + Space + F1~F6. 如果你在图形界面下请按Alt + Shift + Ctrl + F1~F6 切换至命令窗口。

- 关机
  - 正确流程： sync > shutdown > reboot > halt
  - sync将数据同步到硬盘中
  - shutdown 关机指令
  - reboot是重启 相当于 shutdown -r now
  - halt 关闭系统，相当于 shutdown -h now 和 poweroff
- eg
  - sync
  - shutdown -h 10

## ls和主要目录的作用

### 各个目录说明

- ls 查看目录接口

  - /bin  存放了常用的命令

  - /boot 存放了启动linux时使用的一些核心文件，包括一些连接文件以及镜像文件

  - /dev 存放Linux的外部设备

  - /etc 存放系统关联所需要的配置文件和子目录。一般我们需要给安装的软件修改配置就是从这里面找

  - /home 用户的主目录

  - /lib OS最基本的动态链接共享库，类似于windows的DLL文件

  - /lost+found 一般为空，os非法关机后 这里会存放一些文件

  - /media 被识别的设备会被挂载在这里

  - /mnt 让用户临时挂载别的文件系统的，我们可以将光驱挂载在/mnt/上，然后进入该目录就可以查看光驱里的内容了。

  - /opt 这是给主机额外安装软件所摆放的目录。比如你安装一个ORACLE数据库则就可以放到这个目录下。默认是空的。

  - /proc 这个目录是一个虚拟的目录，它是系统内存的映射，我们可以通过直接访问这个目录来获取系统信息。
    这个目录的内容不在硬盘上而是在内存里，我们也可以直接修改里面的某些文件，比如可以通过下面的命令来屏蔽主机的ping命令，使别人无法ping你的机器：

  - /root 系统管理员

  - /sbin 存放系统管理员使用的系统管理程序

  - /selinux Redhat/CentOS特有的目录，类似于wingdows的防火墙

  - /srv 存放服务启动之后需要提取的数据

  - /sys 这是linux2.6内核的一个很大的变化。该目录下安装了2.6内核中新出现的一个文件系统 sysfs 。

    sysfs文件系统集成了下面3种文件系统的信息：针对进程信息的proc文件系统、针对设备的devfs文件系统以及针对伪终端的devpts文件系统。

    

    该文件系统是内核设备树的一个直观反映。

    当一个内核对象被创建的时候，对应的文件和目录也在内核对象子系统中被创建。

  - /tmp 存放临时文件

  - /usr 用户的很多应用程序和文件都放在这个目录下

  - /usr/bin 系统用户使用的应用程序。

  - /usr/sbin 超级用户使用的比较高级的管理程序和系统守护程序。

  - /usr/src 内核源代码默认的放置目录。

  - /var 这个目录中存放着在不断扩充着的东西

  - /run 是一个临时文件系统，存储系统启动以来的信息。当系统重启时，这个目录下的文件应该被删掉或清除。如果你的系统上有 /var/run 目录，应该让它指向 run。

### 文件的基本属性

**可读可写可执行**

- r 可读
- w 可写
- x 可执行

d rwx

- d 目录文件
- -文件
- l 链接文件
- b 可供存储的接口设备
- c 串行端口设备，如鼠标，键盘。

  