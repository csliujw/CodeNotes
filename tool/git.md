# Git分布式版本控制工具

## 概述

- [x] 了解Git基本概念
- [x] 了解Git工作流程
- [x] 熟悉Git常用命令
- [x] 熟悉IDEA中Git的使用

> Git的作用

- [x] 代码备份：代码备份到GitHub上。
- [x] 代码还原：Git上记录了代码的提交内容，可以还原任一时间点的代码。
- [x] 协同开发：多人协作开发项目，通过Git协同开发，记录，分享所写代码。
- [x] 追溯代码问题，编写人和编写时间

> 版本控制方式

- 集中式版本控制工具 集中式版本控制工具

    - 版本库是集中存放在中央服务器的，team里每个人work时从中央服务器下载代码，是必须联网才能工作，局域网或互联网。个人修改后然后提交到中央版本库。
    -  举例：SVN和CVS b

- 分布式版本控制工具 

    - 分布式版本控制系统没有“中央服务器”，每个人的电脑上都是一个完整的版本库，这样工作的时候，无 需要联网了，因为版本库就在你自己的电脑上。多人协作只需要各自的修改推送给对方，就能互相看到对方的修改了。
    -  举例：Git

    > Git

    Git是分布式的,Git不需要有中心服务器，我们每台电脑拥有的东西都是一样的。我们使用Git并且有个中心服务器，仅仅是为了方便交换大家的修改，但是这个服务器的地位和我们每个人的PC是一样的。我们可以把它当做一个开发者的pc就可以就是为了大家代码容易交流不关机用的。没有它大家一样可以工作，只不过“交换”修改不方便而已。 

    git是一个开源的分布式版本控制系统，可以有效、高速地处理从很小到非常大的项目版本管理。Git是 Linus Torvalds 为了帮助管理 Linux 内核开发而开发的一个开放源码的版本控制软件。 同生活中的许多伟大事物一样，Git 诞生于一个极富纷争大举创新的年代。Linux 内核开源项目有着为数众 多的参与者。 绝大多数的 Linux 内核维护工作都花在了提交补丁和保存归档的繁琐事务上（1991－2002 年间）。 到 2002 年，整个项目组开始启用一个专有的分布式版本控制系统 BitKeeper 来管理和维护代 码。 

    到了 2005 年，开发 BitKeeper 的商业公司同 Linux 内核开源社区的合作关系结束，他们收回了 Linux 内核社区免费使用 BitKeeper 的权力。 这就迫使 Linux 开源社区（特别是 Linux 的缔造者 Linus Torvalds）基于使用 BitKeeper 时的经验教训，开发出自己的版本系统。 他们对新的系统制订 了若干目标： 

    - 速度
    - 简单的设计
    - 对非线性开发模式的强力支持（允许成千上万个并行开发的分支） 
    - 完全分布式
    - 有能力高效管理类似 Linux 内核一样的超大规模项目（速度和数量）

![image-20211217200528119](.\git.assets\image-20211217200528119.png)

## 工作流程图

![image-20211217200735238](.\git.assets\image-20211217200735238.png)

> 命令如下：

- clone（克隆）：从远程仓库中克隆代码到本地仓库 
- checkout （检出）:从本地仓库中检出一个仓库分支然后进行修订 
- add（添加）: 在提交前先将代码提交到暂存区 
- commit（提交）: 提交到本地仓库。本地仓库中保存修改的各个历史版本 
- fetch (抓取) ： 从远程库，抓取到本地仓库，不进行任何的合并动作，一般操作比较少。
- pull (拉取) ： 从远程库拉到本地库，自动进行合并(merge)，然后放到到工作区，相当于 fetch+merge 
- push（推送） : 修改完成后，需要和团队成员共享代码时，将代码推送到远程仓库

## 安装

### 常见Linux命令

- ls/ll 查看当前目录 
- cat 查看文件内容 
- touch 创建文件 
- vi vi编辑器（使用vi编辑器是为了方便展示效果，学员可以记事本、editPlus、notPad++等其它编 辑器）

### 安装

下载地址： https://git-scm.com/download

![image-20211217201447099](.\git.assets\image-20211217201447099.png)

安装时：Use git from git bash only...其他默认下一步。安装成功后鼠标点击右键可以看到。

![image-20211217201824831](.\git.assets\image-20211217201824831.png)

备注：

Git GUI：Git提供的图像界面工具

Git Bash：Git提供的命令行工具，提供了一些常见的Linux命令，如 curl。

当安装Git后首先要做的事情是设置用户名称和email地址。每次Git提交都会使用该用户信息

### 基本配置

- 点击右键 选择Git bash
- 配置全局变量
- git config -- global user.name "username" 如：git config -- global user.name "csxx"
- git config --global user.email "邮箱" 如：git config --global user.email "12312331@qq.com"
- 查看配置信息
    - git config --global user.name
    - git config --global user.email


### 为常用指令配置别名

有些常用的指令参数非常多，每次都要输入好多参数，我们可以使用别名。

- 打开用户目录，创建 .bashrc 文件 

- 部分windows系统不允许用户创建点号开头的文件，可以打开gitBash,执行 `touch ~/.bashrc`

- 在 .bashrc 文件中输入如下内容

    ```shell
    #用于输出git提交日志
    alias git-log='git log --pretty=oneline --all --graph --abbrev-commit'
    #用于输出当前目录所有文件及基本信息
    alias ll='ls -al'
    ```

### Git乱码

打开GitBash执行下面命令

```shell
git config --global core.quotepath false 
```

在`${git_home}/etc/bash.bashrc$` 文件后面加入下面两行

```shell
export LANG="zh_CN.UTF-8"
export LC_ALL="zh_CN.UTF-8"
```

### 配置免密登录

配置ssh：先在本地配置，发送给远程

输入以下命令 ssh-keygen -t rsa -C 邮箱 如：ssh-keygen -t rsa -C 324234234@qq.com 然后一直回车

打开github网站 找到setting --> new ssh - title任意/key输入本地生成的pubkey（公钥）,pubkey的存放地址请仔细看git控制台的输出。

测试连通性 ssh -T git@github.com[写死的]

本地和远程成功通信 则可以izai/.ssh中发现known_hosts文件，出错就多试几次 可能是网路问题。不行就检测建立ssh时输入的pub key。

## Git 常用命令

### 获取本地仓库

- 在电脑的任意位置创建一个空目录（例如test）作为我们的本地Git仓库
- 进入这个目录中，点击右键打开Git bash窗口                                           
- 执行命令git init 
- 如果创建成功后可在文件夹下看到隐藏的.git目录。

