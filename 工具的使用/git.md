# git版本控制软件

## 命令

```powershell
git add: # 将本地文件 增加到暂存区
git commit # 将暂存区的内容 提交到 本地仓库
git push # 将本地仓库的内容 推送到 远程仓库
git pull # 将远程仓库的内容 拉取到 本地仓库
```

## 安装

安装时：Use git from git bash only。。。其他默认下一步。

安装好后将git的bin配置到系统环境变量中。是系统环境变量。
配置github
    点击右键 选择Git bash

配置全局变量

git config -- global user.name "liujiaweiReal"

git config --global user.email "695466632@qq.com"

## 配置免密登录

配置ssh：现在本地配置，发送给远程

输入以下命令
ssh-keygen -t rsa -C 695466632@qq.com
然后一直回车

打开github网站 找到setting -- new ssh - title任意/key输入本地生成的pubkey（公钥）

测试连通性

ssh -T git@github.com[写死]

本地和远程成功通信 则可以izai/.ssh中发现known_hosts文件
出错就多试几次 可能是网路问题。不行就检测建立ssh时输入的pub key

## Git练习

```powershell
# 初始化git项目
git init

# 关联git仓库
git remote add origin git项目的免密地址
# 例如

# 添加
git add .

# 提交
git commit .  或者 git commit -m "提交的内容说明"

# 存入仓库
git push origin master

# 如果push失败提示这些内容
To github.com:liujiaweiReal/layui.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:liujiaweiReal/layui.git'

# 则使用 这个命令同步下本地仓库和git仓库
git pull origin master --allow-unrelated-histories

# 再 存入仓库
git push origin master

# 从仓库中拉取项目
git pull
```

```powershell
# 第一次下载项目（远程-本地）
git clone git@github.com:yanqun/mygitremote.git

# 提交(本地-远程)
# (在当前工作目录 右键-git bash)
git add.
git commit -m "提交到分支"
git push  origin master

# 更新(远程-本地)
git pull
# 从远程仓库中把代码拉过来！
git pull origin master
```

# Git创建分支

```powershell
Git init
```

2、上传修改的文件

```powershell
git add *
```

(*可替换成具体要上传的文件名，*表示提交所有有变化的文件) 3、添加上传文件的描述

```powershell
git commit -m "test" 
```

 （”test“为分支名）

4、（创建分支）

```powershell
git branch test
```

5、（切换分支）

```powershell
git checkout test
```

6、与远程分支相关联

```powershell
git remote add origin https://github.com/yangxiaoyan20/BowlingScore.git
```

  （”BowlingScore“ 为工程名）

7、（将分支上传）

```powershell
git push origin test
```



git常用命令总结，开发流程

架构师在gitlab上新建了一个项目，搭好了框架

1.我作为开发者之一，首先git clone [https://xx](https://xx/)

用idea打开项目，然后点开idea下面的console, 在这里面执行git命令

![img](https://img-blog.csdnimg.cn/20190227200719109.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21hdHJpeF9nb29nbGU=,size_16,color_FFFFFF,t_70)

刚进来自然是master分支，然后我们首先创建一个自己的分支并切换过去，命令如下

git checkout -b feature/20190227_col

执行完后如下

![img](https://img-blog.csdnimg.cn/20190227200719107.png)

下一次从master切换到这个分支执行git checkout  feature/20190227_col，注意不能有-b选项，否则报错，该分支已存在，使用git branch可以列出所有分支

另外这里记录一下如何删除分支

git branch -d dev

 

2.然后在自己的分支上做代码开发，开发完成之后，提交之前，先执行git pull origin feature/20190227_col，origin是远程仓库名，feature/20190227_col是分支名，一个仓库下有多个分支，这个概念一定要清楚，执行完后就和自己的仓库分支同步了，注意直接git pull不能拉到自己的分支，必须加上仓库名和自己的分支名

如果切换到master分支下，拉远程仓库master分支的代码，直接执行git pull即可

 

3.执行git add .

直接执行上述命令可能会加入很多idea自带文件，可以一个一个文件add，也可以一次添加一个目录下的文件

git add util/

git add util/redisUtil.scala 

如果想要撤销add

如果是git add . 撤销操作 git reset HEAD .

如果是git add file 撤销操作 git reset HEAD file

 

4.添加完成后git commit -m “update 01”

 

5.执行git push origin feature/20190227_col

提交到远程仓库自己分支上，因为是自己的分支，所以不需要评审，直接就进去了

 

6.和master合并，在gitlab的web页面上发起一个合入请求，并指定评审人

 

7.假设下一次开发仍然使用feature/20190227_col这个分支，使用之前先切换到这个分支

执行git pull origin master 从远程origin仓库中拉取master分支

 

8.在idea中如何查看两个分支的diff

选中工程-》右键-》git-》compare with branch

----

**git错误**

```java
ssh -T git@github.com

ssh: connect to host github.com port 22: Connection timed out
```

错误原因，端口不行（换个端口）。

```text
Host github.com
User git
Hostname ssh.github.com
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_rsa
Port 443

Host gitlab.com
Hostname altssh.gitlab.com
User git
Port 443
PreferredAuthentications publickey
IdentityFile ~/.ssh/id_rsa
```