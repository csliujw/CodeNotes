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

