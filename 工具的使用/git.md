# git版本控制软件

## 命令

```powershell
git add: # 将本地文件 增加到暂存区
git commit # 将暂存区的内容 提交到 本地仓库
git push # 将本地仓库的内容 推送到 远程仓库
git pull # 将远程仓库的内容 拉取到 本地仓库
```

## 安装

安装时：Use git from git bash only...其他默认下一步。

安装好后将git的bin配置到系统环境变量中。是系统环境变量。
配置github

- 点击右键 选择Git bash

- 配置全局变量

- git config -- global user.name "liujiaweiReal"

- git config --global user.email "695466632@qq.com"

## 配置免密登录

配置ssh：先在本地配置，发送给远程

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
git commit .  或者 git commit -m"提交的内容说明"

# 存入仓库
git push origin master

# 如果push失败提示这些内容
To github.com:liujiaweiReal/layui.git
 ! [rejected]        master -> master (non-fast-forward)
error: failed to push some refs to 'git@github.com:liujiaweiReal/layui.git'

# 则使用 这个命令同步下本地仓库和git仓库
git pull origin master --allow-unrelated-histories

# 再存入仓库
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
git commit -m"提交到分支"
git push origin master

# 更新(远程-本地)
git pull
# 从远程仓库中把代码拉过来！
git pull origin master
```

# Git创建分支

## 完整流程

初始化git

```powershell
Git init
```

上传修改的文件：(*可替换成具体要上传的文件名，*表示提交所有有变化的文件)

```powershell
git add *
```

添加上传文件的描述

```powershell
git commit -m "test" 
```

 创建分支：（”test“为分支名）

```powershell
git branch test
```

切换分支

```powershell
git checkout test
```

与远程分支相关联：（”BowlingScore“ 为工程名）

```powershell
git remote add origin https://github.com/yangxiaoyan20/BowlingScore.git
```

 将分支上传

```powershell
git push origin test
```

## git错误

```shell
ssh -T git@github.com

ssh: connect to host github.com port 22: Connection timed out
```

错误原因，端口不行（换个端口）。

```shell
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

# Git版本控制

## 版本回退

```shell
git log # 查看提交记录
git reset --hard 版本id
```

# 常用Git命令汇总

**日常积累**

```shell
git log # 查看提交记录
git reset --hard 版本id

# 强制修改分支名称
$ git branch -M [<原分支名称>] <新的分支名称>
# 删除指定的本地分支
$ git branch -d <分支名称>
# 强制删除指定的本地分支
$ git branch -D <分支名称>

# 删除git服务器上的分支
git push origin -d BranchName
git push origin --delete BranchName
```



# Git命令大全

## git config

配置 Git 的相关参数。

Git 一共有3个配置文件：

1. 仓库级的配置文件：在仓库的 `.git/.gitconfig`，该配置文件只对所在的仓库有效。
2. 全局配置文件：Mac 系统在 `~/.gitconfig`，Windows 系统在 `C:\Users\<用户名>\.gitconfig`。
3. 系统级的配置文件：在 Git 的安装目录下（Mac 系统下安装目录在 `/usr/local/git`）的 `etc` 文件夹中的 `gitconfig`。

```ruby
# 查看配置信息
# --local：仓库级，--global：全局级，--system：系统级
$ git config <--local | --global | --system> -l

# 查看当前生效的配置信息
$ git config -l

# 编辑配置文件
# --local：仓库级，--global：全局级，--system：系统级
$ git config <--local | --global | --system> -e

# 添加配置项
# --local：仓库级，--global：全局级，--system：系统级
$ git config <--local | --global | --system> --add <name> <value>

# 获取配置项
$ git config <--local | --global | --system> --get <name>

# 删除配置项
$ git config <--local | --global | --system> --unset <name>

# 配置提交记录中的用户信息
$ git config --global user.name <用户名>
$ git config --global user.email <邮箱地址>

# 更改Git缓存区的大小
# 如果提交的内容较大，默认缓存较小，提交会失败
# 缓存大小单位：B，例如：524288000（500MB）
$ git config --global http.postBuffer <缓存大小>

# 调用 git status/git diff 命令时以高亮或彩色方式显示改动状态
$ git config --global color.ui true

# 配置可以缓存密码，默认缓存时间15分钟
$ git config --global credential.helper cache

# 配置密码的缓存时间
# 缓存时间单位：秒
$ git config --global credential.helper 'cache --timeout=<缓存时间>'

# 配置长期存储密码
$ git config --global credential.helper store
```

## git clone

从远程仓库克隆一个版本库到本地。

```bash
# 默认在当前目录下创建和版本库名相同的文件夹并下载版本到该文件夹下
$ git clone <远程仓库的网址>

# 指定本地仓库的目录
$ git clone <远程仓库的网址> <本地目录>

# -b 指定要克隆的分支，默认是master分支
$ git clone <远程仓库的网址> -b <分支名称> <本地目录>
```

## git init

初始化项目所在目录，初始化后会在当前目录下出现一个名为 .git 的目录。

```ruby
# 初始化本地仓库，在当前目录下生成 .git 文件夹
$ git init
```

## git status

查看本地仓库的状态。

```ruby
# 查看本地仓库的状态
$ git status

# 以简短模式查看本地仓库的状态
# 会显示两列，第一列是文件的状态，第二列是对应的文件
# 文件状态：A 新增，M 修改，D 删除，?? 未添加到Git中
$ git status -s
```

## git remote

操作远程库。

```ruby
# 列出已经存在的远程仓库
$ git remote

# 列出远程仓库的详细信息，在别名后面列出URL地址
$ git remote -v
$ git remote --verbose

# 添加远程仓库
$ git remote add <远程仓库的别名> <远程仓库的URL地址>

# 修改远程仓库的别名
$ git remote rename <原远程仓库的别名> <新的别名>

# 删除指定名称的远程仓库
$ git remote remove <远程仓库的别名>

# 修改远程仓库的 URL 地址
$ git remote set-url <远程仓库的别名> <新的远程仓库URL地址>
```

## git branch

操作 Git 的分支命令。

```ruby
# 列出本地的所有分支，当前所在分支以 "*" 标出
$ git branch

# 列出本地的所有分支并显示最后一次提交，当前所在分支以 "*" 标出
$ git branch -v

# 创建新分支，新的分支基于上一次提交建立
$ git branch <分支名>

# 修改分支名称
# 如果不指定原分支名称则为当前所在分支
$ git branch -m [<原分支名称>] <新的分支名称>
# 强制修改分支名称
$ git branch -M [<原分支名称>] <新的分支名称>

# 删除指定的本地分支
$ git branch -d <分支名称>

# 强制删除指定的本地分支
$ git branch -D <分支名称>
```

```shell
# 删除git服务器上的分支
git push origin -d BranchName
git push origin --delete BranchName
```

## git checkout

检出命令，用于创建、切换分支等。

```ruby
# 切换到已存在的指定分支
$ git checkout <分支名称>

# 创建并切换到指定的分支，保留所有的提交记录
# 等同于 "git branch" 和 "git checkout" 两个命令合并
$ git checkout -b <分支名称>

# 创建并切换到指定的分支，删除所有的提交记录
$ git checkout --orphan <分支名称>

# 替换掉本地的改动，新增的文件和已经添加到暂存区的内容不受影响
$ git checkout <文件路径>
```

## git cherry-pick

把已经提交的记录合并到当前分支。

```ruby
# 把已经提交的记录合并到当前分支
$ git cherry-pick <commit ID>
```

## git add

把要提交的文件的信息添加到暂存区中。当使用 git commit 时，将依据暂存区中的内容来进行文件的提交。

```csharp
# 把指定的文件添加到暂存区中
$ git add <文件路径>

# 添加所有修改、已删除的文件到暂存区中
$ git add -u [<文件路径>]
$ git add --update [<文件路径>]

# 添加所有修改、已删除、新增的文件到暂存区中，省略 <文件路径> 即为当前目录
$ git add -A [<文件路径>]
$ git add --all [<文件路径>]

# 查看所有修改、已删除但没有提交的文件，进入一个子命令系统
$ git add -i [<文件路径>]
$ git add --interactive [<文件路径>]
```

## git commit

将暂存区中的文件提交到本地仓库中。

```ruby
# 把暂存区中的文件提交到本地仓库，调用文本编辑器输入该次提交的描述信息
$ git commit

# 把暂存区中的文件提交到本地仓库中并添加描述信息
$ git commit -m "<提交的描述信息>"

# 把所有修改、已删除的文件提交到本地仓库中
# 不包括未被版本库跟踪的文件，等同于先调用了 "git add -u"
$ git commit -a -m "<提交的描述信息>"

# 修改上次提交的描述信息
$ git commit --amend
```

## git fetch

从远程仓库获取最新的版本到本地的 tmp 分支上。

```ruby
# 将远程仓库所有分支的最新版本全部取回到本地
$ git fetch <远程仓库的别名>

# 将远程仓库指定分支的最新版本取回到本地
$ git fetch <远程主机名> <分支名>
```

## git merge

合并分支。

```ruby
# 把指定的分支合并到当前所在的分支下
$ git merge <分支名称>
```

## git diff

比较版本之间的差异。

```ruby
# 比较当前文件和暂存区中文件的差异，显示没有暂存起来的更改
$ git diff

# 比较暂存区中的文件和上次提交时的差异
$ git diff --cached
$ git diff --staged

# 比较当前文件和上次提交时的差异
$ git diff HEAD

# 查看从指定的版本之后改动的内容
$ git diff <commit ID>

# 比较两个分支之间的差异
$ git diff <分支名称> <分支名称>

# 查看两个分支分开后各自的改动内容
$ git diff <分支名称>...<分支名称>
```

## git pull

从远程仓库获取最新版本并合并到本地。
首先会执行 `git fetch`，然后执行 `git merge`，把获取的分支的 HEAD 合并到当前分支。

```ruby
# 从远程仓库获取最新版本。
$ git pull
```

## git push

把本地仓库的提交推送到远程仓库。

```ruby
# 把本地仓库的分支推送到远程仓库的指定分支
$ git push <远程仓库的别名> <本地分支名>:<远程分支名>

# 删除指定的远程仓库的分支
$ git push <远程仓库的别名> :<远程分支名>
$ git push <远程仓库的别名> --delete <远程分支名>
```

## git log

显示提交的记录。

```bash
# 打印所有的提交记录
$ git log

# 打印从第一次提交到指定的提交的记录
$ git log <commit ID>

# 打印指定数量的最新提交的记录
$ git log -<指定的数量>
```

## git reset

还原提交记录。

```ruby
# 重置暂存区，但文件不受影响
# 相当于将用 "git add" 命令更新到暂存区的内容撤出暂存区，可以指定文件
# 没有指定 commit ID 则默认为当前 HEAD
$ git reset [<文件路径>]
$ git reset --mixed [<文件路径>]

# 将 HEAD 的指向改变，撤销到指定的提交记录，文件未修改
$ git reset <commit ID>
$ git reset --mixed <commit ID>

# 将 HEAD 的指向改变，撤销到指定的提交记录，文件未修改
# 相当于调用 "git reset --mixed" 命令后又做了一次 "git add"
$ git reset --soft <commit ID>

# 将 HEAD 的指向改变，撤销到指定的提交记录，文件也修改了
$ git reset --hard <commit ID>
```

## git revert

生成一个新的提交来撤销某次提交，此次提交之前的所有提交都会被保留。

```ruby
# 生成一个新的提交来撤销某次提交
$ git revert <commit ID>
```

## git tag

操作标签的命令。

```ruby
# 打印所有的标签
$ git tag

# 添加轻量标签，指向提交对象的引用，可以指定之前的提交记录
$ git tag <标签名称> [<commit ID>]

# 添加带有描述信息的附注标签，可以指定之前的提交记录
$ git tag -a <标签名称> -m <标签描述信息> [<commit ID>]

# 切换到指定的标签
$ git checkout <标签名称>

# 查看标签的信息
$ git show <标签名称>

# 删除指定的标签
$ git tag -d <标签名称>

# 将指定的标签提交到远程仓库
$ git push <远程仓库的别名> <标签名称>

# 将本地所有的标签全部提交到远程仓库
$ git push <远程仓库的别名> –tags
```

## git mv

重命名文件或者文件夹。

```ruby
# 重命名指定的文件或者文件夹
$ git mv <源文件/文件夹> <目标文件/文件夹>
```

## git rm

删除文件或者文件夹。

```ruby
# 移除跟踪指定的文件，并从本地仓库的文件夹中删除
$ git rm <文件路径>

# 移除跟踪指定的文件夹，并从本地仓库的文件夹中删除
$ git rm -r <文件夹路径>

# 移除跟踪指定的文件，在本地仓库的文件夹中保留该文件
$ git rm --cached
```

## Git操作场景示例

### 1. 删除掉本地不存在的远程分支

多人合作开发时，如果远程的分支被其他开发删除掉，在本地执行 `git branch --all` 依然会显示该远程分支，可使用下列的命令进行删除：

```ruby
# 使用 pull 命令，添加 -p 参数
$ git pull -p

# 等同于下面的命令
$ git fetch -p
$ git fetch --prune origin
```

