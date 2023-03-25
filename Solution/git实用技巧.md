# Learn Git Branch

## 基础



## 基础

`git stash save “xxx”` # 临时保存修改的文件

`git log --oneline --graph` # 日志单行流程图

`git push :branch_name` # 删除远端分支

`git rm --cached filename` # 取消追踪某个文件  （去除已经commit了的文件）

`git commit --amend` # 提交一个 commit 但是记录到上一次 log 中（commit 时不想再新增一个记录，压缩两个日志）

`git cherry-pick commit-id` # 提交特定一次修改 （把特定的某一次提交的内容给粘到另一个分支上面）[git 教程 --git cherry-pick 命令 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/355413226)

`git archive -o archive.zip master` # 文件快照打包

## 代码冲突

git merge master

git rebase -i master

## 远程操作

如何将本地文件与远程仓库文件关联起来（将本地的一个文件夹推到 github 上先用 git init）

`git remote -v` # 查看当前目录被哪些仓库关联

`git remote add origin git@github.com:xxx/xxx.git` // 本地的仓库与远程的关联

`git remote rm origin` # 不想和远程仓库关联

`git push -u origin master` # 推送到上游分支（下次不用指定分支）

`git checkout -b local_branch_yy origin/remote_branch_yy` #从远程仓库获取指定分支

## 压缩提交记录

如果有多次提交记录 却只想保留一个记录

方法一 RESET

1. `git reset commit-id` # 将这次的commit压缩到前一个commit到id上
2. `git add filename` #
3. `git commit -m "xxx"`

方法二 AMEND

1. `git commit --amend "xxx"` # 每次压缩一条提交记录

方法三 REBASE

1. `git rebase -i commit-id` # 到想要压缩内容的前一次提交
2. `pick ef124f` 保留
3. `squash fa4f8d` 压缩

## 标签

不同版本 v1.0.0 的由来

```shell
git tag tag_name

git tag tag_name commit-id

git tag -a tag_name -m "xxx" commit-id

git tag

git show tag_name

git push origin tag_name # 推送到远程仓库
```



