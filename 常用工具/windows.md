> **termius**

终端输出乱码怎么办

```shell
export LANG=en_US.UTF-8
```

> <b>wsl 内存占用过大</b>

- 按下 Windows + R 键，输入 %UserProfile% 并运行进入用户文件夹

- 新建文件 .wslconfig ，然后记事本编辑

- 填入以下内容并保存, memory 为系统内存上限，可根据自身电脑配置设置

    ```notepad++
    [wsl2]
    memory=4GB
    swap=6
    localhostForwarding=true
    ```

    终端输入：wls --shutdown 。

- 重新启动 wls 子系统。

[(28条消息) 限制wsl2占用过多内存_ZZULI_星.夜的博客-CSDN博客_wsl2内存占用](https://blog.csdn.net/weixin_43906799/article/details/111562984?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-111562984-blog-120802756.pc_relevant_recovery_v2&spm=1001.2101.3001.4242.3&utm_relevant_index=7)

> <b>关闭 wsl</b>

```shell
wsl --shutdown
```

> 配置 wsl 的最大内存

在 `C:\Users\<你的用户名>` 下创建 `.wslconfig` 文件，在里面输入

```shell
[wsl2]
memory=4GB			# 最大内存为 4G
processors=4
swap=1GB
```

[docker中 WSL 配置 修改 - PanPan003 - 博客园 (cnblogs.com)](https://www.cnblogs.com/panpanwelcome/p/15739895.html)

[【Linux】自定义WSL2安装位置，安装到其他磁盘(非C盘)_wsl2指定安装路径-CSDN博客](https://blog.csdn.net/weixin_48076899/article/details/135214749#:~:text=可以右键文件夹–>属性–>常规–>高级找到并关闭这个选项 等待一段时间后安装完成，自行这是用户名及密码 安装成功后,文件夹下多一个ext4.vhdx镜像，可以理解为安装的位置 这样安装后，linux产生的文件是默认在刚刚自定义选择的路径下。 WSL1的安装位置下有个rootfs文件夹就是子系统里的全部文件。)
