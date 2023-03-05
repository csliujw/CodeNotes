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
