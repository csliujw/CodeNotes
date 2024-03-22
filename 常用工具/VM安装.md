安装 VM 时遇到了一些问题，在此记录。

- 使用 vm 安装 centos 时报错

  - 您的主机不满足在启用 Hyper-V 或 Device/Credential Guard 的情况下运行 VMware Workstation 的最低要求

- 解决方案
  - 关闭 Hyper -v
    - 控制面板 -- 程序功能 -- 启用或关闭 Windows 功能 -- 取消勾选 Hyper-V 整个大项
    - 管理员运行 cmd 执行指令 bcdedit /set hypervisorlaunchtype off

- <a href="https://blog.csdn.net/dling8/article/details/106809555">参考博客</a>

centos 默认是不开启网卡的，需要我们进行下设置。

在 root 用户下 `vi /etc/sysconfig/network-scripts/ifcfg-ens33`

onboot=no 改为 onboot=yes

service network restart 保存设置。

最后发现还是不行。是我的 vm 没有设置对！

<a href="https://blog.csdn.net/ghjzzhg/article/details/70260805?locationNum=8&fps=1">请看该博客</a>

VM 的虚拟网络编辑器桥接模式设置为自动！

具体步骤

- 编辑 --> 虚拟网络编辑器 -->更改设置-->任意添加一个网络
- VMnet信息 选择桥接模式 --> 自动桥接

- OK！