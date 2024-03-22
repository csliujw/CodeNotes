# 使用公网服务器搭建 ssh 转发

- 确保内网服务器和公网服务器都安装了 openssh-server，并且公网服务器和内网服务器都要开放对应的端口。

- 此处的设定

  - 内网服务器 close：安装了 openssh-server，开放了 22 端口，用户名为 user
  - 公网服务器 open：
    - 安装了 openssh-server，开放了 22、8022、8023 端口，用户名为 root；
    - /etc/ssh/sshd_config 中配置 `GatewayPorts yes`
  - 其他电脑 other

  安装软件

  ```bash
  apt install openssh-server
  service ssh restart
  
  apt install autossh
  
  # ssh 连接
  ssh root@xxx -p 8022
  # autossh 连接，8999端口监听8022端口，8022断了就重连
  autossh -M 8999 root@xxx -p 8022
  ```

  注意：远程端口转发需要设置 `/etc/ssh/sshd_config` 中 `GatewayPorts yes` 。默认情况下，OpenSSH 只允许从服务器主机连接到远程转发端口。能够防止从服务器计算机外部连接到转发端口。设置成 ‘yes’ 后，外部主机就可以连接了。

  ## 步骤

  ### 内网服务器建立方向代理

  内网服务器的 22 端口和公网服务器的 8022 端口建立 ssh 连接，也可以安装 autossh 使用 autossh 建立连接。

  ```bash
  ssh -fCNR 8022:localhost:22 -o ServerAliveInterval=60 root@19.168.100.4  -p 22 # 这里 root 就是 公网服务器的用户名
  # ssh -fCNR [公网IP(可省略)]:[公网端口]:[内网IP]:[内网端口] [公网用户名@公网IP] -p [公网ssh端口(默认22)]
  # 或者使用autossh:
  # autossh -M 8999 -CNR 8022:localhost:22 root@19.168.100.4 -p 22
  ```

  该命令实现的功能是，让一个远端机器（这里是指公网服务器）的 8022 端口代理自己（这里是指内网服务器）的 22 端口。

  执行 ps aux | grep ssh 可查看是否成功启动了该进程

  ### 公网服务器

  查看内网服务器是否成功和公网服务器建立了连接

  ```shell
  # t 表示 time？ p 的话表示 pid
  netstat -antpul | grep 8022
  
  tcp		0	0 127.0.0.1:8022	0.0.0.0:*	LISTEN	2094/sshd: root
  tcp6	0	0 ::1:8022				:::*			LISTEN	2094/sshd: root
  ```

  这样就可以使用公网服务器访问内网服务器了。

  ```shell
  # localhost 是写死的！ -p 端口号 一定要加！
  # 我们是通过本机（远程服务器）的8022端口和内网服务器建立通信的
  ssh user@localhost -p 8022 
  ```

  使用 ssh 向内网服务器发送文件，发送到内网服务器的 /home/code 目录下

  ```shell
  scp -p 8022 filename user@localhost:/home/code/
  ```

  此时的连接方式

  ```mermaid
  graph LR
  公网服务器-->|连接|内网服务器
  ```

  我们期望的是用公网服务器做中间人，让任意的电脑都可以连接到内网服务器，这时候就需要公网服务器做正向代理了。

  ```mermaid
  graph LR
  任意的服务器-->公网服务器-->内网服务器
  ```

  ### 公网服务器正向代理

  我们让公网服务器的 8023 代理 8022，这里我们连接公网服务器的 8023 端口，这个请求会被代理到 8022 端口（这个端口正好是和内网服务器进行通讯的）实测不用做正向代理，直接连接服务器的 8022 端口，ssh 连接就会被转发到内网的服务器上。

  ```shell
  # ssh -fCNL [本机IP(可省略)]:[本机端口]:[远端IP]:[远端端口] [远端用户名@远端IP] -p [远端ssh端口(默认22)]
  # * 表示接受来自任意 ip 的访问
  ssh -fCNL *:8023:localhost:8022 -o ServerAliveInterval=60 root@localhost -p 22
  # 或者使用autossh:
  # autossh -M 8999 -CNL *:8023:localhost:8022 root@localhost -p 22
  ```