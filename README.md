# Docker简介

Docker是一个开源的应用容器引擎，基于Go语言并遵循Apache2.0协议开源。

Docker可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到任何流行的Linux机器上，也可实现虚拟化。

Docker容器使用的是沙箱机制，相互之间不会有任何接口，更重要的是容器性能开销极低。

大概意思是：Docker可以集成很多软件，然后把软件弄成镜像，让使用者可以直接用镜像，无需再安装软件。

运行中的这个镜像成为容器，容器的启动很迅速。

# Docker核心概念

Docker主机：安装了Docker程序的机器。Docker是直接安装在OS上的。

Docker客户端：连接Docker主机进行操作。

Docker仓库：用来保存各种打包好的软件镜像。

Docker镜像：软件打包好的镜像，放在仓库中。

Docker容器：镜像启动后的实例，称为一个容器。tomcat镜像运行五次，就会有五个容器，就有5个tomcat了。

使用Docker的步骤：

- 安装docker
- 去docker仓库找到软件对应的镜像
- 直接使用docker运行这个镜像，这个镜像就会生成一个docker容器
- 停止容器就是停止软件。一个容器对应一个软件。

# 安装Docker

## 安装虚拟机

- 安装虚拟机VMA 或 VirtualBox[免费哦]

- 导入虚拟机文件。导入时记得选中重新初始化所有网卡的MAC地址

  - 导入的那个虚拟机：账户 root 密码 123456

- 使用客户端连接linux服务器。挑一个自己熟悉的就行。

- 设置虚拟机网络

  - 右键点击设置网络
  - 连接方式：桥接
  - 界面名称：根据网络的类型选择有线网络/无线网络 接入网线。就好了。
  - 然后 输入命令 `service network restart` 【centos 7】

- 查看Linux的IP地址

  - ```shell
    ip addr 找到 inet 的地址
    ```

## 安装Docker

- `docker`要求内核版本高于3.10
  - 查看内核版本 `uname -r`
  - 不是的话，用 `yum update`更新
- 安装： `yum install docker`
- 启动：`systemctl start docker`
  - `docker -v` 查看docker版本号
  - 这里我出现了问题，看了这篇<a href="https://blog.csdn.net/E09620126/article/details/86577917?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.channel_param">博客</a>解决了问题
- 停止：`systemctl stop docker`
- 将`docker`设置为开机自启：`systemctl enable docker`
- 停止`docker`：`systemctl stop docker`

## 安装软件

- 搜索mysql镜像：`docker search mysql`
  - 其实就是去docker hub仓库里搜索mysql镜像
- 拉取mysql镜像：`docker pull mysql`不加标签，默认用最新的。
  - 指定标签 `docker pull mysql:5.5`
- 查看所有镜像：`docker images` 
- 删除镜像：`docker rmi image-id`【传镜像id】 [rmi= remove image]

## Docker容器操作

- 得到软件镜像====运行镜像====产生一个容器
- 示例
  - 搜索镜像：`docker search tomcat`
  - 拉取镜像：`docker pull tomcat`
  - 查看镜像：`docker images`
  - 运行镜像：`docker run --name container-name -d image-name`
    - eg：`docker run --name myredis -d redis.`
    - --name 自定义容器名
    - -d 后台运行
    - image-name 指定镜像模板
  - 查看运行的镜像：`docker ps`
  - 停止当前容器：`docker stop TAINER ID 或 容器名称`
  - 启动容器：`docker run container-id`
  - 删除一个容器：`docker rm container-id`
- 我们想访问docker中的tomcat是访问不了的。我们访问虚拟机里的8080，我们访问不到docker里的8080.我们要做一个映射。
- 停掉，重新安装tomcat
  - 停止容器：`docker stop xsafsf[TAINER ID]`

- 端口映射：`docker run -d -p 8888:8080 --name mytomcat tomcat` [把主机的8888映射到容器的8080]  可以简写`docker run -d -p 8888:8080 tomcat`，会自动为我们取名
  - -p：把主机端口映射到容器内部的端口  主机端口：容器内部端口
  - -d：后台启动。
- 我们需要关闭Linux的防火墙，才可以访问！
  - 查看防火墙状态：`service status firewalld.service`
  - 关闭防火墙：`service firewalld stop`
- 查看容器日志：`docker logs`
- 更多请看官网文档<a href="https://hub.docker.com/">地址</a>

# Docker安装mysql

- docker search mysql
- docker pull mysql
- dokcer run --name mysql01 -d mysql 发现运行错误
  - 查看日志，看错误原因 【docker logs 对应容器的id】
  - 如何正确运行？去官网看哦！
  - docker run --name mysql01 -e MYSQL_ROOT_PASSWORD=123456 -d mysql
  - docker run -p 3306:3306 --name mysql01 -e MYSQL_ROOT_PASSWORD=123456 -d mysql