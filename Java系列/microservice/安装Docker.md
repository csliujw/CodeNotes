# 安装Docker

Docker 分为 CE 和 EE 两大版本。CE 即社区版（免费，支持周期 7 个月），EE 即企业版，强调安全，付费使用，支持周期 24 个月。

Docker CE 分为 `stable` `test` 和 `nightly` 三个更新频道。

官方网站上有各种环境下的[安装指南](https://docs.docker.com/install/)，这里主要介绍 Docker CE 在 CentOS上的安装。

# CentOS安装Docker

Docker CE 支持 64 位版本 CentOS 7，并且要求内核版本不低于 3.10， CentOS 7 满足最低内核的要求，所以我们在 CentOS 7 安装 Docker。

## 卸载（可选）

如果之前安装过旧版本的 Docker，可以使用下面命令卸载：

```bash
yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-selinux \
                  docker-engine-selinux \
                  docker-engine \
                  docker-ce
```

## 安装docker

虚拟机联网，安装 yum 工具

```sh
yum install -y yum-utils \
           device-mapper-persistent-data \
           lvm2 --skip-broken
```

更新本地镜像源

```shell
# 设置docker镜像源
yum-config-manager \
    --add-repo \
    https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
    
sed -i 's/download.docker.com/mirrors.aliyun.com\/docker-ce/g' /etc/yum.repos.d/docker-ce.repo

yum makecache fast
```

然后输入命令

```shell
yum install -y docker-ce
```

docker-ce 为社区免费版本。

## 启动docker

Docker 应用需要用到各种端口，逐一去修改防火墙设置。非常麻烦，因此建议直接关闭防火墙！

启动 docker 前，一定要关闭防火墙后！！

启动 docker 前，一定要关闭防火墙后！！

启动 docker 前，一定要关闭防火墙后！！

```sh
# 关闭
systemctl stop firewalld
# 禁止开机启动防火墙
systemctl disable firewalld
```

通过命令启动 docker

```sh
systemctl start docker  # 启动docker服务

systemctl stop docker  # 停止docker服务

systemctl restart docker  # 重启docker服务
```

输入命令，可以查看 docker 版本

```shell
docker -v

wsl@cv:/mnt/c/Users/69546$ docker -v
Docker version 20.10.12, build e91ed57
```

## 配置镜像加速

docker 官方镜像仓库网速较差，需要设置国内镜像服务

参考阿里云的镜像加速文档：https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors

# CentOS安装DockerCompose

## 下载

Linux 下需要通过命令下载：

```sh
# 安装
curl -L https://github.com/docker/compose/releases/download/1.23.1/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose
```

## 修改文件权限

修改文件权限

```sh
# 修改权限
chmod +x /usr/local/bin/docker-compose
```

## Base自动补全命令

```sh
# 补全命令
curl -L https://raw.githubusercontent.com/docker/compose/1.29.1/contrib/completion/bash/docker-compose > /etc/bash_completion.d/docker-compose
```

如果这里出现错误，需要修改自己的 hosts 文件

```sh
echo "199.232.68.133 raw.githubusercontent.com" >> /etc/hosts
```

# Docker镜像仓库

搭建镜像仓库可以基于 Docker 官方提供的 DockerRegistry 来实现。

官网地址：https://hub.docker.com/_/registry

## 简化版镜像仓库

Docker 官方的 Docker Registry 是一个基础版本的 Docker 镜像仓库，具备仓库管理的完整功能，但是没有图形化界面。

搭建方式比较简单，命令如下：

```sh
docker run -d \
    --restart=always \
    --name registry	\
    -p 5000:5000 \
    -v registry-data:/var/lib/registry \
    registry
```

命令中挂载了一个数据卷 registry-data 到容器内的 /var/lib/registry 目录，这是私有镜像库存放数据的目录。

访问 http://YourIp:5000/v2/_catalog 可以查看当前私有镜像服务中包含的镜像

## 带有图形化界面版本

使用 DockerCompose 部署带有图象界面的 DockerRegistry，命令如下

```yaml
version: '3.0'
services:
  registry:
    image: registry
    volumes:
      - ./registry-data:/var/lib/registry
  ui:
    image: joxit/docker-registry-ui:static
    ports:
      - 8080:80
    environment:
      - REGISTRY_TITLE=传智教育私有仓库
      - REGISTRY_URL=http://registry:5000
    depends_on:
      - registry
```

## 配置Docker信任地址

我们的私服采用的是 http 协议，默认不被 Docker 信任，所以需要做一个配置

```sh
# 打开要修改的文件
vi /etc/docker/daemon.json
# 添加内容：
"insecure-registries":["http://192.168.150.101:8080"]
# 重加载
systemctl daemon-reload
# 重启docker
systemctl restart docker
```

# Ubuntun补充

- sudo ufw disable 关闭防火墙
- sudo ufw status 查看防火墙状态
- sudo ufw enable 开启防火墙

Ubuntu 启动 Docker

```shell
su root
systemctl enable docker # 设置开机自动启用 docker 服务
systemctl start docker # #启动 docker 服务
```





