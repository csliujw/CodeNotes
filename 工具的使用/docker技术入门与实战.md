# Docker

## Docker三大要素

- 镜像：
  - 类似于虚拟机镜像，可将其理解为面向Docker引擎的只读模版，包含了文件系统。
- 容器：
  - 类似于轻量级的沙箱，Docker利用容器来运行和隔离应用。
  - 容器是从镜像创建的应用运行实例。
  - 一个镜像可创建出多个容器，容器直接相互隔离。
- 仓库：存放镜像文件的场所。

Docker利用仓库管理镜像的设计理念与Git非常相似。

## 安装Docker

```shell
# ubuntu自带的源
sudo apt-get update
sudo apt-get install -y docker.io

# 从docker官网中安装最新的。
# 去docker官网看怎么安装就行了，不用记忆。
```

<a href="https://docs.docker.com/engine/install/ubuntu/">docker官网</a>

**PS：**`Docker`的命令 按tab键会自动补全的

# Docker镜像

## 获取镜像

```shell
sudo docker pull ubuntu # 默认拉取最新的镜像
sudo docker pull ubuntu:14.04 #拉取指定版本
sudo docker pull dl.dockerpool.com:5000/ubuntu # 从其他注册服务器的仓库下载
sudo docker run -t -i ubuntu /bin/bash # 利用镜像创建一个容器，在其中运行bash应用
```

## 查看镜像

```shell
sudo docker images # 查看本地主机中已有的docker镜像
"""
REPOSITORY   TAG      IMAGE ID        CREATED       SIZE
ubuntu       latest   d70eaf7277ea    4 weeks ago   72.9MB
"""
```

- REPOSITORY：来自于那个仓库
- TAG：镜像的标签信息，来自统一仓库的不同镜像
- IMAGE ID：镜像的id，唯一！
- CREATED：创建时间
- SIZE：大小

**为已有镜像添加标签**

```shell
sudo docker tag ubuntu:latest ubuntuxxx
# sudo docker tag 已有的镜像 新的镜像标签
# 新的镜像标签和已有镜像的标签是一致的~~ 可以理解为一个快捷方式
```

**查看镜像的详细信息**

```shell
sudo docker inspect image_id
sudo docker inspect -f 查看指定参数，不会，用的时候再查
```

## 搜寻镜像

`sudo docker search 镜像名称`

```shell
sudo docker search mysql
sudo docker search mysql --filter=stars=3 # 查询stars大于3k的镜像
```

**几个重要的参数**

- --automated=false 仅显示自动创建的镜像
- --no-trunc=false 输出信息不截断显示

## 删除镜像

`sudo docker rmi IMAGE` IMAGE可以是标签（tag）或ID

```shell
sudo docker rmi dl.asf.com/ubuntu:latest
# sudo docker rmi respository:tag
```

正确步骤：

- 先删除依赖该镜像的容器 `sudo docker rm 容器ID`
- 然后删除镜像

## 创建镜像

###  基于已有容器创建

```shell
sudo docker run -ti ubuntu:latest /bin/bash
touch test
exit

# 记住容器的id
sudo docker commit -m "some message" -a "author message" container_id test_image
```

### 基于本地模版导入

## 存入存出镜像

```shell
# 存出
sudo docker save -o xxx.tar ubuntu:latest
# 载入
sudo docker load --input xxx.tar
```

# Docker容器

## 创建容器

```shell
# 这种方式创建的不会启动。
sudo docker create -it imageName
sudo docker create -it unbunt:latest
# 启动create创建的容器
sudo docker  start container_id
```

## 容器的各种操作

- 下面的创建方式，输出一句Hello后容器就终止了
  - `sudo docker run ubuntu:latest /bin/echo 'Hello'`
- 启动一个终端，并允许用户进行交互
  - `sudo docker run -t -i ubuntu:latest /bin/bash`
    - -t 分配一个伪终端
    - -i 让容器的标准输入保持打开

- 查看运行中容器
  - `sudo docker ps`
- 查看所有的容器
  - 运行中的和终止了的
  - `sudo docker ps -a`
- 容器以守护态运行
  - 守护态的运行取决于后面的shell语句。
  - -d，此时容器会在后台运行并不会把输出的结果 (STDOUT) 打印到宿主机上面(输出结果可以用 `docker logs` 查看)。
    - `sudo docker run -d ubuntu /bin/sh -c "while true; do echo hello world; sleep 1; done"`
  - 查看容器的输出信息
    - `sudo docker logs 8c`
  - 停止容器
    - `sudo docker stop 8c`
  - 启动终止态的容器
    - 最开始命令用了-it 启动就关闭了，但是start的时候，会在后台一直执行。
    - `sudo docker start container_id`
  - 重启一个容器
    - `sudo docker restart container_id`
- **进入容器**
  - 常用命令
    - attach：多个窗口同时attach到同一个容器的时候，所有窗口都会同步显示，一个窗口被阻塞了，其他的窗口也无法执行操作。
    - exec：`sudo docker exec -ti container_id /bin/bash`
    - nsenter：不学
- **导入导出容器：**（给容器装一些软件，然后导出一直用）
  - 导出容器
    - `sudo docker export container_id  > xxx.tar`
  - 导入容器
    - `cat x.tar | sudo docker import - tt/ubuntu:v1`
    - `cat x.tar | sudo docker import - reporsitory:tag `

# Docker仓库

暂时用不到

# Docker数据管理

暂时用不到

# 网络基础配置

映射容器端口到宿主主机和容器互联机制来为容器提供网络服务。

**就是：容器端口 ---> 宿主主机**

## 端口映射实现访问容器



##  容器互联实现容器通信