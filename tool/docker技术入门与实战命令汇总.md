- 安装docker
  - `sudo apt-get update
    sudo apt-get install -y docker.io`
- 查看容器
  - `sudo docker ps` 查看运行中的容器
  - `sudo docker ps -a` 查看停止的容器
  - `sudo docker ps -a -q` 查看处于终止状态的容器id信息

# 镜像的操作

- **获取镜像**
  - `sudo docker pull ubuntu # 默认拉取最新的镜像`
  - `sudo docker pull ubuntu:14.04 #拉取指定版本`
  - `sudo docker pull xxx/ubuntu # 从其他注册服务器的仓库下载`

- **查看镜像**
  - `sudo docker images_names`
- 为已有镜像添加标签
  - `sudo docker tag ubuntu:latest ubuntuxxx`  <span style="color:green">ubuntu:latest 是已有的标签</span>

- 查看镜像的详细信息
  - `sudo docker inspect image_id`
  - `sudo docker inspect -f 查看指定参数` 具体自行百度
- **搜寻镜像**
  - ``sudo docker search 镜像名称``
  - `sudo docker search mysql`
  - `sudo docker search mysql --filter=stars=3 `
  - 几个重要的参数
    - --automated=false 仅显示自动创建的镜像
    - --no-trunc=false 输出信息不截断显示

- **删除镜像**
  - `sudo docker rmi IMAGE` 
  - IMAGE可以是标签（tag）或ID
  - `sudo docker rmi dl.asf.com/ubuntu:latest`
  - 删除镜像的正确步骤
    - 先删除依赖该镜像的容器 `sudo docker rm 容器ID`
    - 然后删除镜像`sudo docker rmi IMAGE_ID`

- 创建镜像
  - 基于已有容器创建 这个容器要存在！！！
  - `sudo docker commit -m "some message" -a "author message" container_id test_image`
- **存入存出镜像**
  - `sudo docker save -o xxx.tar ubuntu:latest`
  - `sudo docker load --input xxx.tar`

#  容器的操作

- 新建容器
  - 这种方式创建的不会启动。
  - `sudo docker create -it imageName`
  - `sudo docker create -it unbunt:latest`
  - 启动create创建的容器
  - `sudo docker  start container_id`
- 新建容器并启动
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

