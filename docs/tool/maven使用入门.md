要重新学下maven搭建工程了

# Maven的安装与配置

## 前置条件
- JDK
- IDE
- Maven安装包
```
安装包自行前往https://maven.apache.org/download.cgi下载最新版的Maven程序。
```
## 1.开始安装
- 将文件解压
- 新建MAVEN_HOME,环境变量，赋值为mavenbin所在的父级目录
- 注意maven需要依赖于JAVA_HOME这个环境变量，环境变量的key是固定的。
  - 如maven的目录是 D:\Program Files\Apache\maven;
  - MAVEN_HOME赋值为 D:\Program Files\Apache\maven;
  - 在系统环境变量path中追加 %MAVEN_HOME%\bin\;
## 2.检查安装结果

- mvn -v 查看Maven信息

  如果显示的jdk是你按照的本地jdk的目录，则完美成功！

## 3.配置Maven本地仓库
- 在任意目录新建maven-repository文件夹【名称随意】
- 打开maven/conf/settings.xml文件修改仓库默认位置
```xml
<!--1.查找下面这行代码：-->
<localRepository>/path/to/local/repo</localRepository>

<!--2.localRepository节点默认是被注释掉的,把它移到注释之外,将localRepository节点的值改为我们在创建的目录-->
D:\Program Files\Apache\maven-repository。

3.localRepository节点用于配置本地仓库，本地仓库其实起到了一个缓存的作用，它的默认地址是 C:\Users\用户名.m2。
当我们从maven中获取jar包的时候，maven首先会在本地仓库中查找，如果本地仓库有则返回；如果没有则从远程仓库中获取包，并在本地库中保存。
```

## 4.Maven的常用命令

> **maven的命令，一般高级别的命令包含低级别的命令，maven运行tomcat项目使用的是 ==mvn tomcat:run==**

以下命令都需要进入对于项目的根目录进行。如spring为xx项目的根目录【其内部包含main】。

### mvn clean

清理项目生产的临时文件,一般是模块下的target目录。

### mvn package

项目打包工具,会在模块下的target目录生成jar或war等文件

### mvn test

测试命令,或执行src/test/java/下junit的测试用例.

### mvn install 

聚合工程时如果需要应用其他模块的包，需要先将该模块打包，这样才可以正常使用其他模块的代码？

模块安装命令,将打包的的jar/war文件复制到你的本地仓库中,供其他模块使用.

### mvn deploy 

发布命令,将打包的文件发布到远程参考,提供其他人员进行下载依赖,一般是发布到公司的私服.

### mvn  spring-boot:run

mvn启动springboot项目

### mvn compile

mvn编译java代码

## maven常见错误
-   运行maven命令时报错： 
    No compiler is provided in this environment. Perhaps you are running on a JRE rather than a JDK?
```
在maven bin下的mvn.cmd配置文件第一行输入
set JAVA_HOME="E:\program\environment\Java\jdk1.8.0_144"
让maven可以找到jdk
我不是配置了全部java环境吗? 为什么这里还要写?
在mvn.cmd配置文件中有这么一句话
@REM JAVA_HOME Must point at your Java Development Kit installation.
```
-   maven启动springBoot项目时，不要先mvc compilet，直接mvn spring-boot:run
## maven日常用法积累

- maven把三方jar包弄到自己的本地仓库中
```
E:\program\tool\apache-maven-3.6.1\maven-repository\office>
mvn install:install-file  -Dfile=KGDoc.Office.jar   -DgroupId=office  -DartifactId=office  -Dversion=2.0 -Dpackaging=jar

-Dfile=你要导入的jar包名称
-DgroupId=你给他设置的groupId
-DartifactId=你给他设置的dartifactId，应该是jar包名称，(项目标识名)
-Dversion=你给他设置的版本
-Dpackaging=你给他设置的类型
```