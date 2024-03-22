# VSCode 插件推荐

## Python必备插件

- Python。
- Pylance 代码提示功能完美，还有形参提示。
- Eclipse Keymap（个人键位习惯）。
- Visual Studio IntelliCode 代码智能提示。
- anaconda  Extension pack
- Python Docstring Generator 文档生成。
- Better Comments 不同代码注释颜色不一样。

# VSCode&Java

按照官网下载插件

下载好后对 maven 进行调整

Preference-->setting–>界面模式-->修改 maven 的 setting.xml 路径，mvn.cmd 路径

# VSCode创建普通JavaWeb的maven工程

## 创建maven工程

选择 maven 工程 ---> maven-archetype-webapp

groupId --> 包名

artifactId-->项目名

补齐 maven 工程的目录

src

|----maven

​	  |----java

​	  |----resources

​	  |----webapp

​			|----WEB-INF

|----test

​	|----java

​	|----resources

## 补充pom文件

pom 文件中的内容

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>cn.bbxx</groupId>
  <artifactId>CoreJava</artifactId>
  <version>1.0-SNAPSHOT</version>
  <packaging>war</packaging>

  <name>CoreJava Maven Webapp</name>
  <!-- FIXME change it to the project's website -->
  <url>http://www.example.com</url>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>1.7</maven.compiler.source>
    <maven.compiler.target>1.7</maven.compiler.target>
  </properties>

  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.11</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>javax.servlet</groupId>
      <artifactId>servlet-api</artifactId>
      <version>2.5</version>
      <scope>provided</scope> <!-- 为了防止和tomcat中的api冲突，设置为provided可见 -->
    </dependency>
  </dependencies>

  <build>
    <finalName>CoreJava</finalName>
    <pluginManagement>
      <plugins>
        <plugin>
          <artifactId>maven-clean-plugin</artifactId>
          <version>3.1.0</version>
        </plugin>
        <plugin>
          <artifactId>maven-resources-plugin</artifactId>
          <version>3.0.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-compiler-plugin</artifactId>
          <version>3.8.0</version>
        </plugin>
        <plugin>
          <artifactId>maven-surefire-plugin</artifactId>
          <version>2.22.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-war-plugin</artifactId>
          <version>3.2.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-install-plugin</artifactId>
          <version>2.5.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-deploy-plugin</artifactId>
          <version>2.8.2</version>
        </plugin>

        <!-- 设置tomcat版本！ -->
        <plugin>
          <groupId>org.apache.tomcat.maven</groupId>
          <artifactId>tomcat7-maven-plugin</artifactId>
          <version>2.2</version>
          <configuration>
            <port>8080</port>
          </configuration>
        </plugin>
      </plugins>
    </pluginManagement>
  </build>
</project>
```

运行命令

```powershell
mvn tomcat7:run
```

# VSCode查看源码不打开新窗口

```json
{
    "java.configuration.updateBuildConfiguration": "automatic",
    "workbench.editor.enablePreview": false
}
```

# VSCode HTML快捷键

## doc

快速创建一个 html5 模板代码

```html
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>

<body>

</body>

</html>
```

# 搭建在线版本VSCode

## 配置

- github 下载 code-server 的 linux 版本。[Releases · coder/code-server (github.com)](https://github.com/coder/code-server/releases)

- 修改目录名 `mv code-server-3.2.0-linux-x86_64 code-server`（好看点）

- cd code-server

- 运行 ./coder-server --help 查看有那些可以配置的参数

    ```shell
    Usage: code-server [options] [path]
    
    Options
         --auth                The type of authentication to use. [password, none]
         --cert                Path to certificate. Generated if no path is provided.
         --cert-key            Path to certificate key when using non-generated cert.
         --disable-updates     Disable automatic updates.
         --disable-telemetry   Disable telemetry.
      -h --help                Show this output.
         --open                Open in browser on startup. Does not work remotely.
         --bind-addr           Address to bind to in host:port.
         --socket              Path to a socket (bind-addr will be ignored).
      -v --version             Display version information.
         --user-data-dir       Path to the user data directory.
         --extensions-dir      Path to the extensions directory.
         --list-extensions     List installed VS Code extensions.
         --force               Avoid prompts when installing VS Code extensions.
         --install-extension   Install or update a VS Code extension by id or vsix.
         --uninstall-extension Uninstall a VS Code extension by id.
         --show-versions       Show VS Code extension versions.
         --proxy-domain        Domain used for proxying ports.
    -vvv --verbose             Enable verbose logging.
    
    ```

- 常用运行方式

    ```shell
    export PASSWORD="xxxx"
    ./code-server --port 9999 --host 0.0.0.0 --auth password
    ```

    - –port 9999 指定端口，缺省时为 8080
    - –host 0.0.0.0 允许公网访问，缺省时为 127.0.0.1，只能本地访问
    - –auth password 指定访问密码，可通过 export 命令设置，参数为 none 时不启用密码

- 关闭

    - 查询 PID `ps aux | grep ./code-server`
    - kill -9 对应的 PID

## shell脚本运行关闭

启动脚本

```shell
#start.sh
export PASSWORD="xxxx"
nohup ./code-server --port 9999 --host 0.0.0.0 --auth password > test.log 2>&1 &
echo $! > save_pid.txt
```

关闭脚本

```shell
#shut.sh
kill -9 $(cat save_pid.txt) # $(cat save_pid.txt) 将命令的执行结果作为 kill -9 命令的参数 
```

