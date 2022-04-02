# VSCode 插件推荐

## Python必备插件

- Python。
- Pylance 代码提示功能完美，还有形参提示。
- Eclipse Keymap（个人键位习惯）。
- Visual Studio IntelliCode 代码智能提示。
- anaconda  Extension pack
- Python Docstring Generator 文档生成。
- Better Comments 不同代码注释颜色不一样。

## Java必备插件



# VSCode&Java

按照官网下载插件

下载好后对maven进行调整

Preference-->setting–>界面模式-->修改maven的setting.xml路径，mvn.cmd路径-->

# VSCode创建普通JavaWeb的maven工程

## 创建maven工程

选择maven工程 ---> maven-archetype-webapp

groupId --> 包名

artifactId-->项目名

补齐maven工程的目录

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

pom文件中的内容

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

**运行命令**

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

快速创建一个html5模板代码

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

