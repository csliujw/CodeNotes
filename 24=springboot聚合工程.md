# 搭建过程

**从我聚合maven的情况，看到导入其他子项目的类，发现import的包直接是包名！所以，maven的聚合是把不同文件夹的搞到一起！最后不同的只是他们的包名！所以，springboot的配置文件的扫描！可以扫到！但是如果是测试的话，可能就不好测试了。我目前只能在web包下进行测试！明天问杰妹！**

- 创建一个springboot的父项目，留下pom和必要的配置信息即可。

  - pom中的

    ```xml
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
    ```

    删除。不删除的话，打包过程会有问题。因为加了这个，他打包必须要有一个main函数（springboot的那个）

- 然后分别创建各个子项目
  - blog-pojo
  - blog-dao
  - blog-service
  - blog-web
- 然后运行，成功了！之前却一直不成功。。。淦！

----

附上pom文件

父工程的

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <packaging>pom</packaging>
    <modules>
        <module>blog-dao</module>
        <module>blog-service</module>
        <module>blog-pojo</module>
        <module>blog-web</module>
    </modules>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.2.6.RELEASE</version>
        <relativePath/>
    </parent>

    <groupId>com.bbxx</groupId>
    <artifactId>blog</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>blog</name>
    <description>Demo project for Spring Boot</description>

    <properties>
        <java.version>1.8</java.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-jdbc</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <dependency>
            <groupId>org.mybatis.spring.boot</groupId>
            <artifactId>mybatis-spring-boot-starter</artifactId>
            <version>2.1.2</version>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>

        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
            <exclusions>
                <exclusion>
                    <groupId>org.junit.vintage</groupId>
                    <artifactId>junit-vintage-engine</artifactId>
                </exclusion>
            </exclusions>
        </dependency>
    </dependencies>

</project>
```

dao的

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <parent>
        <artifactId>blog</artifactId>
        <groupId>com.bbxx</groupId>
        <version>0.0.1-SNAPSHOT</version>
    </parent>

    <modelVersion>4.0.0</modelVersion>

    <artifactId>blog-dao</artifactId>

</project>
```

service的

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <parent>
        <artifactId>blog</artifactId>
        <groupId>com.bbxx</groupId>
        <version>0.0.1-SNAPSHOT</version>
    </parent>
    <modelVersion>4.0.0</modelVersion>

    <artifactId>blog-service</artifactId>

    <dependencies>
        <dependency>
            <groupId>com.bbxx</groupId>
            <artifactId>blog-dao</artifactId>
            <version>0.0.1-SNAPSHOT</version>
        </dependency>
    </dependencies>
</project>
```



web的。web中书写yml配置文件哦！

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <parent>
        <artifactId>blog</artifactId>
        <groupId>com.bbxx</groupId>
        <version>0.0.1-SNAPSHOT</version>
    </parent>
    <modelVersion>4.0.0</modelVersion>

    <artifactId>blog-web</artifactId>

    <dependencies>
        <dependency>
            <groupId>com.bbxx</groupId>
            <artifactId>blog-service</artifactId>
            <version>0.0.1-SNAPSHOT</version>
        </dependency>
    </dependencies>

</project>
```

运行spring-boot的manve聚合项目时，可能会遇到

```java
ERROR] Failed to execute goal org.springframework.boot:spring-boot-maven-plugin:1.5.7.RELEASE:run (default-cli) on project springboot-integration: Unable to find a suitable main class, please add a 'mainClass' property -> [Help 1]
[ERROR]
[ERROR] To see the full stack trace of the errors, re-run Maven with the -e switch.     
[ERROR] Re-run Maven using the -X switch to enable full debug logging.
[ERROR]
[ERROR] For more information about the errors and possible solutions, please read the following articles:
```

这种情况，我们进入聚合工程那个的web启动子模块里就行。

在子模块里运行 `mvn spring-boot:run`

```powershell
PS E:\Code\VSCode_Java\springboot-integration>


    目录: E:\Code\VSCode_Java\springboot-integration


Mode                LastWriteTime         Length Name
----                -------------         ------ ----
d-----        2020/4/23     11:16                .settings
d-----        2020/4/30     16:28                .vscode
d-----        2020/4/30     16:38                mm-entity
d-----        2020/4/30     16:38                mm-repo
d-----        2020/4/30     16:38                mm-service
d-----        2020/4/30     16:38                mm-web
-a----        2020/4/11     22:17            402 .project
-a----        2020/4/30     16:37           1760 pom.xml


PS E:\Code\VSCode_Java\springboot-integration> cd .\mm-web\
PS E:\Code\VSCode_Java\springboot-integration\mm-web> mvn spring-boot:run
```

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# maven聚合工程笔记

## 项目结构搭建

![](E:\69546\Documents\image\maven聚合工程.png)

## pom文件书写

**pojo层pom书写**

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <packaging>jar</packaging>
    <parent>
        <artifactId>blog</artifactId>
        <groupId>com.bbxx</groupId>
        <version>0.0.1-SNAPSHOT</version>
    </parent>
    <modelVersion>4.0.0</modelVersion>

    <artifactId>blog-pojo</artifactId>

</project>
```

**dao层pom文件书写**

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <packaging>jar</packaging>
    <parent>
        <artifactId>blog</artifactId>
        <groupId>com.bbxx</groupId>
        <version>0.0.1-SNAPSHOT</version>
    </parent>

    <modelVersion>4.0.0</modelVersion>

    <artifactId>blog-dao</artifactId>

</project>
```

**service层pom文件书写**

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <packaging>jar</packaging>
    <parent>
        <artifactId>blog</artifactId>
        <groupId>com.bbxx</groupId>
        <version>0.0.1-SNAPSHOT</version>
    </parent>
    <modelVersion>4.0.0</modelVersion>

    <artifactId>blog-service</artifactId>

    <dependencies>
        <dependency>
            <groupId>com.bbxx</groupId>
            <artifactId>blog-dao</artifactId>
            <version>0.0.1-SNAPSHOT</version>
        </dependency>
    </dependencies>
</project>
```

**web层pom文件书写【important】**

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <parent>
        <artifactId>blog</artifactId>
        <groupId>com.bbxx</groupId>
        <version>0.0.1-SNAPSHOT</version>
    </parent>
    <packaging>jar</packaging>
    <modelVersion>4.0.0</modelVersion>

    <artifactId>blog-web</artifactId>

    <dependencies>
        <dependency>
            <groupId>com.bbxx</groupId>
            <artifactId>blog-service</artifactId>
            <version>0.0.1-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <!-- 这些插件是保证可以正确打包springboot项目的。因为开始我是创建了一个springboot项目作为父目录。然后创建的其他普通maven项目+springbootmaven项目。这种书写方式需要指定启动类，否会无法在根目录用maven启动项目！且打包的时候，会提示确实主启动类！ -->
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <configuration>
                    <source>1.8</source>
                    <target>1.8</target>
                </configuration>
            </plugin>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <version>2.0.3.RELEASE</version>
                <configuration>
                    <mainClass>com.bbxx.blog.WebApplication</mainClass>
                </configuration>
                <executions>
                    <execution>
                        <goals>
                            <goal>repackage</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin>
        </plugins>
    </build>

</project>
```

**顶级父项目的pom文件**

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <packaging>pom</packaging>
    <modules>
        <module>blog-dao</module>
        <module>blog-service</module>
        <module>blog-pojo</module>
        <module>blog-web</module>
    </modules>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.2.6.RELEASE</version>
        <relativePath />
    </parent>

    <groupId>com.bbxx</groupId>
    <artifactId>blog</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>blog</name>
    <description>Demo project for Spring Boot</description>

    <properties>
        <java.version>1.8</java.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-jdbc</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <dependency>
            <groupId>org.mybatis.spring.boot</groupId>
            <artifactId>mybatis-spring-boot-starter</artifactId>
            <version>2.1.2</version>
        </dependency>


        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-devtools</artifactId>
            <scope>runtime</scope>
            <optional>true</optional>
        </dependency>

        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
            <exclusions>
                <exclusion>
                    <groupId>org.junit.vintage</groupId>
                    <artifactId>junit-vintage-engine</artifactId>
                </exclusion>
            </exclusions>
        </dependency>
    </dependencies>
</project>
```

## 总结！

我启动`sb`项目。无法在根目录启动。只能进入对应的web进行启动`mvn spring-boot:run`

之后想对聚合工程进行打包，发现要么报错，要么打包后的jar无法执行，提示没有主class。后来在web子模块中加了上面pom中的两个插件，问题解决了。本质上就是没有指定类【】。plus中指定了启动类！

