# JWT

> 什么是 JWT

JWT = Json Web Token，是一个开放标准(rfc7519)，它定义了一种紧凑的、自包含的方式，用于在各方之间以JSON对象安全地传输信息。此信息可以验证和信任，因为它是数字签名的。JWT 可以使用秘密（使用 HNAC 算法）或使用 RSA 或 ECDSA 的公钥/私钥对进行签名。

JWT 通过 JSON 形式作为 Web 应用中的令牌，用于在各方之间安全地将信息作为 JSON 对象传输。在数据传输过程中还可以完成数据加密、签名等相关处理。

通过数字签名的方式，以 JSON 对象为载体，在不同的服务端之间安全的传输信息。而这个 token 存储在客户端中，服务器中则保存 JWT 签名的密文。

> JWT 的作用

- 授权：一旦用户登录，每个后续请求将包括 JWT，系统在每次处理用户请求之前，都要进行 JWT 安全校验，通过之后再进行处理。JWT 的开销很小并且可以在不同的域中使用。如：单点登录。
- 信息交换：在各方之间安全地传输信息。JWT 可进行签名（如使用公钥/私钥对)，因此可确保发件人。由于签名是使用标头和有效负载计算的，因此还可验证内容是否被篡改。

> JWT 的组成

JWT 就是令牌 token，是一个 String 字符串，由三部分组成，用 `.` 拼接。令牌的组成如下：

- 标头（Header）
- 有效载荷（Payload）
- 签名（Signature）

token 格式：`head.payload.singurater` 如：`xxxxx.yyyy.zzzz`

- Header，有令牌的类型和所使用的签名算法，如 HMAC、SHA256、RSA；使用 Base64 编码组成；（Base64 是一种编码，不是一种加密过程，可以被翻译成原来的样子）

```json
{
    'type': 'JWT',
    'alg': 'HS256'
}
```

- Payload，有效负载，包含声明；声明是有关实体（通常是用户）和其他数据的声明，不放用户敏感的信息，如密码。同样使用 Base64 编码。

```json
{
    'sub': '12345678',
    'name': 'john',
    'admin': true
}
```

- signature，前面两部分都使用 Base64 进行编码，前端可以解开知道里面的信息。Signature 需要使用编码后的 header 和 payload；加上我们提供的一个密钥，使用 header 中指定的签名算法(HS256)进行签名。签名的作用是保证 JWT 没有被篡改过。

```js
let encodedString = base64UrlEncode(header) + '.' + base64UrlEncode(payload); // 用 base64 进行编码
let signature = HMACSHA256(encodedString,'secret'); // 编码后再进行加密
```

<b>签名目的：</b>签名的过程实际上是对头部以及负载内容进行签名，防止内容被窜改。如果有人对头部以及负载的内容解码之后进行修改，再进行编码，最后加上之前的签名组合形成新的 JWT 的话，那么服务器端会判断出新的头部和负载形成的签名和 JWT 附带上的签名是不一样的。如果要对新的头部和负载进行签名，在不知道服务器加密时用的密钥的话，得出来的签名也是不一样的。

信息安全问题：Base64 是一种编码，是可逆的，适合传递一些非敏感信息；JWT 中不应该在负载中加入敏感的数据。如传输用户的ID被知道也是安全的，如密码不能放在 JWT 中；JWT 常用于设计用户认证、授权系统、web 的单点登录。

# JWT 示意代码

引入依赖

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <artifactId>JWT</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>11</maven.compiler.source>
        <maven.compiler.target>11</maven.compiler.target>
    </properties>

    <dependencies>
        <dependency>
            <groupId>com.auth0</groupId>
            <artifactId>java-jwt</artifactId>
            <version>3.10.0</version>
        </dependency>
    </dependencies>

</project>
```

JWT Java 代码示意

```java
package com.jwt.demo;

import com.auth0.jwt.JWT;
import com.auth0.jwt.JWTVerifier;
import com.auth0.jwt.algorithms.Algorithm;
import com.auth0.jwt.interfaces.DecodedJWT;

import java.util.Base64;
import java.util.Date;
import java.util.HashMap;

public class GenerateToken {
    private static String sign = "12sfasarweddad";

    public static void main(String[] args) {
        HashMap<String, Object> map = new HashMap<>();
        map.put("type", "jwt");
        map.put("alg", "HS256");
        String token = JWT.create()
                .withHeader(map) //
                .withClaim("userId", 10086)
                .withClaim("age", 12)
                .withExpiresAt(new Date(System.currentTimeMillis() + 60 * 60 * 24))
                .sign(Algorithm.HMAC256(sign));
        System.out.println(token);

        JWTVerifier build = JWT.require(Algorithm.HMAC256(sign)).build();
        DecodedJWT verify = build.verify(token);
        System.out.println(verify.getClaim("userId").asInt()); // 10086
        System.out.println(verify.getClaim("age").asInt()); // 12

        // 获取 Payload 的 base64 编码后的数据并解码
        byte[] decode = Base64.getDecoder().decode(verify.getPayload());
        // 将解码后的数据重新解析成字符串
        System.out.println(new String(decode)); // {"exp":1664526266,"userId":10086,"age":12}
    }
}
```

JWT 常见异常信息

- SignatureVerificationException -- 签名不一致异常
- TokenExpiredException -- 令牌过期异常
- AlgorithmMismatchException -- 算法不匹配异常
- InvalidaClaimException -- 失效的 payload 异常（传给客户端后，token 被改动，验证不一致）

# JWT 集成 SpringBoot

[JWT详细教程与使用_一支有理想的月月鸟的博客-CSDN博客_jwt教程](https://blog.csdn.net/Top_L398/article/details/109361680)