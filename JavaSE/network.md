# 网络编程

采用windows的`telent`工具作为客户端进行发起连接。

## Client和Server基本Demo

### Client

```java
/**
 * 测试服务器连接
 */
public class SocketTest {

    public static void fun1() {
        // jdk 7 try catch用法
        try (var socket = new Socket("time-a.nist.gov", 13)) {
            var scanner = new Scanner(socket.getInputStream());
            while (scanner.hasNextLine()) {
                System.out.println(scanner.nextLine() + "==");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void fun2() throws UnknownHostException, UnsupportedEncodingException {
        String host = "www.bilibili.com";
        InetAddress[] localhosts = InetAddress.getAllByName(host);
        for (InetAddress tmp : localhosts) {
            System.out.println(tmp.getHostAddress());
            System.out.println(tmp);
        }
    }

    public static void fun3() throws IOException {
        Socket socket = new Socket();
        socket.connect(new InetSocketAddress("time-a.nist.gov", 13), 10000);
        Scanner scanner = new Scanner(socket.getInputStream());
        // Scanner类不是很熟悉
        while (scanner.hasNextLine()) {
            System.out.println(scanner.nextLine());
        }
    }

    public static void main(String[] args) throws IOException {
        fun3();
    }
}
```



### Server

```java
public class EchoServer {
    /**
     * 服务器端的 inputStream 和 outPutStream
     * inPutStream 输入流，输入到Server
     * outPutStream 输出流，输出到client
     *
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(8189);
        Socket accept = serverSocket.accept();
        // 控制台读入数据
        Scanner in = new Scanner(accept.getInputStream(), StandardCharsets.UTF_8);
        // 输出 IO流还是不熟悉 类的组合太复杂了
        // PrintWriter out = new PrintWriter(new OutputStreamWriter(accept.getOutputStream(), StandardCharsets.UTF_8), true);
        OutputStreamWriter out = new OutputStreamWriter(accept.getOutputStream(), StandardCharsets.UTF_8);
        out.write("connected");
        out.flush();

        boolean done = false;
        while (!done && in.hasNextLine()) {
            // 控制台输入数据
            String line = in.nextLine();
            // 输出到客户端
            out.write("Echo:" + line);
            out.flush();
            if ("BYE".equals(line.trim())) done = true;
        }
    }
}
```



