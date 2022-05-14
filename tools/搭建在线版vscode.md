# 配置

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

# shell脚本运行关闭

## 启动脚本

```shell
#start.sh
export PASSWORD="xxxx"
nohup ./code-server --port 9999 --host 0.0.0.0 --auth password > test.log 2>&1 &
echo $! > save_pid.txt
```

## 关闭脚本

```shell
#shut.sh
kill -9 $(cat save_pid.txt) # $(cat save_pid.txt) 将命令的执行结果作为 kill -9 命令的参数 
```

