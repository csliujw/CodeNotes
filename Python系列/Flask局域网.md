flask 想要在局域网内可以访问需要：

- 电脑开放对应的端口

- 控制台命令运行 `flask run --host=0.0.0.0 --port=80`

  - `host=0.0.0.0` 表示局域网内都可以访问
  - `port=80` 是指定连接的端口

- flask main 函数的代码请抱持一致

  ```python
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80) # host 和 port 保存一致，不要写多余的内容！
  ```
