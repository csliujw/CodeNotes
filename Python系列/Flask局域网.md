# 局域网访问 Flask 项目

flask 想要在局域网内可以访问需要

- 电脑开放对应的端口

- 控制台命令运行 `flask run --host=0.0.0.0 --port=80`

  - `host=0.0.0.0` 表示局域网内都可以访问
  - `port=80` 是指定连接的端口

- flask main 函数的代码请保持一致

  ```python
  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80) # host 和 port 保持一致，不要写多余的内容！
  ```

# Flask光速入门

## HelloWorld

安装最新版的 Flask `pip install -U flask`

<b>第一个 Flask 应用</b>

```python
from flask import Flask

app = Flask(__name__)

# 装饰器，将路由 / 和 hello 方法关联
@app.route('/')
def hello():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()
```

app.run 方法有三个有用的参数

- debug - 开启调试模式，代码修改后应用会重新启动
- host - 主机默认是 localhost，可以设置成 0.0.0.0 本机所有 ip
- port - 指定访问端口

## 模板渲染

Flask 提供了一种模板引擎语法，用于将 Flask 后端传递过来的内容显示在前端页面上。包含前端文件的 Flask 项目，其目录结构如下

```shell
项目名|
	---app.py 主启动类
	---static 存放静态资源的目录，如 css js img
	--templates 存放 Flask 的模板页面 .html 结尾的文件
```

在 Flask 后端中，使用 `render_template` 给页面传递参数。

```python
from flask import Flask, render_template

app = Flask(__name__)


@app.route('/test')
def test_render_template():
    # 跳转到 index.html 页面，并将参数 name=’jerry‘ 传递给前端
    return render_template('index.html', name='jerry')


if __name__ == '__main__':
    app.run(debug=True)
```

<b>模板引擎怎么取出后端的传递的数据呢？</b>

- 对于传递给页面的数据可以使用 `{{}}` 语法
- 对于项目中的静态资源，`{{url_for('static', filename='ademo.css')}}` 可以得到 static 下的 ademo.css 的路径，即 `/static/ademo.css`

## 路由参数匹配

```python
@app.route('/test/<name>')
def test_route_param(name):
    return name
```

`<name>` 是一个动态部分，可以匹配任何字符串，并将这个字符串作为参数传递给 `test_route_param` 函数中的 name。

- 注意，路由中的名称要和方法中的名称一致

## request

### 数据获取

Flask 可以从 request 域中获取三种类型的数据

| 获取方式      | 说明                                            |
| ------------- | ----------------------------------------------- |
| request.args  | 获取 URL 中的查询参数，即 GET 请求的参数        |
| request.form  | 获取 POST 请求中的表单数据                      |
| request.json  | 获取 json 数据，GET / POST 均支持 JSON 数据格式 |
| request.files | 获取前端上传的数据                              |

request.args 获取 URL 中的请求参数

```python
@app.route('/test/req/url')
def test_request():
    name = request.args.get('name')
    age = request.args.get('age')
    return {"name": name, "age": age}
# http://127.0.0.1:5000/test/req/url?name=jerry&age=50
```

request.from 获取 POST 请求中的表单数据，request.form 只用于获取 POST 请求中的表单数据

```python
@app.route('/test/req/post', methods=['POST'])
def test_request_post():
    name = request.form.get('name')
    age = request.form.get('age')
    return {"name": name, "age": age}
```

request.json 获取 json 格式的请求数据，GET / POST 均支持

[postman 中传 json格式的参数 的3种简单方式 - raw 简单明了](https://blog.csdn.net/qq_36350532/article/details/80318091)

```python
@app.route('/test/req/json', methods=['GET', 'POST'])
def test_request_json():
    name = request.json.get('name')
    age = request.json.get('age')
    return {"name": name, "age": age}
```

### HTTP消息获取

我们可以从 request 中获取 http 中的很多信息

| API                 | 说明                                                         |
| ------------------- | ------------------------------------------------------------ |
| request.headers     | 获取 headers 中的所有信息                                    |
| request.method      | 获取请求方法                                                 |
| request.origin      | 获取发起请求的原始服务器的主机名和端口号                     |
| request.path        | 获取请求的 URL 的路径部分<br>URL：http://www/x/com/jerry/to<br>path 就是 /jerry/to |
| request.referrer    | 请求的页面的 URL。这个属性是由浏览器提供的，可能不可靠       |
| request.remote_addr | 发起请求的客户端的 IP 地址                                   |





## Session-会话

从 falsk 中导入 session 即可使用，使用前需要设置一个 session-key

| API                             | 说明                                               |
| ------------------------------- | -------------------------------------------------- |
| `session['username'] = name`    | 向 session 中存储 key 为 username 值为 name 的数据 |
| `session.get('username')`       | 从 session 中获取 username 的 值                   |
| `del session['username']`       | 将 username 移除 session                           |
| `session.pop('username', None)` | 将 username 移除 session                           |

示例代码

```python
from flask import Flask, request, session

app = Flask(__name__)
app.secret_key = "asfasf13ehkjfhlkh21lhsahflkh2lkh21lkh21"

@app.route('/login/<name>')
def set_session(name):
    session['username'] = name
    return f"{name} successfully logged in"


@app.route('/logout/<name>')
def get_session(name):
    """此处复习下 参数路由"""
    session.get('username')
    # 一定要加 None
    # session.pop('username', None)
    del session['username']
    return f"successfully logged out {name}"


if __name__ == '__main__':
    app.run(debug=True)
```

## Cookie

Cookie 的用法和 Session 一致。区别请看 JavaWeb 的笔记。

## 项目拆分

项目一般会按照功能进行模块划分，我们对当前的项目进行拆分

```shell
项目名|
	--app.py 主启动类
	--App
		--static
		--templates
		--__init__.py
		--models.py
		--views.py
```

实际的项目拆分的方式是怎么样的呢？
