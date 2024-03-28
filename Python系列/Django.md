# Django

[Django 介绍 - 学习 Web 开发 | MDN (mozilla.org)](https://developer.mozilla.org/zh-CN/docs/Learn/Server-side/Django/Introduction)

[Django 模型 | 菜鸟教程 (runoob.com)](https://www.runoob.com/django/django-model.html)

Django 模板引擎不学

## 跨域问题

Django csrf 工作的顺序是，先从后台获取 csrf_token 发送给前端，然后前端进行 form 提交时，把名字 csrfmiddlewaretoken，值为 csrf_token 的字段发给后端校验

Django 都是在 settings.py 里进行配置的

- 注释掉 MIDDLEWARE 中的 `django.middleware.csrf.CsrfViewMiddleware`，不安全

- 使用 `from django.views.decorators.csrf import csrf_exempt` 装饰器

