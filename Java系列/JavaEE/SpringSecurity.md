# 权限校验框架

- BCryptPasswordEncoder，对同一个字符串进行加密，加密的结果都是不一样的。但是 matches 方法可以和原密码正确匹配。