# 基本配置

- 配置 Vim 的 table 默认缩进

  ```shell
  vim ~/.vimrc
  
  在新创建的 .vimrc 配置文件中输入
  
  set number # 表示打开文件自动显示行号
  set tabstop=4 # 表示一个Tab键显示出来多少个空格的长度，默认是8，这里设置为4
  set shiftwidth=4 # 表示每一级缩进的长度，一般设置成和softtabstop长度一样
  set autoindent # 表示自动缩进
  ```

