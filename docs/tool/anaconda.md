# 直奔主题

```shell
# 查看安装了那些软件
conda list 
# 查看anaconda信息
conda info
# 查看有哪些环境
conda info -e
# 创建新环境，用清华源创建环境总失败
conda create -n 环境名称 python=指定python版本
# 卸载环境中的某些包
conda uninstall pytorch=1.4.0 卸载指定版本的软件

# 现在torch官网安装pytorch的方式快的可怕，用啥清华源。没必要了。好吧，我错了，也就python3.8的安装快。

# 换源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --set show_channel_urls yes

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

# 然后安装pytorch
conda install pytorch==1.5.0 torchvision==0.6.0 cpuonly

# 移除源
conda config --remove-key channels

# 清华源安装opencv-python 也可以用豆瓣源啥的。
pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# Anaconda

```shell
conda create -n test python=3.5
```

- conda：用于指定接下来使用的命令集来自Anaconda。
- create：表示我们的具体操作是要创建一个新环境。
- -n：用于指定新环境的名称，新环境的名称必须紧跟在空格之后
- test：用于指定新环境的名称为test。
- python=3.5：用于指定在新环境下需要预先安装的Python版本，这里使用的Python版本为3.5。

```shell
activate test # 激活环境

deactivate test # 退出当前环境

conda remove -n test -all # 移除环境

conda search numpy # 搜索平台中指定名称的Python包

conda install numpy # 安装包

conda install anaconda # 查看已经安装的包
```

# Jupyter  Notebook

**常用快捷键**

- Enter：进入编辑模式。
- Esc：退出编辑模式，进入命令模式。
- Shift-Enter：运行当前选中的输入单元中的内容，并选中下一个输入单元。
- Ctrl-Enter：仅运行当前选中的输入单元中的内容。
- Alt-Enter：运行当前选中的输入单元中的内容，并在选中的输入单元之后插入新的输入单元。
- 1：将输入单元的内容编辑模式设置为Markdown模式，并在输入单元中的内容的开始处添加一级标题对应的井号个数和一个空格字符。
- 2：将输入单元的内容编辑模式设置为Markdown模式，并在输入单元中的内容的开始处添加二级标题对应的井号个数和一个空格字符。
- 3：将输入单元的内容编辑模式设置为Markdown模式，并在输入单元中的内容的开始处添加三级标题对应的井号个数和一个空格字符。
- 4：将输入单元的内容编辑模式设置为Markdown模式，并在输入单元中的内容的开始处添加四级标题对应的井号个数和一个空格字符。
- 5：将输入单元的内容编辑模式设置为Markdown模式，并在输入单元中的内容的开始处添加五级标题对应的井号个数和一个空格字符。
- 6：将输入单元的内容编辑模式设置为Markdown模式，并在输入单元中的内容的开始处添加六级标题对应的井号个数和一个空格字符
- Y：将输入单元的内容编辑模式设置为Code模式。
- M：将输入单元的内容编辑模式设置为Markdown模式。
- A：在当前的输入单元的上方插入新的输入单元。
- B：在当前的输入单元的下方插入新的输入单元。
- D：删除当前选中的输入单元。
- X：剪切当前选中的输入单元。
- C：复制当前选中的输入单元。
- Shift-V：将复制或者剪切的输入单元粘贴到选定的输入单元上方。
- V：将复制或者剪切的输入单元粘贴到选定的输入单元下方。
- Z：恢复删除的最后一个输入单元。
- S：保存当前正在编辑的Notebook文件。
- L：在Notebook的所有输入单元前显示行号。当处于编辑模式时，我们常用的快捷键如下。
  - Enter：进入编辑模式。
  - Esc：退出编辑模式，进入命令模式。
  - Tab：如果输入单元的内容编辑模式为Code模式，则可以通过该快捷键对不完整的代码进行补全或缩进。
  - Shift-Tab：如果输入单元的内容编辑模式为Code模式，则可以通过该快捷键显示被选取的代码的相关提示信息。
  - Shift-Enter：运行当前选中的输入单元中的内容并选中下一个输入单元，在运行后会退出编辑模式并进入命令模式。
  - Ctrl-Enter：仅运行当前选中的输入单元中的内容，在运行后会退出编辑模式并进入命令模式。
  - Alt-Enter：运行当前选中的输入单元中的内容，并在选中的输入单元之后插入新的输入单元，在运行后会退出编辑模式并进入命令模式。
  - PageUp：将光标上移到输入单元的内容前面。
  - PageDown：将光标下移到输入单元的内容后面。