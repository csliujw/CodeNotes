# CLion 项目配置多个 main 函数

CLion 的项目中默认只允许有一个 main 函数。这样刷题的话不是很方便。我们可以在 cmake 中编写这样一段代码，让他可以同时拥有多个 main 函数。

```cmake
cmake_minimum_required(VERSION 3.29)
project(STL_DS)

set(CMAKE_CXX_STANDARD 14)


# 遍历项目根目录下所有的 .cpp 文件
file (GLOB_RECURSE files *.cpp)
foreach (file ${files})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
    add_executable (${exe} ${file})
    message (\ \ \ \ --\ src/${exe}.cpp\ will\ be\ compiled\ to\ bin/${exe})
endforeach ()
```

