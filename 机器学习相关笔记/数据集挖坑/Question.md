# 概述

记录在下载一些数据集时遇到的问题。

## CompCars数据集

<a href="http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt">下载方式</a>

注意看文中的说明，要先把所有的文件都下载了，然后用下面两个命令进行打包，解压哦！

```txt
Our dataset is provided in the following links:

Dropbox link:
https://www.dropbox.com/sh/46de2cre37fvzu6/AABXtX8QqA6sx37k1IyZmNQ2a?dl=0

Google Drive link:
https://drive.google.com/open?id=18EunmjOJsbE5Lh9zA0cZ4wKV6Um46dkg

1. Download all files with name "data.*", i.e. data.zip and data.z01 - data.z22.
2. Use this password to unzip: d89551fd190e38

On windows, both winzip and winrar are able to extract the files.
On linux, it takes two steps:
    zip -F data.zip --out combined.zip
    unzip -P <password> combined.zip

Refer to "README.txt" in the folder for descriptions of data.

Surveillance data: The surveillance data are released as "sv_data.*". Download
all such files, and unzip it with the same password.
```

