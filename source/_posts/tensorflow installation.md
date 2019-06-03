---
title: TensorFlow安装
categories: 
  - DeepLearning
  - Mix
tags: tensorflow 
---


# TensorFlow for Windows

## 环境

Win10 64位； Anaconda （python3.5及以上）

## tensorflow安装

- 以管理员身份运行Anaconda Prompt
- 使用清华镜像：
<pre><code>conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
</code></pre>
- 创建tensorflow python3.5环境：
<pre><code>conda create -n tensorflow python=3.5
</code></pre>
- 在Anaconda Prompt中激活TensorFlow环境：
<pre><code>activate tensorflow
</code></pre>
- 安装tensorflow：
<pre><code>pip install --upgrade --ignore-installed tensorflow
</code></pre>

## 测试是否安装成功

<pre><code>$ python
import tensorflow as tf
hello=tf.constant("hello TensorFlow !")
sess=tf.Session()
print(sess.run(hello))
</code></pre>

若需要在Ipython或Spyder中使用tensorflow，还需在前面建立的tensorflow环境下再安装一次。这里仅介绍一种图形界面安装方式：通过 Anaconda Navigator-Enviroments-tensorflow-not installed-选择自己所需要的插件安装即可。

注意：在IDE中使用tensorflow，要先激活tensorflow环境，再进入IDE
<pre><code>#以管理员身份运行Anaconda Prompt
(D:\Anaconda3) C:\Users\zzy> activate tensorflow
(tensorflow) C:\Users\zzy> spyder
</code></pre>

上述过程可能出现的问题：
Anaconda Navigator 打开时闪退。解决方式如下：
<pre><code># 以管理员身份运行Anaconda Prompt
conda update anaconda-navigator
anaconda-navigator --reset
conda update anaconda-client
conda update -f anaconda-client
</code></pre>


## Anaconda双版本问题

问题描述：

自己的PC上已安装了python2.7版本的Anaconda，现在需要在python3.5下使用tensorflow，故考虑到了python两个版本共存的问题。

解决方式：

  - 下载一个Anaconda3（Python3.5及以上）

  - 安装时自定义路径（必须安装在envs文件夹下，名字可自定义）：*/Anaconda/envs/py3

  - 安装tensorflow：
<pre><code>activate py3
pip install --upgrade --ignore-installed tensorflow
</code></pre>

使用时激活’py3’环境即可


# TensorFlow for Linux

## 环境

Linux自带的python2.7

## pip安装

<pre><code>sudo pip install tensorflow
</code></pre>

推荐在Anaconda下安装TensorFlow，这样可以做到与系统自带的python环境隔离。安装方式与win下类似，但将anaconda加入环境变量后，命令行查看 python version，发现解释器仅为anaconda下的python版本。