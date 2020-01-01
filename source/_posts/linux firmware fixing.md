---
title: Ubuntu [firmware bug] 修复
categories:
  - Linux
tags:
  - bug
---

# 问题描述

Ubuntu 桌面突然花屏，没办法只好强制关机后重启，结果出现 [firmware bug] 这样的提示。可以进入 grub 菜单，但进不了系统。

# 原因

Ubuntu 内核更新，grub 菜单的默认启动项为最新的那个内核，但是新的内核貌似和现在的 CPU microcode 不兼容。

# 解决方法

先在菜单选择 ubutu 高级选项中选择原先使用的旧内核进入系统

<pre><code># 查看当前使用的内核
uname -a

# 查看系统已安装的内核
dpkg --get-selections | grep linux

# 删除多余内核
sudo apt --purge remove linux-image-4.15.0-24-generic

# 更新启动项
sudo update-grub
</code></pre>

我按上述步骤操作之后，系统启动项中还是有那个新内核。暂时没找到原因，于是我就想直接修改 grub 的默认启动项为当前使用的内核。步骤如下：

<pre><code># 查看启动菜单
grep menuentry /boot/grub/grub.cfg
</code></pre>

我的菜单项如下 (顺序依次为 0, 1, 2 ...)：
- Ubuntu
- Advanced options for Ubuntu
- Memory test
- Memory test
- Windows 10

Advanced options for Ubuntu 这个子菜单下的选项如下：
- Ubuntu, Linux 4.15.0-24-generic
- ...
- Ubuntu, Linux 4.13.0-45-generic

我想要启动的内核为 4.13.0-45，在子菜单的第 3 个选项，所以要修改 grub 默认启动项为 "1> 3"。步骤如下：

<pre><code># 不要按网上一些做法修改 /boot/grub/grub.cfg，这个文件是系统自动生成的，不可更改
sudo gedit /etc/default/grub

# 修改 GRUB_DEFAULT
GRUB_DEFAULT = "1> 3"

sudo update-grub
</code></pre>

重启系统后默认启动内核就变成了 4.13.0-45 了。