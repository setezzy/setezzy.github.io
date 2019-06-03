---
title: Linux grub 修复
categories: Linux
tags: 
  - Ubuntu 
  - bug 
---

# 问题描述

安装了win+linux双系统。在win下对未分配空间进行了重新分区，使磁盘数目变化，再次重启计算机之后直接进入了 Linux grub 模式，提示：

<pre><code>error: file '/grub/i386-pc/normal.mod' not found.
Entering rescue mode ...
grub rescue >
</code></pre>

# 解决方式

grub是大多Linux系统默认的引导程序，进入rescue模式说明引导文件丢失或损坏。我们首先需找到原来的系统引导文件所在的分区

<pre><code>grub rescue > ls       //查看当前磁盘分区
(hd0) (hd0,msmod10) (hd0,msmod9) (hd0,msmod8) (hd0,msmod7) (hd0,msmod6) (hd0,msmod5) (hd0,msmod3) (hd0,msmod2) (hd0,msmod1)
</code></pre>

<pre><code>grub rescue > set      //查看原先引导文件所在的分区
</code></pre>

通过set命令，可以看到有没有将/boot分区单独分出来，这与接下来找引导文件有关

若/boot是独立分区的，则输入以下命令 （若 /boot 分区不独立，用 /boot/grub  替代  /grub ）：
<pre><code>grub rescue > ls (hd0,msmod1)/grub
</code></pre>

对所有列出的分区都执行上面的命令，直到不出现 ‘error: unknown filesystem’字样，说明找到了引导文件所在分区

找到分区后就比较轻松啦。假设在 (hd0, msmod6) 找到了引导文件，然后依次执行以下命令，修改启动分区：
<pre><code>grub rescue > set root=hd0,msmod6
grub rescue > set prefix=(hd0,msmod6)/grub      //注意这里的目录
grub rescue > insmod normal
grub rescue > normal
</code></pre>

接下来出现启动菜单，此时就能正常进入ubuntu了！切记，进入系统后，还要在terminal输入以下命令更新grub引导项：
<pre><code>$ sudo update-grub
$ sudo grub-install /dev/sda     //这里的sda不是分区号，不要写成sda1之类
</code></pre>



