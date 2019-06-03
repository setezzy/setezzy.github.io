---
title: 在 Hexo 中渲染 Latex 公式
categories:
  - Others
tags:
  - bug
mathjax: false
---

这篇记录一下我在文章里增加 latex 公式时被自己作死的过程。

之前因为懒，没有在博客里手打公式，上一篇博客公式较多，所以还是需要用到 latex。我欢欢喜喜地打了一堆公式然后 `hexo g` ，然而报错了。Hexo 的 markdown 竟然不支持 latex 语法！看看人家 Jupyter Notebook 就很完美啊！随之我的折腾之路就开始了。

我先是按照网上大多教程写的，安装了 `hexo-math`这个插件，和latex一样，行内公式用 `$$`，行间公式用 `$$ $$`。但还是报以下错误：
<pre><code>Template render error: (unknown path) [line xx,  column xx]
  expected variable end
</code></pre>

这种情况一般是文章里出现很多大括号，和 hexo 自带渲染引擎的语法冲突。我就开始奇怪，明明支持 latex 语法了怎么还有 bug。然后在 google 上搜了一通，尝试其他插件然后改配置文件都试过，最后反倒出现了其他的 bug，比如 `Error Deployer not found: git`，`Error Local hexo not found`，心态一度爆炸。

最后我的一系列操作：

- 将 Node.js 更新到了较新的稳定版本，删掉 node-modules 这个文件夹后重新 `npm install` （参考[这里](https://github.com/hexojs/hexo/issues/2076)）

- 更新其他 modules，例如 hexo-deployer-git （参考[这里](https://github.com/hexojs/hexo/issues/2757)）
- hexo3 已经支持 mathjax，所以根本不用这么麻烦，只需要在 `_config.yml` 中将 mathjax enable 一下，再更改默认的渲染引擎就好了。[这篇博客](https://www.jianshu.com/p/e8d433a2c5b7)的步骤很详细，也不会踩坑。
- 其实最重要的一步，就是检查 Markdown 里的公式本身有没有多打一个括号少一个括号。大括号太多也可能在 generate 这一步就报错。我就是疏忽了这个问题，最后发现是自己坑了自己，浪费了大笔时间。

最后公式是渲染成功了，但是还存在一些问题。比如在 `$$ $$` 标记的公式下面插入图片时显示不出来，而且`<center></center>`命令也会让图片显示不出来。

总之我的经验教训是，像这种 bug 一定要冷静对待，不然头脑一凌乱就要重头开始安装和配置。然后是多用 google ，百度emmm就算了。