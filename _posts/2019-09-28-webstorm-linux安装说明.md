\---

layout:     post

title:      webstorm-linux安装说明

subtitle:   翻译

date:       2019-09-28

author:     lizhuoran

header-img: img/post-bg-linux.jpg

catalog: true

tags:

\- Linux

\---

# webstorm-linux安装说明 翻译：

## 安装说明：

1. 将下载到的安装包解压缩到你要安装的目录位置。我们将该位置作为您的{安装位置}。

2. 打开一个控制台进入{安装位置}/bin目录下并输入：

   ./webstorm.sh

   启动应用程序。启动的同时，将初始化~/.WebStorm2019.2目录中的各种配置文件。

3. [可选项]在您的PATH环境变量中添加{安装位置}/bin，以便您可以在从任何目录启动WebStorm。

4. [可选项]要调整JVM堆大小的值，请创建～/.WebStorm2019.2/config/webstorm.vmoptions(或webstorm64.vmoptions，如果是64位JDK)，并设置-Xms和-Xmx参数。看看如何这样做，可以参考以下的vmoptions文件，以{安装位置}/bin位模版。



[可选项]更改"config"和"system"目录的位置

---

默认情况下，WebStorm将所有设置存储在~/.WebStorm2019.2/config目录并使用 ~/.WebStorm2019.2/system作为目录缓存。如果您想更改这些设置，

1. 打开一个控制台，并进入~/.WebStorm2019.2/config

2. 创建文件"idea.properties"，然后使用编辑器打开。根据需要，设置idea.system.path 和/或 idea.config.path 变量，示例，

   idea.system.path = ~/custom/system

   idea.config.path = ~/custom/config

3. 注意，我们建议将数据缓存（"system"目录）存储在至少有1GB可用空间的磁盘上。

开始使用吧！

-WebStorm开发团队

-（李卓然翻译）