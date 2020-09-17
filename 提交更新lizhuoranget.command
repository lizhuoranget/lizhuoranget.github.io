#!/bin/bash
cd /Users/lizhuoran/Desktop/代码code/lizhuoranget.github.io

d1=`date +%Y-%m-%d:%H:%M:%S`
echo "The Script begin at $d1."

git pull                   #拉取服务器代码
git add -A              #提交所有变化
git commit -m "action:$d1"	#提交到本地版本库
git push                #代码推送到服务器
