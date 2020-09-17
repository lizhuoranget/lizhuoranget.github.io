#!/bin/bash
cd /Users/lizhuoran/Desktop/代码code/lizhuoranget.github.io

msg=date+“%Y-%m-%d:%H:%M:%S”

git pull                   #拉取服务器代码
git add -A              #提交所有变化
git commit -m action:${msg}	#提交到本地版本库
git push                #代码推送到服务器
