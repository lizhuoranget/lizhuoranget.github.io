#!/bin/bash

#提交代码到github

msg='action:'
msg=${msg}$1

cd /Users/lizhuoran/Desktop/代码code/lizhuoranget.github.io
git pull                   #拉取服务器代码
git add -A              #提交所有变化
git commit -m ${msg}	#提交到本地版本库
git push                #代码推送到服务器
