git pull
git add .
git commit -m action:%date:~0,4%%date:~5,2%%date:~8,2%-%time%
git push origin master
git --version