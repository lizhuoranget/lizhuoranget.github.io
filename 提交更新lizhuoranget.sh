d=$(date "+ %Y%m%d-%H:%M:%S")
git pull
git add -A
git commit -m "${d}"
git push
