cd app
rm -rf .git
git init
git add .
git commit -am "init"
heroku create $GRID_NAME
heroku addons:create rediscloud
git push heroku master
rm -rf .git
cd ../