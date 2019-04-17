cd app;
heroku apps:destroy --confirm $GRID_NAME
cd ../

sh ../scripts/create_and_launch_heroku_app.sh

