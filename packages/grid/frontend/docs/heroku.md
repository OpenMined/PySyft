# Deployment to Heroku

Deploying PyGrid Admmin to Heroku is easy. Before continuing, make sure you have
the following already set up

- A Heroku account ( If you don't have one, create one
  [here](https://signup.heroku.com) )
- Heroku's CLI installed ( If you don't have it yet, head over
  [this](https://devcenter.heroku.com/articles/heroku-cli) page )

## Clone PyGrid Admin

First, clone PyGrid Admin

```bash
git clone git@github.com:OpenMined/PyGrid.git
cd pygrid-admin
```

## Create a new heroku app

Via Heroku CLI we can create a new app. You might aswell do this via Heroku's
UI.

```bash
heroku create $APP_NAME
```

## Push the Code to Heroku via Heroku CLI

Now you can deploy PyGrid Admin to Heroku by simply pushing the code living in
master branch.

```bash
git remote add heroku https://git.heroku.com/[APP_NAME].git
git push heroku master
```

Also, you could deploy PyGrid Admin for a specific branch as follows

```bash
git push heroku [your branch name]:master
```

## Access PyGrid Admin

Once deployed, simply access Pygrid Admin via your favourite web browser at

```
https://[APP_NAME].herokuapp.com/
```
