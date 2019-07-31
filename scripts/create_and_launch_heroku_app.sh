
function rest_api(){
    echo "App type : Rest API "
   cd ../app/rest_api/

}

function websocket(){
     echo "App type : websocket "
      cd ../app/websocket/

}
function createInstance(){

    echo "App name : ${1}"
    echo "Redis cloud name : ${2}"
    echo "Create new Redis instance ? : ${3}"
    app_name="$1"
    redis_name="$2"
    rm -rf .git
    git init
    git add .
    git commit -am "init"
    heroku create ${app_name}
    heroku git:remote -a ${app_name}

    if [ "$3" = "true" ]
    then
        echo "Creating redis cloud addon with name:  ${redis_name}"
       heroku addons:create rediscloud --name ${redis_name}
    else
        echo "Attaching redis cloud addon with name : ${redis_name}"
        heroku addons:attach ${redis_name} -a ${app_name}
    fi
    git push heroku master
    rm -rf .git
}




print_usage() {
  printf "\n\nUsage: ...\n\n"
  printf " ./create_and_launch_heroku_app.sh -type rest_api  -appName testApp -redisName testRedis -create"
  printf "\n\n"
}



while true $# -gt 0; do
  case "$1" in
    -type)
        shift
        if [ "$1" = "rest_api" ]
        then
            rest_api
        elif [ "$1" = "websocket" ]
        then
            websocket
        fi
        shift
        ;;
    -appName)
        shift
        app_name=$1
        shift
        ;;
    -redisName)
        shift
        redis_name=$1
        create_redis_cloud="true"
        shift
        ;;
    -linkRedis)
        shift
        redis_name=$1
        create_redis_cloud="false"
        shift
        ;;
    -create)
        shift
        if [ ${#app_name} -gt 0 ] && [ ${#redis_name} -gt 0 ]
        then
            echo "creating Instance"
            createInstance $app_name $redis_name $create_redis_cloud

        fi
        shift
        exit 1
        ;;
    *) print_usage
       exit 1 ;;
  esac
done