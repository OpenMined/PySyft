
export FLASK_APP=grid
export GRID_HOST=$1
export GRID_PORT=$2

flask run --host $1 --port $2