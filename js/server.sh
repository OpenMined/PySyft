#!/bin/bash
export FLASK_APP=testing
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5001
