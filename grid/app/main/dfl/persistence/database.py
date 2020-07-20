"""This file provides a singleton db instance."""
import redis

redis_db = None


def set_db_instance(database_url):
    global redis_db
    redis_db = redis.from_url(database_url)
    return redis_db


def db_instance():
    global redis_db
    return redis_db
