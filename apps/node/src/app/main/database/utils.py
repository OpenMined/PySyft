from ... import db
from .group import Group
from .role import Role
from .usergroup import UserGroup


def model_to_json(model):
    """Returns a JSON representation of an SQLAlchemy-backed object."""
    json = {}
    for col in model.__mapper__.attrs.keys():
        if col != "hashed_password" and col != "salt":
            json[col] = getattr(model, col)

    return json


def expand_user_object(user):
    def get_group(user_group):
        query = db.session().query
        group = user_group.group
        group = query(Group).get(group)
        group = model_to_json(group)
        return group

    query = db.session().query
    user = model_to_json(user)
    user["role"] = query(Role).get(user["role"])
    user["role"] = model_to_json(user["role"])
    user["groups"] = query(UserGroup).filter_by(user=user["id"]).all()
    user["groups"] = [get_group(user_group) for user_group in user["groups"]]

    return user
