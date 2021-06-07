class BaseConfig:
    TESTING = False
    DEBUG = False


class DevConfig(BaseConfig):
    FLASK_ENV = "development"
    DEBUG = True


class ProductionConfig(BaseConfig):
    FLASK_ENV = "production"
    SQLALCHEMY_DATABASE_URI = (
        "postgresql://db_user:db_password@db-postgres:5432/flask-deploy"
    )


class TestConfig(BaseConfig):
    FLASK_ENV = "development"
    TESTING = True
    DEBUG = True
