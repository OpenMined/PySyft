# grid relative
# syft relative
from sqlalchemy import Boolean, Column, Integer, String, ForeignKey


def create_user_table(db, model):
    class User(model):
        __tablename__ = "user"

        id = Column(Integer(), primary_key=True, autoincrement=True)
        email = Column(String(255))
        hashed_password = Column(String(512))
        salt = Column(String(255))
        private_key = Column(String(2048))
        verify_key = Column(String(2048))
        role = Column(Integer, ForeignKey("role.id"))

        def __str__(self):
            return f"<User id: {self.id}, email: {self.email}, " f"role: {self.role}>"

    return User

def create_user(email, hashed_password, salt, private_key, role):
    new_user = User(
        email=email,
        hashed_password=hashed_password,
        salt=salt,
        private_key=private_key,
        role=role,
    )
    return new_user
