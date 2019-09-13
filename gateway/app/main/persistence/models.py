from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class GridNodes(db.Model):
    """ Grid Nodes table that represents connected grid nodes.
    
        Collumns:
            id (primary_key) : node id, used to recover stored grid nodes (UNIQUE).
            address: Address of grid node.
    """

    __tablename__ = "__gridnode__"

    id = db.Column(db.String(64), primary_key=True)
    address = db.Column(db.String(64))

    def __str__(self):
        return f"< Grid Node {self.id} : {self.address}>"
