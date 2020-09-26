from .. import db


class GridNodes(db.Model):
    """Grid Nodes table that represents connected grid nodes.

    Columns:
        id (primary_key) : node id, used to recover stored grid nodes (UNIQUE).
        address: Address of grid node.
    """

    __tablename__ = "gridnode"

    id = db.Column(db.String(255), primary_key=True)
    address = db.Column(db.String(255))

    def __str__(self):
        return f"< Grid Node {self.id} : {self.address}>"
