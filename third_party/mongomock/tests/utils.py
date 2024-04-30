class DBRef(object):
    def __init__(self, collection, id, database=None):
        self.collection = collection
        self.id = id
        self.database = database

    def as_doc(self):
        doc = {"$ref": self.collection, "$id": self.id}
        if self.database is not None:
            doc["$db"] = self.database
        return doc
