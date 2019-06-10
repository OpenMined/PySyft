from grid.app.models import Worker
from grid.app.models import WorkerObject
from grid.app.config import db
from grid.app.config import app
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_migrate import Migrate

TEST_DB_URI = "sqlite:///:memory:"

import torch

import syft as sy
from grid.workers import WebsocketIOServerWorker

hook = sy.TorchHook(torch)


class WorkerPersistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        app.config["TESTING"] = True
        app.config["DEBUG"] = False
        app.config["SQLALCHEMY_DATABASE_URI"] = TEST_DB_URI
        self.engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
        Worker.metadata.drop_all(self.engine)
        db.create_all()

    def tearDown(self):
        Worker.metadata.drop_all(self.engine)

    def test_worker_persistance(self):
        w = Worker(public_id="test")
        db.session.add(w)
        db.session.commit()
        w2 = Worker.query.filter_by(public_id="test").first()
        assert w2.id == w.id

    def test_tensor_persistance(self):
        w = Worker(public_id="test_persistance")
        t = WorkerObject(worker=w, object=torch.ones(10))
        db.session.add_all([w, t])
        db.session.commit()
        assert w.worker_objects[0].object.shape == t.object.shape
        assert w.worker_objects[0].object.sum() == t.object.sum()
