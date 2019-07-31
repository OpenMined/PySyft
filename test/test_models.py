from app.pg_rest_api.pg_app import create_app
from app.pg_rest_api.lib.models import Worker, WorkerObject
from app.pg_rest_api.lib.models import db
from .flask_base_test import FlaskBaseTestCase
import torch


class ModelTestCase(FlaskBaseTestCase):
    """
    Testing that our models are successfully persisting
    """

    def test_worker_persistence(self):
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
