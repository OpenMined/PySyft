import pytest
import unittest
import sys
from random import randint
from datetime import datetime


from flask import Flask

sys.path.append("./gateway/app/")


from flask_sqlalchemy import SQLAlchemy
from main.storage import models

app = Flask(__name__)

BIG_INT = 2 ** 32


class TestDatabase(unittest.TestCase):
    def setUp(self):
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"
        self.db = models.db
        self.db.init_app(app)
        app.app_context().push()
        self.db.create_all()

    @pytest.mark.skip
    def testCreatePlan(self):
        my_plan = models.Plan(
            id=randint(0, BIG_INT),
            value="list of plan values",
            value_ts="torchscript(plan)",
        )
        self.db.session.add(my_plan)
        self.db.session.commit()

    @pytest.mark.skip
    def testCreateProtocol(self):
        my_protocol = models.Protocol(
            id=randint(0, BIG_INT),
            value="list of protocol values",
            value_ts="torchscript(protocol)",
        )
        self.db.session.add(my_protocol)
        self.db.session.commit()

    @pytest.mark.skip
    def testCreateConfig(self):
        client_config = {
            "name": "my-federated-model",
            "version": "0.1.0",
            "batch_size": 32,
            "lr": 0.01,
            "optimizer": "SGD",
        }

        server_config = {
            "max_workers": 100,
            "pool_selection": "random",  # or "iterate"
            "num_cycles": 5,
            "do_not_reuse_workers_until_cycle": 4,
            "cycle_length": 8 * 60 * 60,  # 8 hours
            "minimum_upload_speed": 2000,  # 2 mbps
            "minimum_download_speed": 4000,  # 4 mbps
        }

        my_server_config = models.Config(id=randint(0, BIG_INT), config=server_config)

        my_client_config = models.Config(id=randint(0, BIG_INT), config=client_config)

        self.db.session.add(my_server_config)
        self.db.session.add(my_client_config)
        self.db.session.commit()

    @pytest.mark.skip
    def testCreateWorker(self):
        worker = models.Worker(
            id=randint(0, BIG_INT),
            format_preference="list",
            ping=randint(0, 100),
            avg_download=randint(0, 100),
            avg_upload=randint(0, 100),
        )

        self.db.session.add(worker)
        self.db.session.commit()

    @pytest.mark.skip
    def testCreateCycle(self):
        new_cycle = models.Cycle(
            id=randint(0, BIG_INT),
            start=datetime(2019, 2, 21, 7, 29, 32, 45),
            sequence=randint(0, 100),
            end=datetime(2019, 2, 22, 7, 29, 32, 45),
        )

        self.db.session.add(new_cycle)
        self.db.session.commit()

    @pytest.mark.skip
    def testCreateModel(self):
        new_model = models.Model(version="0.0.1")

        self.db.session.add(new_model)
        self.db.session.commit()

    @pytest.mark.skip
    def testCreateFLProcess(self):
        new_fl_process = models.FLProcess(id=randint(0, BIG_INT),)

        self.db.session.add(new_fl_process)

        new_model = models.Model(version="0.0.1", flprocess=new_fl_process)

        self.db.session.add(new_model)

        avg_plan = models.Plan(
            id=randint(0, BIG_INT),
            value="list of plan values",
            value_ts="torchscript(plan)",
            avg_flprocess=new_fl_process,
        )

        self.db.session.add(avg_plan)

        training_plan = models.Plan(
            id=randint(0, BIG_INT),
            value="list of plan values",
            value_ts="torchscript(plan)",
            plan_flprocess=new_fl_process,
        )

        self.db.session.add(training_plan)

        validation_plan = models.Plan(
            id=randint(0, BIG_INT),
            value="list of plan values",
            value_ts="torchscript(plan)",
            plan_flprocess=new_fl_process,
        )

        self.db.session.add(validation_plan)

        protocol_1 = models.Protocol(
            id=randint(0, BIG_INT),
            value="list of protocol values",
            value_ts="torchscript(protocol)",
            protocol_flprocess=new_fl_process,
        )

        self.db.session.add(protocol_1)

        protocol_2 = models.Protocol(
            id=randint(0, BIG_INT),
            value="list of protocol values",
            value_ts="torchscript(protocol)",
            protocol_flprocess=new_fl_process,
        )

        self.db.session.add(protocol_2)

        client_config = {
            "name": "my-federated-model",
            "version": "0.1.0",
            "batch_size": 32,
            "lr": 0.01,
            "optimizer": "SGD",
        }

        server_config = {
            "max_workers": 100,
            "pool_selection": "random",  # or "iterate"
            "num_cycles": 5,
            "do_not_reuse_workers_until_cycle": 4,
            "cycle_length": 8 * 60 * 60,  # 8 hours
            "minimum_upload_speed": 2000,  # 2 mbps
            "minimum_download_speed": 4000,  # 4 mbps
        }

        server_config = models.Config(
            id=randint(0, BIG_INT),
            config=server_config,
            server_flprocess_config=new_fl_process,
        )

        client_config = models.Config(
            id=randint(0, BIG_INT),
            config=client_config,
            client_flprocess_config=new_fl_process,
        )

        self.db.session.add(client_config)
        self.db.session.add(server_config)

        cycle_1 = models.Cycle(
            id=randint(0, BIG_INT),
            start=datetime(2019, 2, 21, 7, 29, 32, 45),
            sequence=randint(0, 100),
            end=datetime(2019, 2, 22, 7, 29, 32, 45),
            cycle_flprocess=new_fl_process,
        )

        cycle_2 = models.Cycle(
            id=randint(0, BIG_INT),
            start=datetime(2019, 2, 27, 15, 19, 22),
            sequence=randint(0, 100),
            end=datetime(2019, 2, 28, 15, 19, 22),
            cycle_flprocess=new_fl_process,
        )

        self.db.session.add(cycle_1)
        self.db.session.add(cycle_2)
        self.db.session.commit()

    @pytest.mark.skip
    def testWorkerCycle(self):
        new_fl_process = models.FLProcess(id=randint(0, BIG_INT))

        self.db.session.add(new_fl_process)

        new_model = models.Model(version="0.0.1", flprocess=new_fl_process)

        self.db.session.add(new_model)

        avg_plan = models.Plan(
            id=randint(0, BIG_INT),
            value="list of plan values",
            value_ts="torchscript(plan)",
            avg_flprocess=new_fl_process,
        )

        self.db.session.add(avg_plan)

        training_plan = models.Plan(
            id=randint(0, BIG_INT),
            value="list of plan values",
            value_ts="torchscript(plan)",
            plan_flprocess=new_fl_process,
        )

        self.db.session.add(training_plan)

        validation_plan = models.Plan(
            id=randint(0, BIG_INT),
            value="list of plan values",
            value_ts="torchscript(plan)",
            plan_flprocess=new_fl_process,
        )

        self.db.session.add(validation_plan)

        protocol = models.Protocol(
            id=randint(0, BIG_INT),
            value="list of protocol values",
            value_ts="torchscript(protocol)",
            protocol_flprocess=new_fl_process,
        )

        self.db.session.add(protocol)

        client_config = {
            "name": "my-federated-model",
            "version": "0.1.0",
            "batch_size": 32,
            "lr": 0.01,
            "optimizer": "SGD",
        }

        server_config = {
            "max_workers": 100,
            "pool_selection": "random",  # or "iterate"
            "num_cycles": 5,
            "do_not_reuse_workers_until_cycle": 4,
            "cycle_length": 8 * 60 * 60,  # 8 hours
            "minimum_upload_speed": 2000,  # 2 mbps
            "minimum_download_speed": 4000,  # 4 mbps
        }

        server_config = models.Config(
            id=randint(0, BIG_INT),
            config=server_config,
            server_flprocess_config=new_fl_process,
        )

        client_config = models.Config(
            id=randint(0, BIG_INT),
            config=client_config,
            client_flprocess_config=new_fl_process,
        )

        self.db.session.add(client_config)
        self.db.session.add(server_config)

        cycle = models.Cycle(
            id=randint(0, BIG_INT),
            start=datetime(2019, 2, 21, 7, 29, 32, 45),
            sequence=randint(0, 100),
            end=datetime(2019, 2, 22, 7, 29, 32, 45),
            cycle_flprocess=new_fl_process,
        )

        self.db.session.add(cycle)

        worker = models.Worker(
            id=randint(0, BIG_INT),
            format_preference="list",
            ping=randint(0, 100),
            avg_download=randint(0, 100),
            avg_upload=randint(0, 100),
        )

        self.db.session.add(worker)

        worker_cycle = models.WorkerCycle(
            id=randint(0, BIG_INT),
            request_key="long_hashcode_here",
            worker=worker,
            cycle=cycle,
        )

        self.db.session.add(worker_cycle)
        self.db.session.commit()
