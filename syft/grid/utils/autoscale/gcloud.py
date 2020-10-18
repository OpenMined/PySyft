"""To autoscale pygrid workers on Google Cloud Platfrom"""
import json
import IPython
import terrascript
import terrascript.data
import terrascript.provider
import terrascript.resource

# syft dependencies
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient

# terraform utils
from utils.script import terraform_script
from utils.notebook import terraform_notebook


class GoogleCloud:
    """This class defines the spinning up of Google Cloud Resources"""

    def __init__(self, credentials, project_id, region):
        """
        args:
            credentials: Path to the credentials json file
            project_id: project_id of your project in GCP
            region: region of your GCP project
        """
        self.credentials = credentials
        self.project_id = project_id
        self.region = region
        self.provider = "google"
        self.config = terrascript.Terrascript()
        self.config += terrascript.provider.google(
            credentials=self.credentials,
            project=self.project_id,
            region=self.region,
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if IPython.get_ipython():
            terraform_notebook.init()
        else:
            terraform_script.init()

    def expose_port(self, name, ports=None, apply=True):
        """
        args:
            name: name of the resource
            ports: list of ports to be exposed, defaults to 80
            apply: to call terraform apply at the end
        """
        if ports is not None:
            ports = [80]

        self.config += terrascript.resource.google_compute_firewall(
            name,
            name=name,
            network="default",
            allow={"protocol": "tcp", "ports": ports},
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()

    def reserve_ip(self, name, apply=True):
        """
        args:
            name: name of the reversed ip
            apply: to call terraform apply at the end
        """
        pygrid_network_ip = terrascript.resource.google_compute_address(
            name,
            name=name,
        )
        self.config += pygrid_network_ip

        self.config += terrascript.output(
            name + "-ip",
            value="${" + pygrid_network_ip.address + "}",
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()

    def create_gridnetwork(self, name, machine_type, zone, apply=True):
        """
        args:
            name: name of the compute instance
            machine_type: the type of machine
            zone: zone of your GCP project
            apply: to call terraform apply at the end
        """
        pygrid_network_ip = terrascript.resource.google_compute_address(
            name,
            name=name,
        )
        self.config += pygrid_network_ip

        self.config += terrascript.output(
            name + "-ip",
            value="${" + pygrid_network_ip.address + "}",
        )

        self.expose_port(name="pygrid", apply=False)

        image = terrascript.data.google_compute_image(
            name + "container-optimized-os",
            family="cos-81-lts",
            project="cos-cloud",
        )
        self.config += image

        node = terrascript.resource.google_compute_instance(
            name,
            name=name,
            machine_type=machine_type,
            zone=zone,
            boot_disk={"initialize_params": {"image": "${" + image.self_link + "}"}},
            network_interface={
                "network": "default",
                "access_config": {"nat_ip": "${" + pygrid_network_ip.address + "}"},
            },
            metadata_startup_script="""
                #!/bin/bash
                sleep 30;
                docker pull openmined/grid-network:production;
                docker run \
                -e PORT=80 \
                -e DATABASE_URL=sqlite:///databasenode.db \
                --name gridnetwork\
                -p 80:80 \
                -d openmined/grid-network:production;""",
        )
        self.config += node

        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()

    def create_gridnode(self, name, machine_type, zone, gridnetwork_name=None, apply=True):
        """
        args:
            name: name of the compute instance
            machine_type: the type of machine
            zone: zone of your GCP project
            gridnetwork_name: name of gridnetwork instance created
            apply: to call terraform apply at the end
        """
        if not gridnetwork_name:
            self.expose_port(name="pygrid", ports=[80], apply=False)

        with open("terraform.tfstate", "r") as out:
            outputs = json.load(out)["outputs"]

        gridnetwork_ip = outputs[gridnetwork_name + "-ip"]["value"]
        pygrid_network_address = "=http://" + gridnetwork_ip if gridnetwork_ip else ""
        host = "curl ifconfig.co" if gridnetwork_ip else "hostname -I"

        image = terrascript.data.google_compute_image(
            name + "container-optimized-os",
            family="cos-81-lts",
            project="cos-cloud",
        )
        self.config += image

        # HOST environment variable is set to internal IP address
        node = terrascript.resource.google_compute_instance(
            name,
            name=name,
            machine_type=machine_type,
            zone=zone,
            boot_disk={"initialize_params": {"image": "${" + image.self_link + "}"}},
            network_interface={"network": "default", "access_config": {}},
            metadata_startup_script=f"""
                #!/bin/bash
                sleep 30;
                docker pull openmined/grid-node:production;
                docker run \
                -e NODE_ID={name} \
                -e HOST = "$({host})" \
                -e PORT=80 \
                -e NETWORK={pygrid_network_address} \
                -e DATABASE_URL=sqlite:///databasenode.db \
                --name gridnode \
                -p 80:80 \
                -d openmined/grid-node:production;""",
        )
        self.config += node
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()

    def create_cluster(
        self,
        name,
        machine_type,
        zone,
        reserve_ip_name,
        target_size,
        eviction_policy=None,
        apply=True,
    ):
        """
        args:
            name: name of the compute instance
            machine_type: the type of machine
            zone: zone of your GCP project
            reserve_ip_name: name of the reserved ip created using reserve_ip
            target_size: number of wokers to be created(N workers + 1 master)
            eviction_policy: "delete" to teardown the cluster after calling
            cluster.sweep() else None
            apply: to call terraform apply at the end
        """
        self.expose_port("pygrid", ports=[80, 3000], apply=False)

        with open("terraform.tfstate", "r") as out:
            outputs = json.load(out)["outputs"]

        gridnetwork_ip = outputs[reserve_ip_name + "-ip"]["value"]
        pygrid_network_address = "http://" + gridnetwork_ip

        image = terrascript.data.google_compute_image(
            name + "container-optimized-os",
            family="cos-81-lts",
            project="cos-cloud",
        )
        self.config += image

        self.config += terrascript.resource.google_compute_instance(
            name + "pygridnetwork",
            name=name + "pygridnetwork",
            machine_type=machine_type,
            zone=zone,
            boot_disk={"initialize_params": {"image": "${" + image.self_link + "}"}},
            network_interface={"network": "default", "access_config": {"nat_ip": gridnetwork_ip}},
            metadata_startup_script=f"""
                #!/bin/bash
                sleep 30;
                docker pull openmined/grid-network:production && \
                docker run \
                -e PORT=80 \
                -e DATABASE_URL=sqlite:///databasenode.db \
                --name gridnetwork \
                -p 80:80 \
                -d openmined/grid-network:production;
                docker pull openmined/grid-node:production;
                docker run \
                -e NODE_ID={name+"-network-node"} \
                -e HOST="$(hostname -I)" \
                -e PORT=3000 \
                -e NETWORK={pygrid_network_address} \
                -e DATABASE_URL=sqlite:///databasenode.db \
                --name gridnode \
                -p 3000:3000 \
                -d openmined/grid-node:production;""",
        )

        # HOST environment variable is set to external IP address
        instance_template = terrascript.resource.google_compute_instance_template(
            name + "-template",
            name=name + "-template",
            machine_type=machine_type,
            disk={"source_image": "${" + image.self_link + "}"},
            network_interface={"network": "default", "access_config": {}},
            metadata_startup_script=f"""
                #!/bin/bash
                sleep 30;
                docker pull openmined/grid-node:production;
                docker run \
                -e NODE_ID={name} \
                -e HOST="$(curl ifconfig.co)" \
                -e PORT=80 \
                -e NETWORK={pygrid_network_address} \
                -e DATABASE_URL=sqlite:///databasenode.db \
                --name gridnode \
                -p 80:80 \
                -d openmined/grid-node:production;""",
            lifecycle={"create_before_destroy": True},
        )
        self.config += instance_template

        self.config += terrascript.resource.google_compute_instance_group_manager(
            name + "-cluster",
            name=name + "-cluster",
            version={"instance_template": "${" + instance_template.self_link + "}"},
            base_instance_name=name,
            zone=zone,
            target_size=str(target_size),
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()

        return Cluster(
            name,
            self.provider,
            gridnetwork_ip,
            eviction_policy=eviction_policy,
        )

    def compute_instance(self, name, machine_type, zone, image_family, apply=True):
        """
        args:
            name: name of the compute instance
            machine_type: the type of machine
            zone: zone of your GCP project
            image_family: image of the OS
            apply: to call terraform apply at the end
        """
        self.config += terrascript.resource.google_compute_instance(
            name,
            name=name,
            machine_type=machine_type,
            zone=zone,
            boot_disk={"initialize_params": {"image": image_family}},
            network_interface={"network": "default", "access_config": {}},
        )
        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()

    def destroy(self):
        """
        args:
        """
        if IPython.get_ipython():
            terraform_notebook.destroy()
        else:
            terraform_script.destroy()
        del self.credentials


class Cluster:
    """This class defines the Cluster object which is created in the method create_cluster"""

    def __init__(self, name, provider, gridnetwork_ip, eviction_policy=None):
        """
        args:
            name: name of the cluster
            provider: terrafrom provider for the cluster
            gridnetwork_ip: ip of grid network instance
            eviction_policy: "delete" to teardown the cluster after calling
            cluster.sweep() else None
        """
        self.name = name
        self.provider = provider
        self.gridnetwork_ip = gridnetwork_ip
        self.gridnetwork_node_ip = gridnetwork_ip + ":3000"
        self.master = name + "-master"
        self.cluster = name + "-cluster"
        self.template = name + "-template"
        self.eviction_policy = eviction_policy
        self.network_node = None
        self.config = None

    def sweep(
        self,
        model,
        hook,
        model_id=None,
        mpc=False,
        allow_download=False,
        allow_remote_inference=False,
        apply=True,
    ):
        """
        args:
            model : A jit model or Syft Plan.
            hook : A PySyft hook
            model_id (str): An integer/string representing the model id.
            If it isn't provided and the model is a Plan we use model.id,
            if the model is a jit model we raise an exception.
            allow_download (bool) : Allow to copy the model to run it locally.
            allow_remote_inference (bool) : Allow to run remote inferences.
            apply: to call terraform apply at the end
        """
        print("Connecting to network-node")
        self.network_node = DataCentricFLClient(hook, self.gridnetwork_node_ip)
        print("Sending model to node")
        self.network_node.serve_model(
            model=model,
            model_id=model_id,
            mpc=mpc,
            allow_download=allow_download,
            allow_remote_inference=allow_remote_inference,
        )
        print("Model sent, disconnecting node now")
        self.network_node.close()
        print("Node disconnected")

        with open("main.tf.json", "r") as main_config:
            self.config = json.load(main_config)

        if self.eviction_policy == "delete":
            if self.provider == "google":
                del self.config["resource"]["google_compute_instance_group_manager"][self.cluster]
                del self.config["resource"]["google_compute_instance_template"][self.template]

                if len(self.config["resource"]["google_compute_instance_group_manager"]) == 0:
                    del self.config["resource"]["google_compute_instance_group_manager"]

                if len(self.config["resource"]["google_compute_instance_template"]) == 0:
                    del self.config["resource"]["google_compute_instance_template"]

        with open("main.tf.json", "w") as main_config:
            json.dump(self.config, main_config, indent=2, sort_keys=False)

        if apply:
            if IPython.get_ipython():
                terraform_notebook.apply()
            else:
                terraform_script.apply()
