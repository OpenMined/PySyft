.. _advanced_deployment:

===========================================
Advanced Deployment: Introduction to HaGrid
===========================================

.. toctree::
   :maxdepth: 3

Hagrid (HAppy GRID!) is a command-line tool that speeds up the
deployment of PyGrid, the software providing a peer-to-peer network of
data owners and data scientists who can collectively train AI models
using `PySyft <https://github.com/OpenMined/PySyft/>`__.

Hagrid is able to orchestrate a collection of PyGrid Domain and Network
nodes and scale them in a local development environment (based on a
docker-compose file). By stacking multiple copies of this docker, you
can simulate multiple entities (e.g countries) that collaborate over
data and experiment with more complicated data flows such as SMPC.

Similarly to the local deployment, Hagrid can bootstrap docker on a
Vagrant VM or on a cloud VM, helping you deploy in an user-friendly way
on Azure, AWS\* and GCP*.

*\* Deploying to AWS and GCP is still under development.*

Working with Hagrid & Syft API versions:

-  **Development mode:**
      You can experiment with your own local checked-out version of Syft
      and bootstrap a local Jupyter Notebook where you can use the Syft
      & Grid API to communicate with a prod/local dev system\ *.*

-  **Production mode:**
      You can specify the branch and repository you want to fork (including your own fork) and Hagrid will monitor those branches in a cron job, pull new changes and restart the services to apply them, therefore your deployed system will always stay up to date.

Prerequisites
===============

The following operating systems are currently supported: Linux, Windows, MacOS. Please ensure you have at least 8GB of ram if you intend to run Hagrid locally.

Setting up virtual environment using Python 3.9
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Ensure using **Python3.8+**, which should be first installed in your system. To easily handle further dependencies, we suggest using conda:

   a. Install conda `following these instructions <https://docs.anaconda.com/anaconda/install/index.html/>`_  depending on your OS.

   b. Create a new env specifying the Python version (we recommend Python 3.8/3.9) in the terminal:

      .. code-block:: bash

         $ conda create -n myenv python=3.9
         $ conda activate myenv
         (to exit): conda deactivate

Using latest pip
~~~~~~~~~~~~~~~~~

**Pip** is required to install dependencies, so make sure you have it installed and up-to-date by running the following these `instructions <https://pip.pypa.io/en/stable/installation/#supported-methods/>`__.

If you have it installed, please check it is the latest version:

.. code-block:: bash

    $ pip install --upgrade pip && pip -V (Linux)
    $ python -m pip install --upgrade pip (for Windows)


Install Jupyter Notebook
~~~~~~~~~~~~~~~~~~~~~~~~~

1. A very convenient way to interact with a deployed node is via Python, using a Jupyter Notebook. You can install it by running:

   .. code-block:: bash

      $ pip install jupyter-notebook

2. If you encounter issues, you can also install it using Conda:

   .. code-block:: bash

      $ conda install -c conda-forge notebook

3. To launch the Jupyter Notebook, you can run the following in your terminal:

   .. code-block:: bash

      $ jupyter notebook

Installing and configuring Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install **Docker** and **Docker Composite V2,** which is needed to orchestrate docker, as explained below:

   For **Linux**:

   a. Install **Docker**:

      .. code-block:: bash

         $ sudo apt-get upgrade docker & docker run hello-world

   b. Install **Docker Composite V2** as described `here <https://docs.docker.com/compose/cli-command/#installing-compose-v2>`__.

   c. You should see ‘Docker Compose version v2’ when running:

      .. code-block:: bash

         $ docker compose version

   d. If not, go through the `instructions here <https://www.rockyourcode.com/how-to-install-docker-compose-v2-on-linux-2021/>`__ or if you are using Linux, you can try to do:

      .. code-block:: bash

         $ mkdir -p ~/.docker/cli-plugins
         $ curl -sSL https://github.com/docker/compose-cli/releases/download/v2.0.0-beta.5/docker-compose-linux-amd64 -o ~/.docker/cli-plugins/docker-compose
         $ chmod +x ~/.docker/cli-plugins/docker-compose

   e. Also, make sure you can run without sudo:

      .. code-block:: bash

         $ echo $USER //(should return your username)
         $ sudo usermod -aG docker $USER


   For **Windows**, **MacOs**:

   a. You can install Desktop Docker as explained `here for Windows <https://docs.docker.com/docker-for-windows/install/>`_ or `here for MacOS <https://docs.docker.com/docker-for-mac/install/>`_.

   b. The ``docker-compose`` should be enabled by default. If you encounter issues, you can check it by:

      -  Go to the Docker menu, click ``Preferences (Settings on Windows)`` > ``Experimental features``.

      -  Make sure the Use ``Docker Compose V2`` box is checked.

   c. Ensure at least 8GB of RAM are allocated in the Desktop Docker app:

      -  Go to 'Preferences' -> 'Resources'

      -  Drag the 'Memory' dot until it says at least 8.00GB

      -  Click 'Apply & Restart'

2. Make sure you are using the **dev** branch of the PySyft repository (branch can be found `here <https://github.com/OpenMined/PySyft/tree/0.6.0>`__)


Explore locally with the PySyft API
====================================

1. Install **tox**:

   .. code-block:: bash

      $ pip install tox

2. Move to the correct branch in the PySyft repository:

   .. code-block:: bash

      $ git checkout dev

3. Check current tasks that can be run by tox:

   .. code-block:: bash

      $ tox -l

4. Open an editable Jupyter Notebook which doesn't require to run in a container:

   .. code-block:: bash

      $ tox -e syft.jupyter


Local deployment using Docker
====================================

1. Install Hagrid:

   .. code-block:: bash

      $ pip install -U hagrid

2. Launch a Domain Node:

   .. code-block:: bash

      $ hagrid launch domain


   .. note::

      First run **it might take ~5-10 mins** to build the PyGrid docker image. Afterwards, you should see something like:

      .. code-block:: bash

         Launching a domaing PyGrid node on port 8081 !

         - TYPE: domain
         - NAME: mystifying_wolf
         - TAG: 035c3b6a378a50f78cd74fc641d863c7
         - PORT: 8081
         - DOCKER: v2.2.3

   Optionally, you can provide here additional args to use a certain repository and branch, as:

   .. code-block:: bash

      $ hagrid launch domain --repo $REPO --branch $BRANCH

3. Go to ``localhost:port/login`` in your browser (using the port specified in your CLI, here *8081*) to see the PyGrid Admin UI where you, as a data owner, can manage your PyGrid deployment.

   a. Log in using the following credentials:

       .. code-block:: python

          info@openmined.org

          changethis


   b. Explore the interface or you can even do requests via `Postman <https://www.postman.com/downloads/>`__. You can check all the available endpoints at http://localhost:8081/api/v1/openapi.json/ and have all the following environment variables set (a more detailed explanationcan be found in `this video section <https://youtu.be/GCw7cN7xXJU?t=442>`__):

      |image0|

      The auth token can be obtained by doing a login request as follows:

      |image1|

4. While the Domain Node is online, you can start a Jupyter Notebook as described `above <#explore-locally-with-the-pysyft-api-no-containers-involved>`__ to use PySyft to communicate to it in a Python client rather than a REST API. Connecting to it can be done as following:

   .. code-block:: python

      import syft as sy

      domain = sy.login(email='info@openmined.org', password='changethis', port=8081)

      domain.store

      domain.requests

      Domain.users

5. To stop the node, run:

   .. code-block:: bash

      $ hagrid land --tag=035c3b6a378a50f78cd74fc641d863c7 (using the TAG specified in your CLI)


Local deployment using Vagrant and VirtualBox
===============================================

This is particularly useful to experiment with the Ansible scripts to test new changes.

1. Run hagrid status and ensure all dependencies are checked to make sure you have Vagrant and VirtualBox installed.

   |image2|

2. For installing Vagrant, check the `instructions here. <https://www.vagrantup.com/downloads>`__

3. Additionally to Vagrant, we need to install a plugin called landrush that allows using a custom DNS that points to the IP address used in the VM:

   .. code-block:: bash

      $ vagrant plugin install landrush

3. Move to the correct branch and directory in the PySyft repository:

   .. code-block:: bash

      $ git checkout 0.6.0
      $ cd packages/grid


4. Create the environment using vagrant for the first time:

   .. code-block:: bash

      $ vagrant init
      $ vagrant up


   When the VM is booted up, it starts the docker service and then the docker service starts all the containers as configured. As it is just created, provisioning is always **run** automatically\ **.**

   When deploying locally, the tasks listed in ‘main.yml’ for the node are not being run. Therefore, it does not have to do the lengthy
   setup every time (installing docker, cloning PySyft and launching the cronjob to reload PySyft).

   .. note:: The tasks for the containers and nodes respectively can be found in \*.yml files defined in ``packages/grid/ansible/roles/containers`` and ``packages/grid/ansible/roles/nodes``

5. If you intend to run it frequently and not only once, either run ``vagrant status`` to see if the env has already been created and if yes, to ``run vagrant up --provision`` every time to launch the provisioners, otherwise it is just resuming the existing machine.

6. To access the VM via SSh and jump to the user we are creating in vagrant:

   .. code-block:: bash

      $ vagrant ssh
      $ sudo su -om
      $ whoami # should return 'om'

8. You can go to ``http://10.0.1.2/login`` which is at port 80 to access the PyGrid Admin UI, which you can explore, query via Postman or in a
      local Jupyter Notebook using a Python client as described in `steps 3 and 4 here <#local-deployment-using-docker>`__.

9. To shut down the machine currently managed by Vagrant, you can run the following after exiting this node shell:

   .. code-block:: bash

      $ vagrant halt

10. Or alternatively to destroy it using:

    .. code-block:: bash

       $ vagrant destroy


Deploying on Kubernetes
========================

We provide an option to deploy the stack using kubernetes. To test and run this locally we use ``minikube`` and ``devspace``.

These are the prerequisites needed further, which are explained step-by-step below:

* docker
* hyperkit
* minikube
* devspace
* kubectl
* kubectx

MacOS
~~~~~

* **Hyperkit**

Ingress is not working on Mac and Docker and the issue is `being tracked here <https://github.com/kubernetes/minikube/issues/7332>`_. Until then we will use the ``hyperkit`` backend.

#. Install hyperkit by running:

.. code-block:: bash

    $ brew install hyperkit


* **Docker**

#. See above about using ``hyperkit`` on Mac until the ingress issue is fixed.

#. We will be using Docker - however you do not need to ``enable kubernetes`` in your Docker Desktop App. If it is enabled, disable it and click `Apply & Restart`.

#. This is because we will use ``minikube`` which will create and manage all the k8s resources we require as a normal container in docker engine. We install it by running:

.. code-block:: bash

	$ brew install minikube



* **Minikube**

1. ``minikube`` is a mini master k8s node that you can run on your local machine in a similar manner to Docker. To use minikube you need it to be running:

.. code-block:: bash

    $ minikube config set driver hyperkit
    $ minikube start --disk-size=40g
    $ minikube addons enable ingress

2. If you ever need to reset ``minikube`` you can do:

.. code-block:: bash

    $ minikube delete --all --purge

3. Once ``minikube`` is running, you should see the container in Docker by running:

.. code-block:: bash

    $ docker ps
    CONTAINER ID   IMAGE                                 COMMAND                  CREATED        STATUS              PORTS                                                                                                                                  NAMES
    57f73851bf08   gcr.io/k8s-minikube/kicbase:v0.0.25   "/usr/local/bin/entr…"   46 hours ago   Up About a minute   127.0.0.1:57954->22/tcp, 127.0.0.1:57955->2376/tcp, 127.0.0.1:57957->5000/tcp, 127.0.0.1:57958->8443/tcp, 127.0.0.1:57956->32443/tcp minikube



* **Kubectl**

``kubectl`` is the CLI tool for kubernetes. If you have ran ``minikube``, it should have configured your kubectl to point to the local minikube cluster by default.

You should be able to see this if you run the following command:

.. code-block:: bash

   $ kubectl get all
   NAME                 TYPE        CLUSTER-IP   EXTERNAL-IP   PORT(S)   AGE
   service/kubernetes   ClusterIP   10.96.0.1    <none>        443/TCP   45h

* **k8s Namespaces**

To understand the usage of ``k8s Namespaces``, think of a namespace as a grouping of resources and permissions which lets you easily create and destroy everything related to a single keyword.

.. code-block:: bash

   $ kubectl get namespaces
   NAME                   STATUS   AGE
   default                Active   45h
   kube-node-lease        Active   45h
   kube-public            Active   45h
   kube-system            Active   45h
   kubernetes-dashboard   Active   45h

All k8s have a default namespace and the other ones here are from kubernetes and minikube.

We will use the namespace ``openmined`` to make it clear what belongs to the Grid stack and what is something else. To create it, we can run:

.. code-block:: bash

   $ kubectl create namespace openmined

.. code-block:: bash

   $ kubectl get all -n openmined
   No resources found in openmined namespace.


* **Kubectx**

``kubectx`` is a package of helpful utilities which can help you do things like set a default namespace.

.. code-block:: bash

   $ brew install kubectx

Now we can use a tool like ``kubens`` to change the default namespace to openmined.

.. code-block:: bash

   $ kubens openmined
   Context "minikube" modified.
   Active namespace is "openmined".

Now when we use commands without `-n` we get openmined by default.

.. code-block:: bash

   $ kubectl get all
   No resources found in openmined namespace.

* **Helm Charts**

The most popular way to deploy applications to k8s is with a tool called Helm. What helm aims to do is to provide another layer of abstraction over kubernetes yaml configuration with hierarchical variables, templates and a package definition which can be hosted over HTTP allowing custom applications to depend on other prefabricated helm charts or to provide consumable packages of your code as a helm chart itself.

* **devspace**

To make development and deployment of our kubernetes code easier, we use a tool called ``devspace`` which aims to be like a hot reloading dev optimised version of `docker compose` but for kubernetes. More documentation can be `found here <https://devspace.sh/>`_.

Additionally ``devspace`` allows us to deploy using helm by auto-generating the values and charts from the ``devspace.yaml`` which means the single source of truth can be created which includes both production helm charts and kubernetes yaml configuration as well as local dev overrides.

.. code-block:: bash

   $ brew install devspace


Deploy to local dev
~~~~~~~~~~~~~~~~~~~

1. Check that you have the right namespace:

.. code-block:: bash

    $ devspace list namespaces
    Name                   Default   Exists
    default                false     true
    kube-node-lease        false     true
    kube-public            false     true
    kube-system            false     true
    kubernetes-dashboard   false     true
    openmined              *true*      true

2. Run the ``dev`` command with ``devspace``:

* To run a network with headscale VPN:

.. code-block:: bash

   $ cd packages/grid
   $ devspace dev -b -p network

* To run a domain without the headscale VPN:

.. code-block:: bash

   $ cd packages/grid
   $ devspace dev -b -p domain

3. Connect VPN in dev:

You can run the connect VPN settings using all the opened ports with:

.. code-block:: bash

   $ cd packages/grid
   $ python3 vpn/connect_vpn.py http://localhost:8088 http://localhost:8087 http://headscale:8080

4. Destroy the local deployment

.. code-block:: bash

   $ devspace purge

5. Delete persistent volumes

The database and the VPN containers have persistent volumes.

* You can check them with:

.. code-block:: bash

   $ kubectl get persistentvolumeclaim

* Then delete PostgreSQL as it follows:

.. code-block:: bash

   $ kubectl delete persistentvolumeclaim app-db-data-db-0

6. Check which images / tags are being used

This will show all the unique images and their tags currently deployed which is useful
when debugging which version is actually running in the cluster.

.. code-block:: bash

   $ kubectl get pods --all-namespaces -o jsonpath="{.items[*].spec.containers[*].image}" | tr -s '[[:space:]]' '\n' | sort | uniq -c


7. Restart a container / pod / deployment

* To get all the deployments:

.. code-block:: bash

   $ kubectl get deployments
   NAME             READY   UP-TO-DATE   AVAILABLE   AGE
   backend          1/1     1            1           18m
   backend-stream   1/1     1            1           18m
   backend-worker   1/1     1            1           18m
   frontend         1/1     1            1           18m
   queue            1/1     1            1           19m

* Restart the backend-worker

.. code-block:: bash

   $ kubectl rollout restart deployment backend-worker


Deploy to Google Kubernetes Engine (GKE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.Configure kubectl context with GKE:

.. code-block:: bash

   $ gcloud container clusters get-credentials --region us-central1-c staging-cluster-1

2. Check that you have the correct context

.. code-block:: bash

   $ kubectx

3. Configure your Google Container Registry (GCR):

.. code-block:: bash

   $ gcloud auth configure-docker

4. Check your settings with print

.. code-block:: bash

   $ devspace print -p domain --var=CONTAINER_REGISTRY=gcr.io/reflected-space-315806/

5. You should see that you are creating a domain and that the container registry variable changes the image name to:

.. code-block:: bash

   images:
   	backend:
    	image: gcr.io/reflected-space-315806/openmined/grid-backend

.. note:: This will tell ``devspace`` to publish to the GCR for your active GCP project.

6. Create the openmined namespace

.. code-block:: bash

   $ kubectl create namespace openmined

7. Tell devspace to use the openmined namespace

.. code-block:: bash

   $ devspace use namespace openmined

8. Deploy to GKE:

.. code-block:: bash

   $ devspace deploy -p domain --var=CONTAINER_REGISTRY=gcr.io/reflected-space-315806/

9. Access a container directly:

.. code-block:: bash

   $ devspace enter

10. Attach to container stdout:

.. code-block:: bash

   $ devspace attach

11. Use port forwarding to access an internal service:

.. code-block:: bash

   $ kubectl port-forward deployment/tailscale :4000


Deploying to Azure
====================================

1. Get your virtual machine on Azure ready

   a. To create one, you can either go to `portal.azure.com <http://portal.azure.com>`__ or use `this 1-click template <https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FOpenMined%2FPySyft%2Fdev%2Fpackages%2Fgrid%2Fquickstart%2Ftemplate.json>`__ available off-the-shelves.

   b. If you proceed to create it yourself, make sure you respect the following:

      i.   Use ``Ubuntu Server 20.04`` or newer

      ii.  Select ``SSH``, ``HTTP``, ``HTTPS`` as inbound ports

      iii. Have at least ``2x CPU``, ``4GB RAM``, ``40GB HDD``.

      .. note::
         During creation, write down the username used and save the key locally. In case warnings arise regarding having an unprotected key, you can run:

         .. code-block:: bash

            $ sudo chmod 600 key.pem

2. To deploy to Azure, the following can be run:

   .. code-block:: bash

      $ hagrid launch node --username=azureuser --key-path=~/hagriddeploy_key.pem domain to 51.124.153.133


   Additionally, you are being asked if you want to provide another repository and branch to fetch and update HAGrid, which you can skip by pressing ``Enter``.

3. If successful, you can now access the deployed node at the specified IP address and interact with it via the PyGrid Admin UI at http://51.124.153.133/login (change IP with yours) or use Postman to do API requests.

.. |image0| image:: ../_static/deployment/image2.png
   :width: 95%

.. |image1| image:: ../_static/deployment/image1.png
   :width: 95%

.. |image2| image:: ../_static/deployment/image3.png
   :width: 95%
