print('Hello Tiltfile')

# docker build -t openmined/grid-frontend -f packages/grid/grid-ui/ui.dockerfile ./packages/grid/grid-ui
docker_build(
    "openmined/grid-frontend",
    "packages/grid/grid-ui",
    dockerfile="packages/grid/grid-ui/ui.dockerfile"
)

# docker build -t openmined/grid-backend -f packages/grid/backend/backend.dockerfile ./packages/grid/backend
docker_build(
    "openmined/grid-backend",
    "packages/grid/backend",
    dockerfile="packages/grid/backend/backend.dockerfile"
)

# docker build -t openmined/grid-worker -f packages/grid/backend/celeryworker.dockerfile ./packages/grid/backend
docker_build(
    "openmined/grid-worker",
    "packages/grid/backend",
    dockerfile="packages/grid/backend/celeryworker.dockerfile"
)

#local('helm dep update helm-charts/pysyft -n openmined')

pysyft_yaml = helm(
    'helm-charts/pysyft',
    name='pysyft',
    namespace='openmined'
)
k8s_yaml(pysyft_yaml)
