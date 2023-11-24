# syft absolute
from syft.custom_worker.builder import CustomWorkerBuilder
from syft.custom_worker.config import CustomWorkerConfig

if __name__ == "__main__":
    config = """
    build:
        gpu: false
        python_version: 3.11
        system_packages:
            - ffmpeg
        python_packages:
            - ffmpeg-python==0.2.0
            - llama-index==0.9.6.post2
    """
    config = CustomWorkerConfig.from_str(config)
    builder = CustomWorkerBuilder()

    print("building image...")
    builder.build_image(config)
    print("done")
