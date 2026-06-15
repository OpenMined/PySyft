from syft_client.sync.platforms.base_platform import BasePlatform


class GdriveFilesPlatform(BasePlatform):
    name: str = "Gdrive Files"
    module_path: str = "google_personal.gdrive_files"
