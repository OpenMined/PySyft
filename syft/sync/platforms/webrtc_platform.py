from syft.sync.platforms.base_platform import BasePlatform


class WebRTCPlatform(BasePlatform):
    name: str = "WebRTC"
    module_path: str = "webrtc"
