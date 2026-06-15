from typing import Optional

from syft_bg.services.base import Service, ServiceInfo, ServiceStatus
from syft_bg.services.registry import SERVICE_REGISTRY


class ServiceManager:
    def __init__(self):
        self.services = SERVICE_REGISTRY

    def list_services(self) -> list[str]:
        return list(self.services.keys())

    def get_service(self, name: str) -> Optional[Service]:
        return self.services.get(name)

    def get_service_infos(self) -> dict[str, ServiceInfo]:
        return {name: svc.info() for name, svc in self.services.items()}

    def start_service(self, name: str) -> tuple[bool, str]:
        service = self.get_service(name)
        if not service:
            raise ValueError(f"Unknown service: {name}")
        service.start()

    def stop_service(self, name: str) -> tuple[bool, str]:
        service = self.get_service(name)
        if not service:
            raise ValueError(f"Unknown service: {name}")
        service.stop()

    def restart_service(self, name: str) -> tuple[bool, str]:
        service = self.get_service(name)
        if not service:
            raise ValueError(f"Unknown service: {name}")
        service.restart()

    def start_all(self) -> dict[str, tuple[bool, str]]:
        for name in self.services:
            try:
                self.start_service(name)
            except Exception as e:
                print(f"Error starting service {name}: {e}")

    def stop_all(self) -> dict[str, tuple[bool, str]]:
        for name in self.services:
            try:
                self.stop_service(name)
            except Exception as e:
                print(f"Error stopping service {name}: {e}")

    def get_logs(self, name: str, lines: int = 50) -> list[str]:
        service = self.get_service(name)
        if not service:
            raise ValueError(f"Unknown service: {name}")
        return service.get_logs(lines)

    def any_running(self) -> bool:
        return any(
            svc.info().status == ServiceStatus.RUNNING for svc in self.services.values()
        )

    def is_running(self, name: str) -> bool:
        service = self.get_service(name)
        if not service:
            raise ValueError(f"Unknown service: {name}")
        return service.is_running()

    def all_running(self) -> bool:
        return all(
            svc.info().status == ServiceStatus.RUNNING for svc in self.services.values()
        )
