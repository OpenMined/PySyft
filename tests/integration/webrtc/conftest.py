"""Fixtures for the tests of WebRTC mode on the Firebase emulators."""

import pytest

from tests.integration.webrtc.firebase_emulators import (
    clear_emulators,
    running_emulators,
)


@pytest.fixture(scope="session")
def running_firebase_emulators():
    with running_emulators():
        yield


@pytest.fixture
def firebase_emulators(running_firebase_emulators):
    """The emulators, with no users and no documents."""
    clear_emulators()
