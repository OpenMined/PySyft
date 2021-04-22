# syft absolute
import syft as sy


def test_libsyft() -> None:
    hello = sy.libsyft.hello_rust()
    assert hello == "Hello Rust 🦀"
