# third party
import veilid


def get_typed_key(key: str) -> veilid.types.TypedKey:
    return veilid.types.TypedKey.from_value(
        kind=veilid.CryptoKind.CRYPTO_KIND_VLD0, value=key
    )


# state = await conn.get_state()
# state.config.config
