use crate::core::common::serde::serializable::Serializable;
use bytes::BytesMut;

pub(crate) fn serialize<T: Serializable>(obj: T, _compression: bool) -> BytesMut {
    let bytes_repr = obj._serialize();
    bytes_repr
}

#[cfg(test)]
mod tests {
    use crate::core::common::serde::deserialize::deserialize;
    use crate::core::common::serde::serialize::serialize;
    use crate::core::common::uid::RustUID;
    fn test_uid(result: RustUID) {}
    #[test]
    fn it_works() {
        let obj = RustUID::new();
        let bytes_buf = serialize(obj, true);
        let obj: RustUID = deserialize(bytes_buf);
    }
}
