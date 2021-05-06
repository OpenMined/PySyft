use crate::core::common::serde::serializable::Serializable;
use bytes::BytesMut;

pub(crate) fn deserialize<T>(bytes: BytesMut) -> T
where
    T: Serializable,
{
    Serializable::_deserialize(bytes)
}
