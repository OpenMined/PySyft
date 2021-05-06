use crate::proto::util::DataMessage;
use bytes::BytesMut;
use prost::Message;

pub(crate) trait Serializable {
    type ProtobufType: prost::Message + Default;

    fn _object2protobuf(&self) -> Self::ProtobufType;
    fn _protobuf2object(object: Self::ProtobufType) -> Self;

    fn _serialize(&self) -> BytesMut {
        let protobuf_object = self._object2protobuf();
        let mut obj_bytes = BytesMut::new();
        protobuf_object.encode(&mut obj_bytes).unwrap();

        //this will be changed soon to SyftNative
        let data_msg = DataMessage {
            obj_type: "TODO".to_string(),
            content: obj_bytes.to_vec(),
        };

        let mut data_bytes = BytesMut::new();
        data_msg.encode(&mut data_bytes).unwrap();
        data_bytes
    }

    fn _deserialize(bytes: BytesMut) -> Self
    where
        Self: Sized,
    {
        let data_msg: DataMessage = DataMessage::decode(bytes).unwrap();
        let proto_obj: Self::ProtobufType =
            Self::ProtobufType::decode(data_msg.content.as_slice()).unwrap();
        Serializable::_protobuf2object(proto_obj)
    }
}
