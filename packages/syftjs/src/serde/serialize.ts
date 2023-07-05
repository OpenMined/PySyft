import * as capnp from 'capnp-ts';

import { DataBox } from '../capnp/databox.capnp';
import { RecursiveSerde } from '../capnp/recursive_serde.capnp';

import { PRIMITIVES } from './primitives';
import { SerializableInterface } from './serializable_interface';

function checkPrimitive(obj: object) {
  const serde_obj = PRIMITIVES[typeof obj];
  if (typeof serde_obj !== 'undefined') {
    return serde_obj;
  } else {
    switch (obj) {
      case typeof obj === 'number':
        return Number.isInteger(obj) ? PRIMITIVES.integer : PRIMITIVES.float;
      case Array.isArray(obj):
        return PRIMITIVES.array;
      case obj && obj.byteLength !== undefined:
        return PRIMITIVES.bytes;
      case obj instanceof Map:
        return PRIMITIVES.dict;
      case obj:
        return obj;
    }
  }
}

function splitChunks(serializedObj: Uint8Array) {
  const sizeLimit = 5.12 ** 8;
  const chunks = [];
  let pointer = 0;

  if (serializedObj.byteLength <= sizeLimit) {
    // If the serialized object is smaller than the size limit, add it as a single chunk
    chunks.push(serializedObj);
  } else {
    // If the serialized object is larger than the size limit, split it into multiple chunks
    const numSlices = Math.ceil(serializedObj.byteLength / sizeLimit);
    for (let i = 0; i < numSlices - 1; i++) {
      // Push a slice of the serialized object to the chunks array
      chunks.push(serializedObj.slice(pointer, pointer + sizeLimit));
      pointer += sizeLimit;
    }
    // Push the last slice to the chunks array
    chunks.push(serializedObj.slice(pointer));
  }

  return chunks;
}

function createData(length: number) {
  const newDataMsg = new capnp.Message();
  const dataRoot = newDataMsg.initRoot(DataBox);
  dataRoot.initValue(length);
  return dataRoot.getValue();
}

function serializePrimitive(
  obj: SerializableInterface,
  rs: RecursiveSerde,
  serializer: (obj: SerializableInterface) => Uint8Array
) {
  // Serialize the object using the specified serializer function
  const serializedObj = serializer(obj);
  // Split the serialized object into chunks and initialize the data field
  const chunks = splitChunks(serializedObj);
  const data = rs.initNonrecursiveBlob(chunks.length);

  // Copy each chunk into a new payload and set it in the data field
  for (let i = 0; i < chunks.length; i++) {
    const payload = createData(chunks[i].byteLength);
    payload.copyBuffer(chunks[i]);
    data.set(i, payload);
  }

  return rs;
}

export function serialize(obj: SerializableInterface) {
  const serde_obj = checkPrimitive(obj);

  // Create a new Cap'n Proto message
  const message = new capnp.Message();

  // Initialize the root object of the message
  const rs = message.initRoot(RecursiveSerde);
  rs.setFullyQualifiedName(serde_obj.fqn);
  serializePrimitive(obj, rs, serde_obj.serialize);

  return message.toArrayBuffer();
}
