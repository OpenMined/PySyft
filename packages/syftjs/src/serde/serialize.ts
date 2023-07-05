import * as capnp from 'capnp-ts';

import { RecursiveSerde } from '../capnp/recursive_serde.capnp';

import { getPrimitiveByObj } from './primitives';
import { createData, splitChunks } from './utils';

function serializePrimitive(
  obj: any,
  rs: RecursiveSerde,
  serializer: (obj: any) => ArrayBuffer,
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

/**
 * Method used to serialize a SyftJS Object or Primitives into a Cap'n Proto array buffer.
 * @param {any} obj - Object to be serialized.
 * @returns {ArrayBuffer} Array buffer with capnp structure of the serialized object.
 */
export function serialize(obj: any) {
  const serde_obj = getPrimitiveByObj(obj);

  // Create a new Cap'n Proto message
  const message = new capnp.Message();

  // Initialize the root object of the message
  const rs = message.initRoot(RecursiveSerde);
  rs.setFullyQualifiedName(serde_obj.fqn);
  serializePrimitive(obj, rs, serde_obj.serialize);

  return message.toArrayBuffer();
}
