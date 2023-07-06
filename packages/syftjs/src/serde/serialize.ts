import * as capnp from 'capnp-ts';

import { RecursiveSerde } from '../capnp/recursive_serde.capnp';

import { getSerdeSchema, serializeObject } from './serializable';

/**
 * Method used to serialize a SyftJS Object or Primitives into a Cap'n Proto array buffer.
 * @param {any} obj - Object to be serialized.
 * @returns {ArrayBuffer} Array buffer with capnp structure of the serialized object.
 */
export function serialize(obj: any) {
  // Create a new Cap'n Proto message
  const message = new capnp.Message();
  const rs = message.initRoot(RecursiveSerde);

  // Get the serde schema of the object
  const serdeSchema = getSerdeSchema(obj);
  rs.setFullyQualifiedName(serdeSchema.fqn);

  // Call inner method to decide how to serialize the object (primitive/recursive).
  serializeObject(obj, serdeSchema, rs);

  return message.toArrayBuffer();
}
