import * as capnp from "capnp-ts";

import { RecursiveSerde } from "../capnp/recursive_serde.capnp";

import { deserializeObject } from "./serializable";

/**
 * Method used to deserialize a SyftJS Object or Primitives received in Cap'n Proto array buffer format.
 * @param {ArrayBuffer} buffer - Array buffer containing all the capnp structure and object information.
 * @returns {Any} SyftJS / Primitive object instance.
 */
export function deserialize(buffer: ArrayBuffer) {
  const message = new capnp.Message(buffer, false);
  const rs = message.getRoot(RecursiveSerde);
  const fqn = rs.getFullyQualifiedName();

  // Call inner method to decide how to deserialize the object (primitive/recursive).
  return deserializeObject(fqn, rs);
}
