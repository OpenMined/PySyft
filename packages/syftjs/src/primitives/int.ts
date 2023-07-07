import * as capnp from "capnp-ts";

import { PrimitiveInterface } from "./primitive_interface";

export const INTEGER: PrimitiveInterface = {
  serialize: serializeInt,
  deserialize: deserializeInt,
  fqn: "builtins.int",
};

function serializeInt(obj: number) {
  return capnp.Int64.fromNumber(obj).buffer.reverse().buffer;
}

function deserializeInt(buffer: ArrayBuffer) {
  const buffer_array = new Uint8Array(buffer); // Not sure why but first byte is always zero, so we need to remove it.
  buffer_array.reverse(); // Little endian / big endian
  if (buffer.byteLength < 8) {
    const array64 = new Uint8Array(8);
    array64.set(new Uint8Array(buffer_array));
    buffer = array64.buffer;
  } else {
    buffer = buffer_array.buffer;
  }
  return capnp.Int64.fromArrayBuffer(buffer).toNumber(false);
}
