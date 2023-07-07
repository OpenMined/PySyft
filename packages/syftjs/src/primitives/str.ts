import { PrimitiveInterface } from "./primitive_interface";

export const STRING: PrimitiveInterface = {
  serialize: serializeString,
  deserialize: deserializeString,
  fqn: "builtins.str",
};

function serializeString(obj: string) {
  return new TextEncoder().encode(obj).buffer;
}

function deserializeString(buffer: ArrayBuffer) {
  return new TextDecoder().decode(buffer);
}
