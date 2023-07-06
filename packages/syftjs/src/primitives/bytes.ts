import { PrimitiveInterface } from './primitive_interface';

export const BYTES: PrimitiveInterface = {
  serialize: (obj: Uint8Array) => {
    return obj.buffer;
  },
  deserialize: (buffer: ArrayBuffer) => {
    return new Uint8Array(buffer);
  },
  fqn: 'builtins.bytes',
};
