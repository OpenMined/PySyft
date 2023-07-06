import { PrimitiveInterface } from './primitive_interface';

export const BOOLEAN: PrimitiveInterface = {
  serialize: (obj: boolean) => {
    return obj ? new Uint8Array([49]).buffer : new Uint8Array([48]).buffer;
  },
  deserialize: (buffer: ArrayBuffer) => {
    return new Uint8Array(buffer)[0] == 49 ? true : false;
  },
  fqn: 'builtins.bool',
};
