import * as capnp from 'capnp-ts';

import { Iterable } from '../capnp/iterable.capnp';
import { KVIterable } from '../capnp/kv_iterable.capnp';

import { deserialize, mergeChunks } from './deserialize';
import { SerializableInterface } from './serializable_interface';
import {
  createData,
  serialize,
  serializeChunks,
  splitChunks,
} from './serialize';

const PRIMITIVES: Record<string, SerializableInterface> = {
  string: {
    serialize: (obj: string) => {
      return new TextEncoder().encode(obj).buffer;
    },
    deserialize: (buffer: ArrayBuffer) => {
      return new TextDecoder().decode(buffer);
    },
    fqn: 'builtins.str',
  },
  boolean: {
    serialize: (obj: boolean) => {
      return obj ? new Uint8Array([49]).buffer : new Uint8Array([48]).buffer;
    },
    deserialize: (buffer: ArrayBuffer) => {
      return new Uint8Array(buffer)[0] == 49 ? true : false;
    },
    fqn: 'builtins.bool',
  },

  integer: {
    serialize: (obj: number) => {
      return capnp.Int64.fromNumber(obj).buffer.reverse().buffer;
    },
    deserialize: (buffer: ArrayBuffer) => {
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
    },
    fqn: 'builtins.int',
  },

  float: {
    serialize: (obj: number) => {
      let hex_str = '';

      if (obj < 0) {
        hex_str += '-';
        obj = obj * -1;
      }
      hex_str += '0x';
      const exponent = 10;

      const n_1 = obj / 2 ** exponent;
      hex_str += Math.trunc(n_1).toString(16);
      hex_str += '.';

      let notZeros = true;
      let currNumber = n_1;
      while (notZeros) {
        currNumber = (currNumber % 1) * 16;
        hex_str += Math.trunc(currNumber).toString(16);
        if (currNumber % 1 === 0) {
          notZeros = false;
        }
      }
      hex_str += 'p' + exponent;
      return new TextEncoder().encode(hex_str);
    },
    deserialize: (buffer: ArrayBuffer) => {
      const hex_str = new TextDecoder().decode(buffer);
      let aggr = 0;
      const [signal, int_n, hex_dec_n, exp] = hex_str
        .replaceAll('.', ' ')
        .replaceAll('0x', ' ')
        .replaceAll('p', ' ')
        .split(' ');
      aggr += parseInt(int_n, 16);

      let n_signal: number;
      if (signal) {
        n_signal = -1;
      } else {
        n_signal = 1;
      }
      // bracket notation
      for (let i = 0; i < hex_dec_n.length; i++) {
        aggr += parseInt(hex_dec_n[i], 16) / 16.0 ** (i + 1);
      }
      return aggr * 2 ** parseInt(exp, 10) * n_signal;
    },
    fqn: 'builtins.float',
  },
  bytes: {
    serialize: (obj: Uint8Array) => {
      return obj.buffer;
    },
    deserialize: (buffer: ArrayBuffer) => {
      return new Uint8Array(buffer);
    },
    fqn: 'builtins.bytes',
  },

  list: {
    serialize: (obj: any[]) => {
      const message = new capnp.Message();
      const rs = message.initRoot(Iterable);
      const listStruct = rs.initValues(obj.length);
      let count = 0;
      for (let index = 0; index < obj.length; index++) {
        const serializedObj = serialize(obj[index]);
        const chunks = splitChunks(serializedObj);
        const chunkList = serializeChunks(chunks);
        listStruct.set(count, chunkList);
        count += 1;
      }
      return message.toArrayBuffer();
    },
    deserialize: (buffer: ArrayBuffer) => {
      const iter = [];
      const message = new capnp.Message(buffer, false);
      const rs = message.getRoot(Iterable);
      const values = rs.getValues();
      const size = values.getLength();
      for (let index = 0; index < size; index++) {
        const value = values.get(index);
        const fullBlob = mergeChunks(value);
        const obj = deserialize(fullBlob);
        iter.push(obj);
      }
      return iter;
    },
    fqn: 'builtins.list',
  },
  dict: {
    serialize: (obj: Map<SerializableInterface, SerializableInterface>) => {
      const message = new capnp.Message();
      const rs = message.initRoot(KVIterable);
      const keys = rs.initKeys(obj.size);
      const values = rs.initValues(obj.size);
      let count = 0;
      obj.forEach(
        (value: SerializableInterface, key: SerializableInterface) => {
          // Saving Key
          const serializedKey = serialize(key);
          const keyDataStruct = createData(serializedKey.byteLength);
          keyDataStruct.copyBuffer(serializedKey);
          keys.set(count, keyDataStruct);

          // Saving Values
          const serializedValue = serialize(value);
          const chunks = splitChunks(serializedValue);
          const chunkList = serializeChunks(chunks);
          values.set(count, chunkList);

          count += 1;
        },
      );
      return message.toArrayBuffer();
    },
    deserialize: (buffer: ArrayBuffer) => {
      // Initialize an empty object to store the key-value pairs.
      const kv_iter: Record<string, object> = {};

      // Create a capnp.Message object from the input buffer and get the root object of type KVIterable.
      const message = new capnp.Message(buffer, false);
      const rs = message.getRoot(KVIterable);

      // Get the values, keys, and size of the KVIterable object.
      const values = rs.getValues();
      const keys = rs.getKeys();
      const size = values.getLength();

      // Iterate over the KVIterable object and process each value.
      for (let index = 0; index < size; index++) {
        const value = values.get(index);
        const key = keys.get(index);

        // Process the value using the processObject function.
        const fullBlob = mergeChunks(value);
        const obj = deserialize(fullBlob);

        // Deserialize the key using the deserialize method and convert it to an ArrayBuffer.
        const deserializedKey = deserialize(key.toArrayBuffer());
        let keyObj = '';
        if (
          Object.prototype.toString.call(deserializedKey) === '[object Object]'
        ) {
          keyObj = JSON.stringify(deserializedKey);
        } else {
          keyObj = deserializedKey;
        }
        // Store the processed object as a value corresponding to the deserialized key in the kv_iter object.
        kv_iter[keyObj] = obj;
      }
      return kv_iter;
    },
    fqn: 'builtins.dict',
  },
};

const PRIMITIVE_REVERSE_MAP = new Map<string, string>([
  [PRIMITIVES.string.fqn, 'string'],
  [PRIMITIVES.boolean.fqn, 'boolean'],
  [PRIMITIVES.integer.fqn, 'integer'],
  [PRIMITIVES.float.fqn, 'float'],
  [PRIMITIVES.bytes.fqn, 'bytes'],
  [PRIMITIVES.list.fqn, 'list'],
  [PRIMITIVES.dict.fqn, 'dict'],
]);

export const getPrimitiveByObj = (obj: any): SerializableInterface => {
  let obj_type: string = typeof obj;

  if (!(obj_type in PRIMITIVES)) {
    if (obj_type === 'number') {
      obj_type = Number.isInteger(obj) ? 'integer' : 'float';
    } else if (Array.isArray(obj)) {
      obj_type = 'list';
    } else if (obj && obj.byteLength !== undefined) {
      obj_type = 'bytes';
    } else if (obj instanceof Map) {
      obj_type = 'dict';
    } else {
      return obj;
    }
  }

  const serde_obj: SerializableInterface = PRIMITIVES[obj_type];
  return serde_obj;
};

export const getPrimitiveByFqn = (fqn: string) => {
  return PRIMITIVES[PRIMITIVE_REVERSE_MAP.get(fqn)];
};
