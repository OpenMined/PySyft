import * as capnp from 'capnp-ts';

import { KVIterable } from '../capnp/kv_iterable.capnp';
import { deserialize } from '../serde/deserialize';
import { mergeChunks } from '../serde/serializable';
import { serialize } from '../serde/serialize';
import { createData, serializeChunks, splitChunks } from '../serde/utils';

import { PrimitiveInterface } from './primitive_interface';

export const MAP: PrimitiveInterface = {
  serialize: serializeMap,
  deserialize: deserializeMap,
  fqn: 'builtins.dict',
};

function serializeMap(obj: Map<any, any>): ArrayBuffer {
  const message = new capnp.Message();
  const rs = message.initRoot(KVIterable);
  const keys = rs.initKeys(obj.size);
  const values = rs.initValues(obj.size);
  let count = 0;
  obj.forEach((value: any, key: any) => {
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
  });
  return message.toArrayBuffer();
}

function deserializeMap(buffer: ArrayBuffer): Record<any, any> {
  // Initialize an empty object to store the key-value pairs.
  const kv_iter: Record<any, any> = {};

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
    if (Object.prototype.toString.call(deserializedKey) === '[object Object]') {
      keyObj = JSON.stringify(deserializedKey);
    } else {
      keyObj = deserializedKey;
    }
    // Store the processed object as a value corresponding to the deserialized key in the kv_iter object.
    kv_iter[keyObj] = obj;
  }
  return kv_iter;
}
