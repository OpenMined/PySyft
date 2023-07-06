import * as capnp from 'capnp-ts';

import { Iterable } from '../capnp/iterable.capnp';
import { deserialize } from '../serde/deserialize';
import { mergeChunks } from '../serde/serializable';
import { serialize } from '../serde/serialize';
import { serializeChunks, splitChunks } from '../serde/utils';

import { PrimitiveInterface } from './primitive_interface';

export const ARRAY: PrimitiveInterface = {
  serialize: serializeArray,
  deserialize: deserializeArray,
  fqn: 'builtins.list',
};

function serializeArray(obj: any[]) {
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
}

function deserializeArray(buffer: ArrayBuffer) {
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
}
