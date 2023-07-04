import * as capnp from 'capnp-ts';

import { DataBox } from '../capnp/databox.capnp';
import { DataList } from '../capnp/datalist.capnp';
import { Iterable } from '../capnp/iterable.capnp';

function createData(length: number) {
  const newDataMsg = new capnp.Message();
  const dataRoot = newDataMsg.initRoot(DataBox);
  dataRoot.initValue(length);
  return dataRoot.getValue();
}

/**
 * Splits a serialized data object into chunks of a maximum size.
 *
 * @param {DataObject} serializedObj - The serialized data object to split.
 * @returns {ArrayBuffer[]} An array of binary data chunks.
 */
function splitChunks(serializedObj: Uint8Array) {
  const sizeLimit = 5.12 ** 8;
  const chunks = [];
  let pointer = 0;

  if (serializedObj.byteLength <= sizeLimit) {
    // If the serialized object is smaller than the size limit, add it as a single chunk
    chunks.push(serializedObj);
  } else {
    // If the serialized object is larger than the size limit, split it into multiple chunks
    const numSlices = Math.ceil(serializedObj.byteLength / sizeLimit);
    for (let i = 0; i < numSlices - 1; i++) {
      // Push a slice of the serialized object to the chunks array
      chunks.push(serializedObj.slice(pointer, pointer + sizeLimit));
      pointer += sizeLimit;
    }
    // Push the last slice to the chunks array
    chunks.push(serializedObj.slice(pointer));
  }

  return chunks;
}

function createDataList(length: number) {
  const newDataList = new capnp.Message();
  const dataListRoot = newDataList.initRoot(DataList);
  dataListRoot.initValues(length);
  return dataListRoot.getValues();
}

/**
 * Serializes an array of binary data chunks into a data list.
 *
 * @param {ArrayBuffer[]} chunks - An array of binary data chunks.
 * @returns {DataList} A serialized data list.
 */
function serializeChunks(chunks: any) {
  // Create a new data list with a length equal to the number of input chunks
  const dataList = createDataList(chunks.length);

  // Iterate over each chunk and add it to the data list
  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];

    // Create a new data structure with the same length as the current chunk
    const dataStruct = createData(chunk.byteLength);

    // Copy the contents of the current chunk into the data structure
    dataStruct.copyBuffer(chunk);

    // Add the data structure to the data list
    dataList.set(i, dataStruct);
  }

  return dataList;
}

export const PRIMITIVES = {
  string: {
    serialize: (text: string) => {
      return new TextEncoder().encode(text);
    },
    deserialize: (buffer: ArrayBuffer) => {
      return new TextDecoder().decode(buffer);
    },
  },

  list: {
    serialize: (list: any[]) => {
      const message = new capnp.Message();
      const rs = message.initRoot(Iterable);
      const listStruct = rs.initValues(list.length);
      let count = 0;
      for (let index = 0; index < list.length; index++) {
        const serializedObj = PRIMITIVES.string.serialize(list[index]);
        const chunks = splitChunks(serializedObj);
        const chunkList = serializeChunks(chunks);
        listStruct.set(count, chunkList);
        count += 1;
      }
      return message.toArrayBuffer();
    },
  },
};
