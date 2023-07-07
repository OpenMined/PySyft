import * as capnp from "capnp-ts";

import { DataBox } from "../capnp/databox.capnp";
import { DataList } from "../capnp/datalist.capnp";

export function splitChunks(serializedObj: ArrayBuffer) {
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

export function createData(length: number) {
  const newDataMsg = new capnp.Message();
  const dataRoot = newDataMsg.initRoot(DataBox);
  dataRoot.initValue(length);
  return dataRoot.getValue();
}

export function createDataList(length: number) {
  const newDataList = new capnp.Message();
  const dataListRoot = newDataList.initRoot(DataList);
  dataListRoot.initValues(length);
  return dataListRoot.getValues();
}

export function serializeChunks(chunks: ArrayBuffer[]) {
  const dataList = createDataList(chunks.length);

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const dataStruct = createData(chunk.byteLength);
    dataStruct.copyBuffer(chunk);
    dataList.set(i, dataStruct);
  }

  return dataList;
}
