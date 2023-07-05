import * as capnp from 'capnp-ts';

import { RecursiveSerde } from '../capnp/recursive_serde.capnp';

import { PRIMITIVE_MAP, PRIMITIVES } from './primitives';

function mergeChunks(chunks: capnp.List<capnp.Data>) {
  let totalSize = 0;

  // Calculate the total size of all the chunks
  for (let i = 0; i < chunks.getLength(); i++) {
    totalSize += chunks.get(i).getLength();
  }

  // Create a new array with the total size
  const tmp = new Uint8Array(totalSize);
  let position = 0;

  // Fill the new array with the data from the chunks
  for (let i = 0; i < chunks.getLength(); i++) {
    const chunkData = deserialize(chunks.get(i).toArrayBuffer());
    const dataChunk = new Uint8Array(chunkData);
    tmp.set(dataChunk, position);
    position += dataChunk.byteLength;
  }

  return tmp;
}

export function deserialize(buffer: ArrayBuffer) {
  const message = new capnp.Message(buffer, false);
  const rs = message.getRoot(RecursiveSerde);
  const fqn = rs.getFullyQualifiedName();

  const serde_obj = PRIMITIVES[PRIMITIVE_MAP.get(fqn, false)];

  if (serde_obj) {
    // If the data is a blob, deserialize the blob and return it
    const blob = rs.getNonrecursiveBlob();
    if (blob.getLength() === 1) {
      return serde_obj.deserialize(blob.get(0).toArrayBuffer());
    } else {
      const totalChunk = mergeChunks(blob);
      return serde_obj.deserialize(totalChunk.buffer);
    }
  }
}
