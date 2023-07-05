import * as capnp from 'capnp-ts';

import { RecursiveSerde } from '../capnp/recursive_serde.capnp';

import { getPrimitiveByFqn } from './primitives';

export function mergeChunks(chunks: capnp.List<capnp.Data>) {
  if (chunks.getLength() === 1) {
    return chunks.get(0).toArrayBuffer();
  }

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

  return tmp.buffer;
}

export function deserialize(buffer: ArrayBuffer) {
  const message = new capnp.Message(buffer, false);
  const rs = message.getRoot(RecursiveSerde);
  const fqn = rs.getFullyQualifiedName();

  const serde_obj = getPrimitiveByFqn(fqn);

  if (serde_obj) {
    // If the data is a blob, deserialize the blob and return it
    const blob = rs.getNonrecursiveBlob();

    const totalChunk = mergeChunks(blob);
    return serde_obj.deserialize(totalChunk);
  }
}
