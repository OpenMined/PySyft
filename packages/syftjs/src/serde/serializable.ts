import * as capnp from 'capnp-ts';

import { RecursiveSerde } from '../capnp/recursive_serde.capnp';
import * as SYFT_OBJECT_TYPES from '../objects';
import * as PRIMITIVE_TYPES from '../primitives';

import { deserialize } from './deserialize';
import { serialize } from './serialize';
import { splitChunks } from './utils';
import { createData } from './utils';
import { serializeChunks } from './utils';

export function serializeObject(
  obj: any,
  serde_schema: any,
  rs: RecursiveSerde,
) {
  if (obj !== serde_schema) {
    return SERIALIZABLE.PRIMITIVE.serialize(obj, rs, serde_schema.serialize);
  } else {
    return SERIALIZABLE.RECURSIVE.serialize(obj, rs);
  }
}

export function deserializeObject(fqn: string, rs: RecursiveSerde) {
  const serde_schema = getSerdeSchemaByFqn(fqn);
  if (serde_schema) {
    return SERIALIZABLE.PRIMITIVE.deserialize(rs, serde_schema.deserialize);
  } else {
    return SERIALIZABLE.RECURSIVE.deserialize(rs);
  }
}

export function getSerdeSchema(obj: any) {
  if (typeof obj !== 'undefined' && obj !== null) {
    switch (obj.constructor) {
      case Number:
        return Number.isInteger(obj)
          ? PRIMITIVE_TYPES.INTEGER
          : PRIMITIVE_TYPES.FLOAT;
      case String:
        return PRIMITIVE_TYPES.STRING;
      case Boolean:
        return PRIMITIVE_TYPES.BOOLEAN;
      case Array:
        return PRIMITIVE_TYPES.ARRAY;
      case Map:
        return PRIMITIVE_TYPES.MAP;
      case Uint8Array:
        return PRIMITIVE_TYPES.BYTES;
      default:
        return obj;
    }
  } else {
    return obj;
  }
}

export function getSerdeSchemaByFqn(fqn: string) {
  switch (fqn) {
    case PRIMITIVE_TYPES.INTEGER.fqn:
      return PRIMITIVE_TYPES.INTEGER;
    case PRIMITIVE_TYPES.FLOAT.fqn:
      return PRIMITIVE_TYPES.FLOAT;
    case PRIMITIVE_TYPES.STRING.fqn:
      return PRIMITIVE_TYPES.STRING;
    case PRIMITIVE_TYPES.BOOLEAN.fqn:
      return PRIMITIVE_TYPES.BOOLEAN;
    case PRIMITIVE_TYPES.ARRAY.fqn:
      return PRIMITIVE_TYPES.ARRAY;
    case PRIMITIVE_TYPES.MAP.fqn:
      return PRIMITIVE_TYPES.MAP;
    case PRIMITIVE_TYPES.BYTES.fqn:
      return PRIMITIVE_TYPES.BYTES;
    default:
      return null;
  }
}

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

const SERIALIZABLE = {
  PRIMITIVE: {
    serialize: primitiveSerialization,
    deserialize: primitiveDeserialization,
  },
  RECURSIVE: {
    serialize: recursiveSerialization,
    deserialize: recursiveDeserialization,
  },
};

function primitiveSerialization(
  obj: any,
  rs: RecursiveSerde,
  serializer: (obj: any) => ArrayBuffer,
) {
  // Serialize the object using the specified serializer function
  const serializedObj = serializer(obj);
  // Split the serialized object into chunks and initialize the data field
  const chunks = splitChunks(serializedObj);
  const data = rs.initNonrecursiveBlob(chunks.length);

  // Copy each chunk into a new payload and set it in the data field
  for (let i = 0; i < chunks.length; i++) {
    const payload = createData(chunks[i].byteLength);
    payload.copyBuffer(chunks[i]);
    data.set(i, payload);
  }
  return rs;
}

function primitiveDeserialization(
  rs: RecursiveSerde,
  deserializer: (buffer: ArrayBuffer) => any,
) {
  const blob = rs.getNonrecursiveBlob();
  const totalChunk = mergeChunks(blob);
  return deserializer(totalChunk);
}

function recursiveSerialization(obj: object, rs: RecursiveSerde) {
  // Remove fqn obj property to avoid serializing it as a valid field attribute.
  const { fqn, ...newObj }: any = obj;

  // Initialize the fields for the object's text and data
  const txt = rs.initFieldsName(Object.keys(newObj).length);
  const data = rs.initFieldsData(Object.keys(newObj).length);

  // Loop over each property of the object
  let count = 0;
  for (const attr in newObj) {
    // Serialize the property's value and store it in the Cap'n Proto message
    txt.set(count, attr);
    const serializedObj = serialize(newObj[attr]);
    const chunks = splitChunks(serializedObj);
    const chunkList = serializeChunks(chunks);
    data.set(count, chunkList);
    count += 1;
  }
  return rs;
}

function recursiveDeserialization(rs: RecursiveSerde) {
  // If the data is a structured object, deserialize its fields into a map
  const fieldsName = rs.getFieldsName();
  const fieldsData = rs.getFieldsData();
  const fqn = rs.getFullyQualifiedName();
  const kvIterable: Record<string, any> = {};
  // Check if the number of fields in the object matches the number of field names
  if (fieldsData.getLength() !== fieldsName.getLength()) {
    console.log('Mismatch between Fields Data and Fields Name!!');
  } else {
    // Iterate over the fields in the object and deserialize their values
    for (let i = 0; i < fieldsName.getLength(); i++) {
      const key = fieldsName.get(i); // Get the name of the current field
      const bytes = fieldsData.get(i); // Get the binary data buffer for the current field

      // IF bytes are huge, merge everything in a single chunk and deserialize it recursively.
      const totalChunk = mergeChunks(bytes);
      const obj = deserialize(totalChunk);
      kvIterable[key] = obj; // Add the deserialized value to the key-value iterable
    }
  }
  let syftClass = OBJ_MAP.get(fqn);
  if (syftClass) {
    const objInstance = new syftClass();
    Object.assign(objInstance, kvIterable);
    return objInstance;
  } else {
    kvIterable['fqn'] = fqn;
    return kvIterable;
  }
}

export const OBJ_MAP = new Map();

Object.keys(SYFT_OBJECT_TYPES).forEach((key) => {
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore
  const obj = SYFT_OBJECT_TYPES[key];
  OBJ_MAP.set(obj.classFqn, obj);
});
