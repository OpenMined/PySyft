export interface SerializableInterface {
  serialize: (obj: SerializableInterface) => Uint8Array;
  deserialize: (buffer: ArrayBuffer) => SerializableInterface;
}
