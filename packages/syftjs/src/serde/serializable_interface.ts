export interface SerializableInterface {
  serialize: (obj: any) => ArrayBuffer;
  deserialize: (buffer: ArrayBuffer) => any;
  fqn: string;
}
