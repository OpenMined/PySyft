export interface PrimitiveInterface {
  serialize: (obj: any) => ArrayBuffer;
  deserialize: (buffer: ArrayBuffer) => any;
  fqn: string;
}
