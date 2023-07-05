export const PRIMITIVE_MAP = new Map<string, string>([
  ['builtins.str', 'string'],
]);

export const PRIMITIVES = {
  string: {
    serialize: (text: string) => {
      return new TextEncoder().encode(text);
    },
    deserialize: (buffer: ArrayBuffer) => {
      return new TextDecoder().decode(buffer);
    },
    fqn: 'builtins.str',
  },
};
