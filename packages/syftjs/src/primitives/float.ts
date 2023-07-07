import { PrimitiveInterface } from "./primitive_interface";

export const FLOAT: PrimitiveInterface = {
  serialize: serializeFloat,
  deserialize: deserializeFloat,
  fqn: "builtins.float",
};

function serializeFloat(obj: number) {
  let hex_str = "";

  if (obj < 0) {
    hex_str += "-";
    obj = obj * -1;
  }
  hex_str += "0x";
  const exponent = 10;

  const n_1 = obj / 2 ** exponent;
  hex_str += Math.trunc(n_1).toString(16);
  hex_str += ".";

  let notZeros = true;
  let currNumber = n_1;
  while (notZeros) {
    currNumber = (currNumber % 1) * 16;
    hex_str += Math.trunc(currNumber).toString(16);
    if (currNumber % 1 === 0) {
      notZeros = false;
    }
  }
  hex_str += "p" + exponent;
  return new TextEncoder().encode(hex_str);
}

function deserializeFloat(buffer: ArrayBuffer) {
  const hex_str = new TextDecoder().decode(buffer);
  let aggr = 0;
  const [signal, int_n, hex_dec_n, exp] = hex_str
    .replaceAll(".", " ")
    .replaceAll("0x", " ")
    .replaceAll("p", " ")
    .split(" ");
  aggr += parseInt(int_n, 16);

  let n_signal: number;
  if (signal) {
    n_signal = -1;
  } else {
    n_signal = 1;
  }
  // bracket notation
  for (let i = 0; i < hex_dec_n.length; i++) {
    aggr += parseInt(hex_dec_n[i], 16) / 16.0 ** (i + 1);
  }
  return aggr * 2 ** parseInt(exp, 10) * n_signal;
}
