import typescript from "@rollup/plugin-typescript";
import resolve from "@rollup/plugin-node-resolve";
import babel from "@rollup/plugin-babel";

export default {
  input: "src/index.ts",
  output: {
    dir: "lib/",
    format: "cjs",
  },
  external: ["capnp-ts", "uuid"],
  plugins: [typescript(), resolve(), babel({ babelHelpers: "bundled" })],
};
