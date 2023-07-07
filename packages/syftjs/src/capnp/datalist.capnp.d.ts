/**
 * This file has been automatically generated by the [capnpc-ts utility](https://github.com/jdiaz5513/capnp-ts).
 */
import * as capnp from "capnp-ts";
import { Struct as __S } from "capnp-ts";
export declare const _capnpFileId: bigint;
export declare class DataList extends __S {
  static readonly _capnp: {
    displayName: string;
    id: string;
    size: capnp.ObjectSize;
  };
  adoptValues(value: capnp.Orphan<capnp.List<capnp.Data>>): void;
  disownValues(): capnp.Orphan<capnp.List<capnp.Data>>;
  getValues(): capnp.List<capnp.Data>;
  hasValues(): boolean;
  initValues(length: number): capnp.List<capnp.Data>;
  setValues(value: capnp.List<capnp.Data>): void;
  toString(): string;
}
