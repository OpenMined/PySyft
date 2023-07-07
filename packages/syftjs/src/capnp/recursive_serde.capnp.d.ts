/**
 * This file has been automatically generated by the [capnpc-ts utility](https://github.com/jdiaz5513/capnp-ts).
 */
import * as capnp from "capnp-ts";
import { Struct as __S } from "capnp-ts";
export declare const _capnpFileId: bigint;
export declare class RecursiveSerde extends __S {
  static readonly _capnp: {
    displayName: string;
    id: string;
    size: capnp.ObjectSize;
  };
  static _FieldsData: capnp.ListCtor<capnp.List<capnp.Data>>;
  adoptFieldsName(value: capnp.Orphan<capnp.List<string>>): void;
  disownFieldsName(): capnp.Orphan<capnp.List<string>>;
  getFieldsName(): capnp.List<string>;
  hasFieldsName(): boolean;
  initFieldsName(length: number): capnp.List<string>;
  setFieldsName(value: capnp.List<string>): void;
  adoptFieldsData(
    value: capnp.Orphan<capnp.List<capnp.List<capnp.Data>>>,
  ): void;
  disownFieldsData(): capnp.Orphan<capnp.List<capnp.List<capnp.Data>>>;
  getFieldsData(): capnp.List<capnp.List<capnp.Data>>;
  hasFieldsData(): boolean;
  initFieldsData(length: number): capnp.List<capnp.List<capnp.Data>>;
  setFieldsData(value: capnp.List<capnp.List<capnp.Data>>): void;
  getFullyQualifiedName(): string;
  setFullyQualifiedName(value: string): void;
  adoptNonrecursiveBlob(value: capnp.Orphan<capnp.List<capnp.Data>>): void;
  disownNonrecursiveBlob(): capnp.Orphan<capnp.List<capnp.Data>>;
  getNonrecursiveBlob(): capnp.List<capnp.Data>;
  hasNonrecursiveBlob(): boolean;
  initNonrecursiveBlob(length: number): capnp.List<capnp.Data>;
  setNonrecursiveBlob(value: capnp.List<capnp.Data>): void;
  toString(): string;
}
