"use strict";
/* tslint:disable */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Foo = exports._capnpFileId = void 0;
const capnp_ts_1 = require("capnp-ts");
exports._capnpFileId = BigInt("0xe0b7ff464fbc7ee1");
class Foo extends capnp_ts_1.Struct {
    getBar() { return capnp_ts_1.Struct.getText(0, this); }
    setBar(value) { capnp_ts_1.Struct.setText(0, value, this); }
    toString() { return "Foo_" + super.toString(); }
}
exports.Foo = Foo;
Foo._capnp = { displayName: "Foo", id: "9e8e6186e9348688", size: new capnp_ts_1.ObjectSize(0, 1) };
