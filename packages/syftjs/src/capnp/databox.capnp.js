"use strict";
/* tslint:disable */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DataBox = exports._capnpFileId = void 0;
const capnp_ts_1 = require("capnp-ts");
exports._capnpFileId = BigInt("0x915f3367709de788");
export class DataBox extends capnp_ts_1.Struct {
    adoptValue(value) { capnp_ts_1.Struct.adopt(value, capnp_ts_1.Struct.getPointer(0, this)); }
    disownValue() { return capnp_ts_1.Struct.disown(this.getValue()); }
    getValue() { return capnp_ts_1.Struct.getData(0, this); }
    hasValue() { return !capnp_ts_1.Struct.isNull(capnp_ts_1.Struct.getPointer(0, this)); }
    initValue(length) { return capnp_ts_1.Struct.initData(0, length, this); }
    setValue(value) { capnp_ts_1.Struct.copyFrom(value, capnp_ts_1.Struct.getPointer(0, this)); }
    toString() { return "DataBox_" + super.toString(); }
}
exports.DataBox = DataBox;
DataBox._capnp = { displayName: "DataBox", id: "e21ed6551e7a8638", size: new capnp_ts_1.ObjectSize(0, 1) };
