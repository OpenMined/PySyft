(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.converter = f()}})(function(){var define,module,exports;return (function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
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

},{"capnp-ts":5}],2:[function(require,module,exports){
// const root = require('./proto/tensor.proto')
// const tensor_pb = root.lookupType('syft.Tensor')
// console.log("tensor_pb", tensor_pb)

const capnp = require("capnp-ts");

const Foo = require("./address.capnp.js").Foo;

// lets us play in the browser console
window.capnp = capnp
window.Foo = Foo



const API_URL = "http://127.0.0.1:5001"



// const oldMessage = new capnp.Message();
// const oldFoo = oldMessage.initRoot(Foo);

// oldFoo.setBar("bar");


// const packed = Buffer.from(oldMessage.toPackedArrayBuffer());

// // const newMessage = new capnp.Message(packed);
// // newMessage.getRoot(NewFoo);

// t.pass("should not 💩 the 🛏");


function deserialize(buffer, struct) {
  window.buffer = buffer
  window.struct = struct

  console.log("trying to deserialize", buffer, struct)
  // needs false to say bytes are not packed
  const message = new capnp.Message(buffer, false);
  return message.getRoot(struct);
}

function make_foo(bar) {
    const foo = new capnp.Message().initRoot(Foo)
    foo.setBar(bar);
    console.log(foo)
    return foo
}

function serialize(obj) {

}

async function run() {
    // get protobuf message bytes from server
    const response = await fetch(`${API_URL}/rcv`)
    const bytes = await response.arrayBuffer()
    console.log(
        "bytes", bytes
    )

    const server_foo = deserialize(bytes, Foo)
    console.log("server_foo", server_foo, server_foo.bar)

    const client_foo = make_foo(`${server_foo.getBar()} updated`)
    console.log(client_foo, client_foo.bar)

    // lets us play around in the browser console
    window.foo = client_foo

    client_bytes = client_foo.segment.message.toArrayBuffer()
    console.log("client_bytes", client_bytes)
    // capnp.util.dumpBuffer(client_foo.segment)

    const response2 = await fetch(`${API_URL}/send`, {
        method: "POST",
        headers: {"content-type": "application/octect-stream"},
        body: client_bytes,
    })
    console.log("finished posting", response2)

}
run()

},{"./address.capnp.js":1,"capnp-ts":5}],3:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.MAX_SEGMENT_LENGTH = exports.MAX_DEPTH = exports.VAL32 = exports.POINTER_TYPE_MASK = exports.POINTER_DOUBLE_FAR_MASK = exports.POINTER_COPY_LIMIT = exports.PACK_SPAN_THRESHOLD = exports.NATIVE_LITTLE_ENDIAN = exports.MIN_SINGLE_SEGMENT_GROWTH = exports.MIN_SAFE_INTEGER = exports.MAX_STREAM_SEGMENTS = exports.MAX_SAFE_INTEGER = exports.MAX_UINT32 = exports.MAX_INT32 = exports.MAX_BUFFER_DUMP_BYTES = exports.LIST_SIZE_MASK = exports.GROWTH_FACTOR = exports.DEFAULT_TRAVERSE_LIMIT = exports.DEFAULT_DEPTH_LIMIT = exports.DEFAULT_DECODE_LIMIT = exports.DEFAULT_BUFFER_SIZE = void 0;
// Perform some bit gymnastics to determine the native endian format.
const tmpWord = new DataView(new ArrayBuffer(8));
new Uint16Array(tmpWord.buffer)[0] = 0x0102;
/** Default size (in bytes) for newly allocated segments. */
exports.DEFAULT_BUFFER_SIZE = 4096;
exports.DEFAULT_DECODE_LIMIT = 64 << 20; // 64 MiB
/**
 * Limit to how deeply nested pointers are allowed to be. The root struct of a message will start at this value, and it
 * is decremented as pointers are dereferenced.
 */
exports.DEFAULT_DEPTH_LIMIT = 64;
/**
 * Limit to the number of **bytes** that can be traversed in a single message. This is necessary to prevent certain
 * classes of DoS attacks where maliciously crafted data can be self-referencing in a way that wouldn't trigger the
 * depth limit.
 *
 * For this reason, it is advised to cache pointers into variables and not constantly dereference them since the
 * message's traversal limit gets decremented each time.
 */
exports.DEFAULT_TRAVERSE_LIMIT = 64 << 20; // 64 MiB
/**
 * When allocating array buffers dynamically (while packing or in certain Arena implementations) the previous buffer's
 * size is multiplied by this number to determine the next buffer's size. This is chosen to keep both time spent
 * reallocating and wasted memory to a minimum.
 *
 * Smaller numbers would save memory at the expense of CPU time.
 */
exports.GROWTH_FACTOR = 1.5;
/** A bitmask applied to obtain the size of a list pointer. */
exports.LIST_SIZE_MASK = 0x00000007;
/** Maximum number of bytes to dump at once when dumping array buffers to string. */
exports.MAX_BUFFER_DUMP_BYTES = 8192;
/** The maximum value for a 32-bit integer. */
exports.MAX_INT32 = 0x7fffffff;
/** The maximum value for a 32-bit unsigned integer. */
exports.MAX_UINT32 = 0xffffffff;
/** The largest integer that can be precisely represented in JavaScript. */
exports.MAX_SAFE_INTEGER = 9007199254740991;
/** Maximum limit on the number of segments in a message stream. */
exports.MAX_STREAM_SEGMENTS = 512;
/** The smallest integer that can be precisely represented in JavaScript. */
exports.MIN_SAFE_INTEGER = -9007199254740991;
/** Minimum growth increment for a SingleSegmentArena. */
exports.MIN_SINGLE_SEGMENT_GROWTH = 4096;
/**
 * This will be `true` if the machine running this code stores numbers natively in little-endian format. This is useful
 * for some numeric type conversions when the endianness does not affect the output. Using the native endianness for
 * these operations is _slightly_ faster.
 */
exports.NATIVE_LITTLE_ENDIAN = tmpWord.getUint8(0) === 0x02;
/**
 * When packing a message, this is the number of zero bytes required after a SPAN (0xff) tag is written to the packed
 * message before the span is terminated.
 *
 * This little detail is left up to the implementation because it can be tuned for performance. Setting this to a higher
 * value may help with messages that contain a ton of text/data.
 *
 * It is imperative to never set this below 1 or else BAD THINGS. You have been warned.
 */
exports.PACK_SPAN_THRESHOLD = 2;
/**
 * How far to travel into a nested pointer structure during a deep copy; when this limit is exhausted the copy
 * operation will throw an error.
 */
exports.POINTER_COPY_LIMIT = 32;
/** A bitmask for looking up the double-far flag on a far pointer. */
exports.POINTER_DOUBLE_FAR_MASK = 0x00000004;
/** A bitmask for looking up the pointer type. */
exports.POINTER_TYPE_MASK = 0x00000003;
/** Used for some 64-bit conversions, equal to Math.pow(2, 32). */
exports.VAL32 = 0x100000000;
/** The maximum value allowed for depth traversal limits. */
exports.MAX_DEPTH = exports.MAX_INT32;
/** The maximum byte length for a single segment. */
exports.MAX_SEGMENT_LENGTH = exports.MAX_UINT32;

},{}],4:[function(require,module,exports){
"use strict";
/**
 * This file contains all the error strings used in the library. Also contains silliness.
 *
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PTR_WRITE_CONST_STRUCT = exports.PTR_WRITE_CONST_LIST = exports.TYPE_SET_GENERIC_LIST = exports.TYPE_GET_GENERIC_LIST = exports.TYPE_COMPOSITE_SIZE_UNDEFINED = exports.SEG_SIZE_OVERFLOW = exports.SEG_REPLACEMENT_BUFFER_TOO_SMALL = exports.SEG_NOT_WORD_ALIGNED = exports.SEG_ID_OUT_OF_BOUNDS = exports.SEG_GET_NON_ZERO_SINGLE = exports.SEG_BUFFER_NOT_ALLOCATED = exports.RANGE_UINT32_OVERFLOW = exports.RANGE_SIZE_OVERFLOW = exports.RANGE_INVALID_UTF8 = exports.RANGE_INT64_UNDERFLOW = exports.RANGE_INT32_OVERFLOW = exports.PTR_WRONG_STRUCT_PTR_SIZE = exports.PTR_WRONG_STRUCT_DATA_SIZE = exports.PTR_WRONG_COMPOSITE_PTR_SIZE = exports.PTR_WRONG_COMPOSITE_DATA_SIZE = exports.PTR_WRONG_POINTER_TYPE = exports.PTR_WRONG_LIST_TYPE = exports.PTR_TRAVERSAL_LIMIT_EXCEEDED = exports.PTR_STRUCT_POINTER_OUT_OF_BOUNDS = exports.PTR_STRUCT_DATA_OUT_OF_BOUNDS = exports.PTR_OFFSET_OUT_OF_BOUNDS = exports.PTR_INVALID_UNION_ACCESS = exports.PTR_INVALID_POINTER_TYPE = exports.PTR_INVALID_LIST_SIZE = exports.PTR_INVALID_FAR_TARGET = exports.PTR_INIT_NON_GROUP = exports.PTR_INIT_COMPOSITE_STRUCT = exports.PTR_DISOWN_COMPOSITE_STRUCT = exports.PTR_DEPTH_LIMIT_EXCEEDED = exports.PTR_COMPOSITE_SIZE_UNDEFINED = exports.PTR_ALREADY_ADOPTED = exports.PTR_ADOPT_WRONG_MESSAGE = exports.PTR_ADOPT_COMPOSITE_STRUCT = exports.NOT_IMPLEMENTED = exports.MSG_SEGMENT_TOO_SMALL = exports.MSG_SEGMENT_OUT_OF_BOUNDS = exports.MSG_PACK_NOT_WORD_ALIGNED = exports.MSG_NO_SEGMENTS_IN_ARENA = exports.MSG_INVALID_FRAME_HEADER = exports.assertNever = exports.INVARIANT_UNREACHABLE_CODE = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("./constants");
const trace = debug_1.default("capnp:errors");
trace("load");
// Invariant violations (sometimes known as "precondition failed").
//
// All right, hold up the brakes. This is a serious 1 === 0 WHAT THE FAILURE moment here. Tell the SO's you won't be
// home for dinner.
exports.INVARIANT_UNREACHABLE_CODE = "CAPNP-TS000 Unreachable code detected.";
function assertNever(n) {
    // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
    throw new Error(exports.INVARIANT_UNREACHABLE_CODE + ` (never block hit with: ${n})`);
}
exports.assertNever = assertNever;
// Message errors.
//
// Now who told you it would be a good idea to fuzz the inputs? You just made the program sad.
exports.MSG_INVALID_FRAME_HEADER = "CAPNP-TS001 Attempted to parse an invalid message frame header; are you sure this is a Cap'n Proto message?";
exports.MSG_NO_SEGMENTS_IN_ARENA = "CAPNP-TS002 Attempted to preallocate a message with no segments in the arena.";
exports.MSG_PACK_NOT_WORD_ALIGNED = "CAPNP-TS003 Attempted to pack a message that was not word-aligned.";
exports.MSG_SEGMENT_OUT_OF_BOUNDS = "CAPNP-TS004 Segment ID %X is out of bounds for message %s.";
exports.MSG_SEGMENT_TOO_SMALL = "CAPNP-TS005 First segment must have at least enough room to hold the root pointer (8 bytes).";
// Used for methods that are not yet implemented.
//
// My bad. I'll get to it. Eventually.
exports.NOT_IMPLEMENTED = "CAPNP-TS006 %s is not implemented.";
// Pointer-related errors.
//
// Look, this is probably the hardest part of the code. Cut some slack here! You probably found a bug.
exports.PTR_ADOPT_COMPOSITE_STRUCT = "CAPNP-TS007 Attempted to adopt a struct into a composite list (%s).";
exports.PTR_ADOPT_WRONG_MESSAGE = "CAPNP-TS008 Attempted to adopt %s into a pointer in a different message %s.";
exports.PTR_ALREADY_ADOPTED = "CAPNP-TS009 Attempted to adopt %s more than once.";
exports.PTR_COMPOSITE_SIZE_UNDEFINED = "CAPNP-TS010 Attempted to set a composite list without providing a composite element size.";
exports.PTR_DEPTH_LIMIT_EXCEEDED = "CAPNP-TS011 Nesting depth limit exceeded for %s.";
exports.PTR_DISOWN_COMPOSITE_STRUCT = "CAPNP-TS012 Attempted to disown a struct member from a composite list (%s).";
exports.PTR_INIT_COMPOSITE_STRUCT = "CAPNP-TS013 Attempted to initialize a struct member from a composite list (%s).";
exports.PTR_INIT_NON_GROUP = "CAPNP-TS014 Attempted to initialize a group field with a non-group struct class.";
exports.PTR_INVALID_FAR_TARGET = "CAPNP-TS015 Target of a far pointer (%s) is another far pointer.";
exports.PTR_INVALID_LIST_SIZE = "CAPNP-TS016 Invalid list element size: %x.";
exports.PTR_INVALID_POINTER_TYPE = "CAPNP-TS017 Invalid pointer type: %x.";
exports.PTR_INVALID_UNION_ACCESS = "CAPNP-TS018 Attempted to access getter on %s for union field %s that is not currently set (wanted: %d, found: %d).";
exports.PTR_OFFSET_OUT_OF_BOUNDS = "CAPNP-TS019 Pointer offset %a is out of bounds for underlying buffer.";
exports.PTR_STRUCT_DATA_OUT_OF_BOUNDS = "CAPNP-TS020 Attempted to access out-of-bounds struct data (struct: %s, %d bytes at %a, data words: %d).";
exports.PTR_STRUCT_POINTER_OUT_OF_BOUNDS = "CAPNP-TS021 Attempted to access out-of-bounds struct pointer (%s, index: %d, length: %d).";
exports.PTR_TRAVERSAL_LIMIT_EXCEEDED = "CAPNP-TS022 Traversal limit exceeded! Slow down! %s";
exports.PTR_WRONG_LIST_TYPE = "CAPNP-TS023 Cannot convert %s to a %s list.";
exports.PTR_WRONG_POINTER_TYPE = "CAPNP-TS024 Attempted to convert pointer %s to a %s type.";
exports.PTR_WRONG_COMPOSITE_DATA_SIZE = "CAPNP-TS025 Attempted to convert %s to a composite list with the wrong data size (found: %d).";
exports.PTR_WRONG_COMPOSITE_PTR_SIZE = "CAPNP-TS026 Attempted to convert %s to a composite list with the wrong pointer size (found: %d).";
exports.PTR_WRONG_STRUCT_DATA_SIZE = "CAPNP-TS027 Attempted to convert %s to a struct with the wrong data size (found: %d).";
exports.PTR_WRONG_STRUCT_PTR_SIZE = "CAPNP-TS028 Attempted to convert %s to a struct with the wrong pointer size (found: %d).";
// Custom error messages for the built-in `RangeError` class.
//
// You don't get a witty comment with these.
exports.RANGE_INT32_OVERFLOW = "CAPNP-TS029 32-bit signed integer overflow detected.";
exports.RANGE_INT64_UNDERFLOW = "CAPNP-TS030 Buffer is not large enough to hold a word.";
exports.RANGE_INVALID_UTF8 = "CAPNP-TS031 Invalid UTF-8 code sequence detected.";
exports.RANGE_SIZE_OVERFLOW = `CAPNP-TS032 Size %x exceeds maximum ${constants_1.MAX_SEGMENT_LENGTH.toString(16)}.`;
exports.RANGE_UINT32_OVERFLOW = "CAPNP-TS033 32-bit unsigned integer overflow detected.";
// Segment-related errors.
//
// These suck. Deal with it.
exports.SEG_BUFFER_NOT_ALLOCATED = "CAPNP-TS034 allocate() needs to be called at least once before getting a buffer.";
exports.SEG_GET_NON_ZERO_SINGLE = "CAPNP-TS035 Attempted to get a segment other than 0 (%d) from a single segment arena.";
exports.SEG_ID_OUT_OF_BOUNDS = "CAPNP-TS036 Attempted to get an out-of-bounds segment (%d).";
exports.SEG_NOT_WORD_ALIGNED = "CAPNP-TS037 Segment buffer length %d is not a multiple of 8.";
exports.SEG_REPLACEMENT_BUFFER_TOO_SMALL = "CAPNP-TS038 Attempted to replace a segment buffer with one that is smaller than the allocated space.";
exports.SEG_SIZE_OVERFLOW = `CAPNP-TS039 Requested size %x exceeds maximum value (${constants_1.MAX_SEGMENT_LENGTH}).`;
// Custom error messages for the built-in `TypeError` class.
//
// If it looks like a duck, quacks like an elephant, and has hooves for feet, it's probably JavaScript.
exports.TYPE_COMPOSITE_SIZE_UNDEFINED = "CAPNP-TS040 Must provide a composite element size for composite list pointers.";
exports.TYPE_GET_GENERIC_LIST = "CAPNP-TS041 Attempted to call get() on a generic list.";
exports.TYPE_SET_GENERIC_LIST = "CAPNP-TS042 Attempted to call set() on a generic list.";
exports.PTR_WRITE_CONST_LIST = "CAPNP-TS043 Attempted to write to a const list.";
exports.PTR_WRITE_CONST_STRUCT = "CAPNP-TS044 Attempted to write to a const struct.";

},{"./constants":3,"debug":52,"tslib":56}],5:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Uint64 = exports.Int64 = exports.getUint8Mask = exports.getUint64Mask = exports.getUint32Mask = exports.getUint16Mask = exports.getInt8Mask = exports.getInt64Mask = exports.getInt32Mask = exports.getInt16Mask = exports.getFloat64Mask = exports.getFloat32Mask = exports.getBitMask = exports.Void = exports.VoidList = exports.Uint8List = exports.Uint64List = exports.Uint32List = exports.Uint16List = exports.TextList = exports.Text = exports.Struct = exports.Pointer = exports.PointerType = exports.PointerList = exports.Orphan = exports.List = exports.InterfaceList = exports.Interface = exports.Int8List = exports.Int64List = exports.Int32List = exports.Int16List = exports.Float64List = exports.Float32List = exports.DataList = exports.Data = exports.CompositeList = exports.BoolList = exports.AnyPointerList = exports.readRawPointer = exports.ObjectSize = exports.Message = exports.ListElementSize = void 0;
var serialization_1 = require("./serialization");
Object.defineProperty(exports, "ListElementSize", { enumerable: true, get: function () { return serialization_1.ListElementSize; } });
Object.defineProperty(exports, "Message", { enumerable: true, get: function () { return serialization_1.Message; } });
Object.defineProperty(exports, "ObjectSize", { enumerable: true, get: function () { return serialization_1.ObjectSize; } });
Object.defineProperty(exports, "readRawPointer", { enumerable: true, get: function () { return serialization_1.readRawPointer; } });
Object.defineProperty(exports, "AnyPointerList", { enumerable: true, get: function () { return serialization_1.AnyPointerList; } });
Object.defineProperty(exports, "BoolList", { enumerable: true, get: function () { return serialization_1.BoolList; } });
Object.defineProperty(exports, "CompositeList", { enumerable: true, get: function () { return serialization_1.CompositeList; } });
Object.defineProperty(exports, "Data", { enumerable: true, get: function () { return serialization_1.Data; } });
Object.defineProperty(exports, "DataList", { enumerable: true, get: function () { return serialization_1.DataList; } });
Object.defineProperty(exports, "Float32List", { enumerable: true, get: function () { return serialization_1.Float32List; } });
Object.defineProperty(exports, "Float64List", { enumerable: true, get: function () { return serialization_1.Float64List; } });
Object.defineProperty(exports, "Int16List", { enumerable: true, get: function () { return serialization_1.Int16List; } });
Object.defineProperty(exports, "Int32List", { enumerable: true, get: function () { return serialization_1.Int32List; } });
Object.defineProperty(exports, "Int64List", { enumerable: true, get: function () { return serialization_1.Int64List; } });
Object.defineProperty(exports, "Int8List", { enumerable: true, get: function () { return serialization_1.Int8List; } });
Object.defineProperty(exports, "Interface", { enumerable: true, get: function () { return serialization_1.Interface; } });
Object.defineProperty(exports, "InterfaceList", { enumerable: true, get: function () { return serialization_1.InterfaceList; } });
Object.defineProperty(exports, "List", { enumerable: true, get: function () { return serialization_1.List; } });
Object.defineProperty(exports, "Orphan", { enumerable: true, get: function () { return serialization_1.Orphan; } });
Object.defineProperty(exports, "PointerList", { enumerable: true, get: function () { return serialization_1.PointerList; } });
Object.defineProperty(exports, "PointerType", { enumerable: true, get: function () { return serialization_1.PointerType; } });
Object.defineProperty(exports, "Pointer", { enumerable: true, get: function () { return serialization_1.Pointer; } });
Object.defineProperty(exports, "Struct", { enumerable: true, get: function () { return serialization_1.Struct; } });
Object.defineProperty(exports, "Text", { enumerable: true, get: function () { return serialization_1.Text; } });
Object.defineProperty(exports, "TextList", { enumerable: true, get: function () { return serialization_1.TextList; } });
Object.defineProperty(exports, "Uint16List", { enumerable: true, get: function () { return serialization_1.Uint16List; } });
Object.defineProperty(exports, "Uint32List", { enumerable: true, get: function () { return serialization_1.Uint32List; } });
Object.defineProperty(exports, "Uint64List", { enumerable: true, get: function () { return serialization_1.Uint64List; } });
Object.defineProperty(exports, "Uint8List", { enumerable: true, get: function () { return serialization_1.Uint8List; } });
Object.defineProperty(exports, "VoidList", { enumerable: true, get: function () { return serialization_1.VoidList; } });
Object.defineProperty(exports, "Void", { enumerable: true, get: function () { return serialization_1.Void; } });
Object.defineProperty(exports, "getBitMask", { enumerable: true, get: function () { return serialization_1.getBitMask; } });
Object.defineProperty(exports, "getFloat32Mask", { enumerable: true, get: function () { return serialization_1.getFloat32Mask; } });
Object.defineProperty(exports, "getFloat64Mask", { enumerable: true, get: function () { return serialization_1.getFloat64Mask; } });
Object.defineProperty(exports, "getInt16Mask", { enumerable: true, get: function () { return serialization_1.getInt16Mask; } });
Object.defineProperty(exports, "getInt32Mask", { enumerable: true, get: function () { return serialization_1.getInt32Mask; } });
Object.defineProperty(exports, "getInt64Mask", { enumerable: true, get: function () { return serialization_1.getInt64Mask; } });
Object.defineProperty(exports, "getInt8Mask", { enumerable: true, get: function () { return serialization_1.getInt8Mask; } });
Object.defineProperty(exports, "getUint16Mask", { enumerable: true, get: function () { return serialization_1.getUint16Mask; } });
Object.defineProperty(exports, "getUint32Mask", { enumerable: true, get: function () { return serialization_1.getUint32Mask; } });
Object.defineProperty(exports, "getUint64Mask", { enumerable: true, get: function () { return serialization_1.getUint64Mask; } });
Object.defineProperty(exports, "getUint8Mask", { enumerable: true, get: function () { return serialization_1.getUint8Mask; } });
var types_1 = require("./types");
Object.defineProperty(exports, "Int64", { enumerable: true, get: function () { return types_1.Int64; } });
Object.defineProperty(exports, "Uint64", { enumerable: true, get: function () { return types_1.Uint64; } });

},{"./serialization":12,"./types":48}],6:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ArenaAllocationResult = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const trace = debug_1.default("capnp:serialization:arena:arena-allocation-result");
trace("load");
class ArenaAllocationResult {
    constructor(id, buffer) {
        this.id = id;
        this.buffer = buffer;
        trace("new", this);
    }
}
exports.ArenaAllocationResult = ArenaAllocationResult;

},{"debug":52,"tslib":56}],7:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ArenaKind = void 0;
var ArenaKind;
(function (ArenaKind) {
    ArenaKind[ArenaKind["SINGLE_SEGMENT"] = 0] = "SINGLE_SEGMENT";
    ArenaKind[ArenaKind["MULTI_SEGMENT"] = 1] = "MULTI_SEGMENT";
})(ArenaKind = exports.ArenaKind || (exports.ArenaKind = {}));

},{}],8:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.getNumSegments = exports.getBuffer = exports.allocate = exports.Arena = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const errors_1 = require("../../errors");
const arena_kind_1 = require("./arena-kind");
const multi_segment_arena_1 = require("./multi-segment-arena");
const single_segment_arena_1 = require("./single-segment-arena");
const trace = debug_1.default("capnp:arena");
trace("load");
class Arena {
}
exports.Arena = Arena;
Arena.allocate = allocate;
Arena.getBuffer = getBuffer;
Arena.getNumSegments = getNumSegments;
function allocate(minSize, segments, a) {
    switch (a.kind) {
        case arena_kind_1.ArenaKind.MULTI_SEGMENT:
            return multi_segment_arena_1.MultiSegmentArena.allocate(minSize, a);
        case arena_kind_1.ArenaKind.SINGLE_SEGMENT:
            return single_segment_arena_1.SingleSegmentArena.allocate(minSize, segments, a);
        default:
            return errors_1.assertNever(a);
    }
}
exports.allocate = allocate;
function getBuffer(id, a) {
    switch (a.kind) {
        case arena_kind_1.ArenaKind.MULTI_SEGMENT:
            return multi_segment_arena_1.MultiSegmentArena.getBuffer(id, a);
        case arena_kind_1.ArenaKind.SINGLE_SEGMENT:
            return single_segment_arena_1.SingleSegmentArena.getBuffer(id, a);
        default:
            return errors_1.assertNever(a);
    }
}
exports.getBuffer = getBuffer;
function getNumSegments(a) {
    switch (a.kind) {
        case arena_kind_1.ArenaKind.MULTI_SEGMENT:
            return multi_segment_arena_1.MultiSegmentArena.getNumSegments(a);
        case arena_kind_1.ArenaKind.SINGLE_SEGMENT:
            return single_segment_arena_1.SingleSegmentArena.getNumSegments();
        default:
            return errors_1.assertNever(a);
    }
}
exports.getNumSegments = getNumSegments;

},{"../../errors":4,"./arena-kind":7,"./multi-segment-arena":10,"./single-segment-arena":11,"debug":52,"tslib":56}],9:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.SingleSegmentArena = exports.MultiSegmentArena = exports.ArenaKind = exports.Arena = void 0;
var arena_1 = require("./arena");
Object.defineProperty(exports, "Arena", { enumerable: true, get: function () { return arena_1.Arena; } });
var arena_kind_1 = require("./arena-kind");
Object.defineProperty(exports, "ArenaKind", { enumerable: true, get: function () { return arena_kind_1.ArenaKind; } });
var multi_segment_arena_1 = require("./multi-segment-arena");
Object.defineProperty(exports, "MultiSegmentArena", { enumerable: true, get: function () { return multi_segment_arena_1.MultiSegmentArena; } });
var single_segment_arena_1 = require("./single-segment-arena");
Object.defineProperty(exports, "SingleSegmentArena", { enumerable: true, get: function () { return single_segment_arena_1.SingleSegmentArena; } });

},{"./arena":8,"./arena-kind":7,"./multi-segment-arena":10,"./single-segment-arena":11}],10:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.getNumSegments = exports.getBuffer = exports.allocate = exports.MultiSegmentArena = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("../../constants");
const errors_1 = require("../../errors");
const util_1 = require("../../util");
const arena_allocation_result_1 = require("./arena-allocation-result");
const arena_kind_1 = require("./arena-kind");
const trace = debug_1.default("capnp:arena:multi");
trace("load");
class MultiSegmentArena {
    constructor(buffers = []) {
        this.kind = arena_kind_1.ArenaKind.MULTI_SEGMENT;
        this.buffers = buffers;
        trace("new %s", this);
    }
    toString() {
        return util_1.format("MultiSegmentArena_segments:%d", getNumSegments(this));
    }
}
exports.MultiSegmentArena = MultiSegmentArena;
MultiSegmentArena.allocate = allocate;
MultiSegmentArena.getBuffer = getBuffer;
MultiSegmentArena.getNumSegments = getNumSegments;
function allocate(minSize, m) {
    const b = new ArrayBuffer(util_1.padToWord(Math.max(minSize, constants_1.DEFAULT_BUFFER_SIZE)));
    m.buffers.push(b);
    return new arena_allocation_result_1.ArenaAllocationResult(m.buffers.length - 1, b);
}
exports.allocate = allocate;
function getBuffer(id, m) {
    if (id < 0 || id >= m.buffers.length) {
        throw new Error(util_1.format(errors_1.SEG_ID_OUT_OF_BOUNDS, id));
    }
    return m.buffers[id];
}
exports.getBuffer = getBuffer;
function getNumSegments(m) {
    return m.buffers.length;
}
exports.getNumSegments = getNumSegments;

},{"../../constants":3,"../../errors":4,"../../util":51,"./arena-allocation-result":6,"./arena-kind":7,"debug":52,"tslib":56}],11:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.getNumSegments = exports.getBuffer = exports.allocate = exports.SingleSegmentArena = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("../../constants");
const errors_1 = require("../../errors");
const util_1 = require("../../util");
const arena_allocation_result_1 = require("./arena-allocation-result");
const arena_kind_1 = require("./arena-kind");
const trace = debug_1.default("capnp:arena:single");
trace("load");
class SingleSegmentArena {
    constructor(buffer = new ArrayBuffer(constants_1.DEFAULT_BUFFER_SIZE)) {
        this.kind = arena_kind_1.ArenaKind.SINGLE_SEGMENT;
        if ((buffer.byteLength & 7) !== 0) {
            throw new Error(util_1.format(errors_1.SEG_NOT_WORD_ALIGNED, buffer.byteLength));
        }
        this.buffer = buffer;
        trace("new %s", this);
    }
    toString() {
        return util_1.format("SingleSegmentArena_len:%x", this.buffer.byteLength);
    }
}
exports.SingleSegmentArena = SingleSegmentArena;
SingleSegmentArena.allocate = allocate;
SingleSegmentArena.getBuffer = getBuffer;
SingleSegmentArena.getNumSegments = getNumSegments;
function allocate(minSize, segments, s) {
    trace("Allocating %x bytes for segment 0 in %s.", minSize, s);
    const srcBuffer = segments.length > 0 ? segments[0].buffer : s.buffer;
    if (minSize < constants_1.MIN_SINGLE_SEGMENT_GROWTH) {
        minSize = constants_1.MIN_SINGLE_SEGMENT_GROWTH;
    }
    else {
        minSize = util_1.padToWord(minSize);
    }
    s.buffer = new ArrayBuffer(srcBuffer.byteLength + minSize);
    // PERF: Assume that the source and destination buffers are word-aligned and use Float64Array to copy them one word
    // at a time.
    new Float64Array(s.buffer).set(new Float64Array(srcBuffer));
    return new arena_allocation_result_1.ArenaAllocationResult(0, s.buffer);
}
exports.allocate = allocate;
function getBuffer(id, s) {
    if (id !== 0)
        throw new Error(util_1.format(errors_1.SEG_GET_NON_ZERO_SINGLE, id));
    return s.buffer;
}
exports.getBuffer = getBuffer;
function getNumSegments() {
    return 1;
}
exports.getNumSegments = getNumSegments;

},{"../../constants":3,"../../errors":4,"../../util":51,"./arena-allocation-result":6,"./arena-kind":7,"debug":52,"tslib":56}],12:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ObjectSize = exports.readRawPointer = exports.Message = exports.ListElementSize = void 0;
const tslib_1 = require("tslib");
tslib_1.__exportStar(require("./mask"), exports);
var list_element_size_1 = require("./list-element-size");
Object.defineProperty(exports, "ListElementSize", { enumerable: true, get: function () { return list_element_size_1.ListElementSize; } });
var message_1 = require("./message");
Object.defineProperty(exports, "Message", { enumerable: true, get: function () { return message_1.Message; } });
Object.defineProperty(exports, "readRawPointer", { enumerable: true, get: function () { return message_1.readRawPointer; } });
var object_size_1 = require("./object-size");
Object.defineProperty(exports, "ObjectSize", { enumerable: true, get: function () { return object_size_1.ObjectSize; } });
tslib_1.__exportStar(require("./pointers/index"), exports);

},{"./list-element-size":13,"./mask":14,"./message":15,"./object-size":16,"./pointers/index":25,"tslib":56}],13:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.ListElementOffset = exports.ListElementSize = void 0;
var ListElementSize;
(function (ListElementSize) {
    ListElementSize[ListElementSize["VOID"] = 0] = "VOID";
    ListElementSize[ListElementSize["BIT"] = 1] = "BIT";
    ListElementSize[ListElementSize["BYTE"] = 2] = "BYTE";
    ListElementSize[ListElementSize["BYTE_2"] = 3] = "BYTE_2";
    ListElementSize[ListElementSize["BYTE_4"] = 4] = "BYTE_4";
    ListElementSize[ListElementSize["BYTE_8"] = 5] = "BYTE_8";
    ListElementSize[ListElementSize["POINTER"] = 6] = "POINTER";
    ListElementSize[ListElementSize["COMPOSITE"] = 7] = "COMPOSITE";
})(ListElementSize = exports.ListElementSize || (exports.ListElementSize = {}));
exports.ListElementOffset = [
    0,
    0.125,
    1,
    2,
    4,
    8,
    8,
    NaN // composite
];

},{}],14:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.getVoidMask = exports.getUint64Mask = exports.getInt64Mask = exports.getBitMask = exports.getUint8Mask = exports.getUint32Mask = exports.getUint16Mask = exports.getInt8Mask = exports.getInt32Mask = exports.getInt16Mask = exports.getFloat64Mask = exports.getFloat32Mask = void 0;
const errors_1 = require("../errors");
function _makePrimitiveMaskFn(byteLength, setter) {
    return (x) => {
        const dv = new DataView(new ArrayBuffer(byteLength));
        setter.call(dv, 0, x, true);
        return dv;
    };
}
/* eslint-disable @typescript-eslint/unbound-method */
exports.getFloat32Mask = _makePrimitiveMaskFn(4, DataView.prototype.setFloat32);
exports.getFloat64Mask = _makePrimitiveMaskFn(8, DataView.prototype.setFloat64);
exports.getInt16Mask = _makePrimitiveMaskFn(2, DataView.prototype.setInt16);
exports.getInt32Mask = _makePrimitiveMaskFn(4, DataView.prototype.setInt32);
exports.getInt8Mask = _makePrimitiveMaskFn(1, DataView.prototype.setInt8);
exports.getUint16Mask = _makePrimitiveMaskFn(2, DataView.prototype.setUint16);
exports.getUint32Mask = _makePrimitiveMaskFn(4, DataView.prototype.setUint32);
exports.getUint8Mask = _makePrimitiveMaskFn(1, DataView.prototype.setUint8);
/* eslint-enable */
function getBitMask(value, bitOffset) {
    const dv = new DataView(new ArrayBuffer(1));
    if (!value)
        return dv;
    dv.setUint8(0, 1 << bitOffset % 8);
    return dv;
}
exports.getBitMask = getBitMask;
function getInt64Mask(x) {
    return x.toDataView();
}
exports.getInt64Mask = getInt64Mask;
function getUint64Mask(x) {
    return x.toDataView();
}
exports.getUint64Mask = getUint64Mask;
function getVoidMask() {
    throw new Error(errors_1.INVARIANT_UNREACHABLE_CODE);
}
exports.getVoidMask = getVoidMask;

},{"../errors":4}],15:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.getStreamFrame = exports.toPackedArrayBuffer = exports.toArrayBuffer = exports.setRoot = exports.readRawPointer = exports.initRoot = exports.getSegment = exports.getRoot = exports.dump = exports.allocateSegment = exports.preallocateSegments = exports.getFramedSegments = exports.initMessage = exports.Message = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("../constants");
const errors_1 = require("../errors");
const util_1 = require("../util");
const arena_1 = require("./arena");
const packing_1 = require("./packing");
const pointers_1 = require("./pointers");
const segment_1 = require("./segment");
const pointer_1 = require("./pointers/pointer");
const struct_1 = require("./pointers/struct");
const trace = debug_1.default("capnp:message");
trace("load");
class Message {
    /**
     * A Cap'n Proto message.
     *
     * SECURITY WARNING: In nodejs do not pass a Buffer's internal array buffer into this constructor. Pass the buffer
     * directly and everything will be fine. If not, your message will potentially be initialized with random memory
     * contents!
     *
     * The constructor method creates a new Message, optionally using a provided arena for segment allocation, or a buffer
     * to read from.
     *
     * @constructor {Message}
     *
     * @param {AnyArena|ArrayBufferView|ArrayBuffer} [src] The source for the message.
     * A value of `undefined` will cause the message to initialize with a single segment arena only big enough for the
     * root pointer; it will expand as you go. This is a reasonable choice for most messages.
     *
     * Passing an arena will cause the message to use that arena for its segment allocation. Contents will be accepted
     * as-is.
     *
     * Passing an array buffer view (like `DataView`, `Uint8Array` or `Buffer`) will create a **copy** of the source
     * buffer; beware of the potential performance cost!
     *
     * @param {boolean} [packed] Whether or not the message is packed. If `true` (the default), the message will be
     * unpacked.
     *
     * @param {boolean} [singleSegment] If true, `src` will be treated as a message consisting of a single segment without
     * a framing header.
     *
     */
    constructor(src, packed = true, singleSegment = false) {
        this._capnp = initMessage(src, packed, singleSegment);
        if (src && !isAnyArena(src))
            preallocateSegments(this);
        trace("new %s", this);
    }
    allocateSegment(byteLength) {
        return allocateSegment(byteLength, this);
    }
    /**
     * Create a pretty-printed string dump of this message; incredibly useful for debugging.
     *
     * WARNING: Do not call this method on large messages!
     *
     * @returns {string} A big steaming pile of pretty hex digits.
     */
    dump() {
        return dump(this);
    }
    /**
     * Get a struct pointer for the root of this message. This is primarily used when reading a message; it will not
     * overwrite existing data.
     *
     * @template T
     * @param {StructCtor<T>} RootStruct The struct type to use as the root.
     * @returns {T} A struct representing the root of the message.
     */
    getRoot(RootStruct) {
        return getRoot(RootStruct, this);
    }
    /**
     * Get a segment by its id.
     *
     * This will lazily allocate the first segment if it doesn't already exist.
     *
     * @param {number} id The segment id.
     * @returns {Segment} The requested segment.
     */
    getSegment(id) {
        return getSegment(id, this);
    }
    /**
     * Initialize a new message using the provided struct type as the root.
     *
     * @template T
     * @param {StructCtor<T>} RootStruct The struct type to use as the root.
     * @returns {T} An initialized struct pointing to the root of the message.
     */
    initRoot(RootStruct) {
        return initRoot(RootStruct, this);
    }
    /**
     * Set the root of the message to a copy of the given pointer. Used internally
     * to make copies of pointers for default values.
     *
     * @param {Pointer} src The source pointer to copy.
     * @returns {void}
     */
    setRoot(src) {
        setRoot(src, this);
    }
    /**
     * Combine the contents of this message's segments into a single array buffer and prepend a stream framing header
     * containing information about the following segment data.
     *
     * @returns {ArrayBuffer} An ArrayBuffer with the contents of this message.
     */
    toArrayBuffer() {
        return toArrayBuffer(this);
    }
    /**
     * Like `toArrayBuffer()`, but also applies the packing algorithm to the output. This is typically what you want to
     * use if you're sending the message over a network link or other slow I/O interface where size matters.
     *
     * @returns {ArrayBuffer} A packed message.
     */
    toPackedArrayBuffer() {
        return toPackedArrayBuffer(this);
    }
    toString() {
        // eslint-disable-next-line @typescript-eslint/restrict-template-expressions
        return `Message_arena:${this._capnp.arena}`;
    }
}
exports.Message = Message;
Message.allocateSegment = allocateSegment;
Message.dump = dump;
Message.getRoot = getRoot;
Message.getSegment = getSegment;
Message.initRoot = initRoot;
Message.readRawPointer = readRawPointer;
Message.toArrayBuffer = toArrayBuffer;
Message.toPackedArrayBuffer = toPackedArrayBuffer;
function initMessage(src, packed = true, singleSegment = false) {
    if (src === undefined) {
        return {
            arena: new arena_1.SingleSegmentArena(),
            segments: [],
            traversalLimit: constants_1.DEFAULT_TRAVERSE_LIMIT,
        };
    }
    if (isAnyArena(src)) {
        return { arena: src, segments: [], traversalLimit: constants_1.DEFAULT_TRAVERSE_LIMIT };
    }
    let buf = src;
    if (isArrayBufferView(buf)) {
        buf = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    if (packed)
        buf = packing_1.unpack(buf);
    if (singleSegment) {
        return {
            arena: new arena_1.SingleSegmentArena(buf),
            segments: [],
            traversalLimit: constants_1.DEFAULT_TRAVERSE_LIMIT,
        };
    }
    return {
        arena: new arena_1.MultiSegmentArena(getFramedSegments(buf)),
        segments: [],
        traversalLimit: constants_1.DEFAULT_TRAVERSE_LIMIT,
    };
}
exports.initMessage = initMessage;
/**
 * Given an _unpacked_ message with a segment framing header, this will generate an ArrayBuffer for each segment in
 * the message.
 *
 * This method is not typically called directly, but can be useful in certain cases.
 *
 * @static
 * @param {ArrayBuffer} message An unpacked message with a framing header.
 * @returns {ArrayBuffer[]} An array of buffers containing the segment data.
 */
function getFramedSegments(message) {
    const dv = new DataView(message);
    const segmentCount = dv.getUint32(0, true) + 1;
    const segments = new Array(segmentCount);
    trace("reading %d framed segments from stream", segmentCount);
    let byteOffset = 4 + segmentCount * 4;
    byteOffset += byteOffset % 8;
    if (byteOffset + segmentCount * 4 > message.byteLength) {
        throw new Error(errors_1.MSG_INVALID_FRAME_HEADER);
    }
    for (let i = 0; i < segmentCount; i++) {
        const byteLength = dv.getUint32(4 + i * 4, true) * 8;
        if (byteOffset + byteLength > message.byteLength) {
            throw new Error(errors_1.MSG_INVALID_FRAME_HEADER);
        }
        segments[i] = message.slice(byteOffset, byteOffset + byteLength);
        byteOffset += byteLength;
    }
    return segments;
}
exports.getFramedSegments = getFramedSegments;
/**
 * This method is called on messages that were constructed with existing data to prepopulate the segments array with
 * everything we can find in the arena. Each segment will have it's `byteLength` set to the size of its buffer.
 *
 * Technically speaking, the message's segments will be "full" after calling this function. Calling this on your own
 * may void your warranty.
 *
 * @param {Message} m The message to allocate.
 * @returns {void}
 */
function preallocateSegments(m) {
    const numSegments = arena_1.Arena.getNumSegments(m._capnp.arena);
    if (numSegments < 1)
        throw new Error(errors_1.MSG_NO_SEGMENTS_IN_ARENA);
    m._capnp.segments = new Array(numSegments);
    for (let i = 0; i < numSegments; i++) {
        // Set up each segment so that they're fully allocated to the extents of the existing buffers.
        const buffer = arena_1.Arena.getBuffer(i, m._capnp.arena);
        const segment = new segment_1.Segment(i, m, buffer, buffer.byteLength);
        m._capnp.segments[i] = segment;
    }
}
exports.preallocateSegments = preallocateSegments;
function isArrayBufferView(src) {
    return src.byteOffset !== undefined;
}
function isAnyArena(o) {
    return o.kind !== undefined;
}
function allocateSegment(byteLength, m) {
    trace("allocating %x bytes for %s", byteLength, m);
    const res = arena_1.Arena.allocate(byteLength, m._capnp.segments, m._capnp.arena);
    let s;
    if (res.id === m._capnp.segments.length) {
        // Note how we're only allowing new segments in if they're exactly the next one in the array. There is no logical
        // reason for segments to be created out of order.
        s = new segment_1.Segment(res.id, m, res.buffer);
        trace("adding new segment %s", s);
        m._capnp.segments.push(s);
    }
    else if (res.id < 0 || res.id > m._capnp.segments.length) {
        throw new Error(util_1.format(errors_1.MSG_SEGMENT_OUT_OF_BOUNDS, res.id, m));
    }
    else {
        s = m._capnp.segments[res.id];
        trace("replacing segment %s with buffer (len:%d)", s, res.buffer.byteLength);
        s.replaceBuffer(res.buffer);
    }
    return s;
}
exports.allocateSegment = allocateSegment;
function dump(m) {
    let r = "";
    if (m._capnp.segments.length === 0) {
        return "================\nNo Segments\n================\n";
    }
    for (let i = 0; i < m._capnp.segments.length; i++) {
        r += `================\nSegment #${i}\n================\n`;
        const { buffer, byteLength } = m._capnp.segments[i];
        const b = new Uint8Array(buffer, 0, byteLength);
        r += util_1.dumpBuffer(b);
    }
    return r;
}
exports.dump = dump;
function getRoot(RootStruct, m) {
    const root = new RootStruct(m.getSegment(0), 0);
    pointer_1.validate(pointers_1.PointerType.STRUCT, root);
    const ts = pointer_1.getTargetStructSize(root);
    // Make sure the underlying pointer is actually big enough to hold the data and pointers as specified in the schema.
    // If not a shallow copy of the struct contents needs to be made before returning.
    if (ts.dataByteLength < RootStruct._capnp.size.dataByteLength ||
        ts.pointerLength < RootStruct._capnp.size.pointerLength) {
        trace("need to resize root struct %s", root);
        struct_1.resize(RootStruct._capnp.size, root);
    }
    return root;
}
exports.getRoot = getRoot;
function getSegment(id, m) {
    const segmentLength = m._capnp.segments.length;
    if (id === 0 && segmentLength === 0) {
        // Segment zero is special. If we have no segments in the arena we'll want to allocate a new one and leave room
        // for the root pointer.
        const arenaSegments = arena_1.Arena.getNumSegments(m._capnp.arena);
        if (arenaSegments === 0) {
            allocateSegment(constants_1.DEFAULT_BUFFER_SIZE, m);
        }
        else {
            // Okay, the arena already has a buffer we can use. This is totally fine.
            m._capnp.segments[0] = new segment_1.Segment(0, m, arena_1.Arena.getBuffer(0, m._capnp.arena));
        }
        if (!m._capnp.segments[0].hasCapacity(8)) {
            throw new Error(errors_1.MSG_SEGMENT_TOO_SMALL);
        }
        // This will leave room for the root pointer.
        m._capnp.segments[0].allocate(8);
        return m._capnp.segments[0];
    }
    if (id < 0 || id >= segmentLength) {
        throw new Error(util_1.format(errors_1.MSG_SEGMENT_OUT_OF_BOUNDS, id, m));
    }
    return m._capnp.segments[id];
}
exports.getSegment = getSegment;
function initRoot(RootStruct, m) {
    const root = new RootStruct(m.getSegment(0), 0);
    struct_1.initStruct(RootStruct._capnp.size, root);
    trace("Initialized root pointer %s for %s.", root, m);
    return root;
}
exports.initRoot = initRoot;
/**
 * Read a pointer in raw form (a packed message with framing headers). Does not
 * care or attempt to validate the input beyond parsing the message
 * segments.
 *
 * This is typically used by the compiler to load default values, but can be
 * useful to work with messages with an unknown schema.
 *
 * @param {ArrayBuffer} data The raw data to read.
 * @returns {Pointer} A root pointer.
 */
function readRawPointer(data) {
    return new pointers_1.Pointer(new Message(data).getSegment(0), 0);
}
exports.readRawPointer = readRawPointer;
function setRoot(src, m) {
    pointers_1.Pointer.copyFrom(src, new pointers_1.Pointer(m.getSegment(0), 0));
}
exports.setRoot = setRoot;
function toArrayBuffer(m) {
    const streamFrame = getStreamFrame(m);
    // Make sure the first segment is allocated.
    if (m._capnp.segments.length === 0)
        getSegment(0, m);
    const segments = m._capnp.segments;
    // Add space for the stream framing.
    const totalLength = streamFrame.byteLength + segments.reduce((l, s) => l + util_1.padToWord(s.byteLength), 0);
    const out = new Uint8Array(new ArrayBuffer(totalLength));
    let o = streamFrame.byteLength;
    out.set(new Uint8Array(streamFrame));
    segments.forEach((s) => {
        const segmentLength = util_1.padToWord(s.byteLength);
        out.set(new Uint8Array(s.buffer, 0, segmentLength), o);
        o += segmentLength;
    });
    return out.buffer;
}
exports.toArrayBuffer = toArrayBuffer;
function toPackedArrayBuffer(m) {
    const streamFrame = packing_1.pack(getStreamFrame(m));
    // Make sure the first segment is allocated.
    if (m._capnp.segments.length === 0)
        m.getSegment(0);
    // NOTE: A copy operation can be avoided here if we capture the intermediate array and use that directly in the copy
    // loop below, rather than have `pack()` copy it to an ArrayBuffer just to have to copy it again later. If the
    // intermediate array can be avoided altogether that's even better!
    const segments = m._capnp.segments.map((s) => packing_1.pack(s.buffer, 0, util_1.padToWord(s.byteLength)));
    const totalLength = streamFrame.byteLength + segments.reduce((l, s) => l + s.byteLength, 0);
    const out = new Uint8Array(new ArrayBuffer(totalLength));
    let o = streamFrame.byteLength;
    out.set(new Uint8Array(streamFrame));
    segments.forEach((s) => {
        out.set(new Uint8Array(s), o);
        o += s.byteLength;
    });
    return out.buffer;
}
exports.toPackedArrayBuffer = toPackedArrayBuffer;
function getStreamFrame(m) {
    const length = m._capnp.segments.length;
    if (length === 0) {
        // Don't bother allocating the first segment, just return a single zero word for the frame header.
        return new Float64Array(1).buffer;
    }
    const frameLength = 4 + length * 4 + (1 - (length % 2)) * 4;
    const out = new DataView(new ArrayBuffer(frameLength));
    trace("Writing message stream frame with segment count: %d.", length);
    out.setUint32(0, length - 1, true);
    m._capnp.segments.forEach((s, i) => {
        trace("Message segment %d word count: %d.", s.id, s.byteLength / 8);
        out.setUint32(i * 4 + 4, s.byteLength / 8, true);
    });
    return out.buffer;
}
exports.getStreamFrame = getStreamFrame;

},{"../constants":3,"../errors":4,"../util":51,"./arena":9,"./packing":17,"./pointers":25,"./pointers/pointer":37,"./pointers/struct":38,"./segment":47,"debug":52,"tslib":56}],16:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.padToWord = exports.getWordLength = exports.getDataWordLength = exports.getByteLength = exports.ObjectSize = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const _ = tslib_1.__importStar(require("../util"));
const trace = debug_1.default("capnp:object-size");
trace("load");
/**
 * A simple object that describes the size of a struct.
 *
 * @export
 * @class ObjectSize
 */
class ObjectSize {
    constructor(dataByteLength, pointerCount) {
        this.dataByteLength = dataByteLength;
        this.pointerLength = pointerCount;
    }
    toString() {
        return _.format("ObjectSize_dw:%d,pc:%d", getDataWordLength(this), this.pointerLength);
    }
}
exports.ObjectSize = ObjectSize;
function getByteLength(o) {
    return o.dataByteLength + o.pointerLength * 8;
}
exports.getByteLength = getByteLength;
function getDataWordLength(o) {
    return o.dataByteLength / 8;
}
exports.getDataWordLength = getDataWordLength;
function getWordLength(o) {
    return o.dataByteLength / 8 + o.pointerLength;
}
exports.getWordLength = getWordLength;
function padToWord(o) {
    return new ObjectSize(_.padToWord(o.dataByteLength), o.pointerLength);
}
exports.padToWord = padToWord;

},{"../util":51,"debug":52,"tslib":56}],17:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.unpack = exports.pack = exports.getZeroByteCount = exports.getUnpackedByteLength = exports.getTagByte = exports.getHammingWeight = void 0;
const constants_1 = require("../constants");
const errors_1 = require("../errors");
/**
 * Compute the Hamming weight (number of bits set to 1) of a number. Used to figure out how many bytes follow a tag byte
 * while computing the size of a packed message.
 *
 * WARNING: Using this with floating point numbers will void your warranty.
 *
 * @param {number} x A real integer.
 * @returns {number} The hamming weight (integer).
 */
function getHammingWeight(x) {
    // Thanks, HACKMEM!
    let w = x - ((x >> 1) & 0x55555555);
    w = (w & 0x33333333) + ((w >> 2) & 0x33333333);
    return (((w + (w >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}
exports.getHammingWeight = getHammingWeight;
/**
 * Compute the tag byte from the 8 bytes of a 64-bit word.
 *
 * @param {byte} a The first byte.
 * @param {byte} b The second byte.
 * @param {byte} c The third byte.
 * @param {byte} d The fourth byte.
 * @param {byte} e The fifth byte.
 * @param {byte} f The sixth byte.
 * @param {byte} g The seventh byte.
 * @param {byte} h The eighth byte (phew!).
 * @returns {number} The tag byte.
 */
function getTagByte(a, b, c, d, e, f, g, h) {
    // Yes, it's pretty. Don't touch it.
    return ((a === 0 ? 0 : 0b00000001) |
        (b === 0 ? 0 : 0b00000010) |
        (c === 0 ? 0 : 0b00000100) |
        (d === 0 ? 0 : 0b00001000) |
        (e === 0 ? 0 : 0b00010000) |
        (f === 0 ? 0 : 0b00100000) |
        (g === 0 ? 0 : 0b01000000) |
        (h === 0 ? 0 : 0b10000000));
}
exports.getTagByte = getTagByte;
/**
 * Efficiently calculate the length of a packed Cap'n Proto message.
 *
 * @export
 * @param {ArrayBuffer} packed The packed message.
 * @returns {number} The length of the unpacked message in bytes.
 */
function getUnpackedByteLength(packed) {
    const p = new Uint8Array(packed);
    let wordLength = 0;
    let lastTag = 0x77;
    for (let i = 0; i < p.byteLength;) {
        const tag = p[i];
        if (lastTag === 0 /* ZERO */) {
            wordLength += tag;
            i++;
            lastTag = 0x77;
        }
        else if (lastTag === 255 /* SPAN */) {
            wordLength += tag;
            i += tag * 8 + 1;
            lastTag = 0x77;
        }
        else {
            wordLength++;
            i += getHammingWeight(tag) + 1;
            lastTag = tag;
        }
    }
    return wordLength * 8;
}
exports.getUnpackedByteLength = getUnpackedByteLength;
/**
 * Compute the number of zero bytes that occur in a given 64-bit word, provided as eight separate bytes.
 *
 * @param {byte} a The first byte.
 * @param {byte} b The second byte.
 * @param {byte} c The third byte.
 * @param {byte} d The fourth byte.
 * @param {byte} e The fifth byte.
 * @param {byte} f The sixth byte.
 * @param {byte} g The seventh byte.
 * @param {byte} h The eighth byte (phew!).
 * @returns {number} The number of these bytes that are zero.
 */
function getZeroByteCount(a, b, c, d, e, f, g, h) {
    return ((a === 0 ? 1 : 0) +
        (b === 0 ? 1 : 0) +
        (c === 0 ? 1 : 0) +
        (d === 0 ? 1 : 0) +
        (e === 0 ? 1 : 0) +
        (f === 0 ? 1 : 0) +
        (g === 0 ? 1 : 0) +
        (h === 0 ? 1 : 0));
}
exports.getZeroByteCount = getZeroByteCount;
/**
 * Pack a section of a Cap'n Proto message into a compressed format. This will efficiently compress zero bytes (which
 * are common in idiomatic Cap'n Proto messages) into a compact form.
 *
 * For stream-framed messages this is called once for the frame header and once again for each segment in the message.
 *
 * The returned array buffer is trimmed to the exact size of the packed message with a single copy operation at the end.
 * This should be decent on CPU time but does require quite a lot of memory (a normal array is filled up with each
 * packed byte until the packing is complete).
 *
 * @export
 * @param {ArrayBuffer} unpacked The message to pack.
 * @param {number} [byteOffset] Starting byte offset to read bytes from, defaults to 0.
 * @param {number} [byteLength] Total number of bytes to read, defaults to the remainder of the buffer contents.
 * @returns {ArrayBuffer} A packed version of the message.
 */
function pack(unpacked, byteOffset = 0, byteLength) {
    if (unpacked.byteLength % 8 !== 0)
        throw new Error(errors_1.MSG_PACK_NOT_WORD_ALIGNED);
    const src = new Uint8Array(unpacked, byteOffset, byteLength);
    // TODO: Maybe we should do this with buffers? This costs more than 8x the final compressed size in temporary RAM.
    const dst = [];
    /* Just have to be sure it's neither ZERO nor SPAN. */
    let lastTag = 0x77;
    /** This is where we need to remember to write the SPAN tag (0xff). */
    let spanTagOffset = NaN;
    /** How many words have been copied during the current span. */
    let spanWordLength = 0;
    /**
     * When this hits zero, we've had PACK_SPAN_THRESHOLD zero bytes pass by and it's time to bail from the span.
     */
    let spanThreshold = constants_1.PACK_SPAN_THRESHOLD;
    for (let srcByteOffset = 0; srcByteOffset < src.byteLength; srcByteOffset += 8) {
        /** Read in the entire word. Yes, this feels silly but it's fast! */
        const a = src[srcByteOffset];
        const b = src[srcByteOffset + 1];
        const c = src[srcByteOffset + 2];
        const d = src[srcByteOffset + 3];
        const e = src[srcByteOffset + 4];
        const f = src[srcByteOffset + 5];
        const g = src[srcByteOffset + 6];
        const h = src[srcByteOffset + 7];
        const tag = getTagByte(a, b, c, d, e, f, g, h);
        /** If this is true we'll skip the normal word write logic after the switch statement. */
        let skipWriteWord = true;
        switch (lastTag) {
            case 0 /* ZERO */:
                // We're writing a span of words with all zeroes in them. See if we need to bail out of the fast path.
                if (tag !== 0 /* ZERO */ || spanWordLength >= 0xff) {
                    // There's a bit in there or we got too many zeroes. Damn, we need to bail.
                    dst.push(spanWordLength);
                    spanWordLength = 0;
                    skipWriteWord = false;
                }
                else {
                    // Kay, let's quickly inc this and go.
                    spanWordLength++;
                }
                break;
            case 255 /* SPAN */: {
                // We're writing a span of nonzero words.
                const zeroCount = getZeroByteCount(a, b, c, d, e, f, g, h);
                // See if we need to bail now.
                spanThreshold -= zeroCount;
                if (spanThreshold <= 0 || spanWordLength >= 0xff) {
                    // Alright, time to get packing again. Write the number of words we skipped to the beginning of the span.
                    dst[spanTagOffset] = spanWordLength;
                    spanWordLength = 0;
                    spanThreshold = constants_1.PACK_SPAN_THRESHOLD;
                    // We have to write this word normally.
                    skipWriteWord = false;
                }
                else {
                    // Just write this word verbatim.
                    dst.push(a, b, c, d, e, f, g, h);
                    spanWordLength++;
                }
                break;
            }
            default:
                // Didn't get a special tag last time, let's write this as normal.
                skipWriteWord = false;
                break;
        }
        // A goto is fast, idk why people keep hatin'.
        if (skipWriteWord)
            continue;
        dst.push(tag);
        lastTag = tag;
        if (a !== 0)
            dst.push(a);
        if (b !== 0)
            dst.push(b);
        if (c !== 0)
            dst.push(c);
        if (d !== 0)
            dst.push(d);
        if (e !== 0)
            dst.push(e);
        if (f !== 0)
            dst.push(f);
        if (g !== 0)
            dst.push(g);
        if (h !== 0)
            dst.push(h);
        // Record the span tag offset if needed, making sure to actually leave room for it.
        if (tag === 255 /* SPAN */) {
            spanTagOffset = dst.length;
            dst.push(0);
        }
    }
    // We're done. If we were writing a span let's finish it.
    if (lastTag === 0 /* ZERO */) {
        dst.push(spanWordLength);
    }
    else if (lastTag === 255 /* SPAN */) {
        dst[spanTagOffset] = spanWordLength;
    }
    return new Uint8Array(dst).buffer;
}
exports.pack = pack;
/**
 * Unpack a compressed Cap'n Proto message into a new ArrayBuffer.
 *
 * Unlike the `pack` function, this is able to efficiently determine the exact size needed for the output buffer and
 * runs considerably more efficiently.
 *
 * @export
 * @param {ArrayBuffer} packed An array buffer containing the packed message.
 * @returns {ArrayBuffer} The unpacked message.
 */
function unpack(packed) {
    // We have no choice but to read the packed buffer one byte at a time.
    const src = new Uint8Array(packed);
    const dst = new Uint8Array(new ArrayBuffer(getUnpackedByteLength(packed)));
    /** The last tag byte that we've seen - it starts at a "neutral" value. */
    let lastTag = 0x77;
    for (let srcByteOffset = 0, dstByteOffset = 0; srcByteOffset < src.byteLength;) {
        const tag = src[srcByteOffset];
        if (lastTag === 0 /* ZERO */) {
            // We have a span of zeroes. New array buffers are guaranteed to be initialized to zero so we just seek ahead.
            dstByteOffset += tag * 8;
            srcByteOffset++;
            lastTag = 0x77;
        }
        else if (lastTag === 255 /* SPAN */) {
            // We have a span of unpacked bytes. Copy them verbatim from the source buffer.
            const spanByteLength = tag * 8;
            dst.set(src.subarray(srcByteOffset + 1, srcByteOffset + 1 + spanByteLength), dstByteOffset);
            dstByteOffset += spanByteLength;
            srcByteOffset += 1 + spanByteLength;
            lastTag = 0x77;
        }
        else {
            // Okay, a normal tag. Let's read past the tag and copy bytes that have a bit set in the tag.
            srcByteOffset++;
            for (let i = 1; i <= 0b10000000; i <<= 1) {
                // We only need to actually touch `dst` if there's a nonzero byte (it's already initialized to zeroes).
                if ((tag & i) !== 0)
                    dst[dstByteOffset] = src[srcByteOffset++];
                dstByteOffset++;
            }
            lastTag = tag;
        }
    }
    return dst.buffer;
}
exports.unpack = unpack;

},{"../constants":3,"../errors":4}],18:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.AnyPointerList = void 0;
const pointer_1 = require("./pointer");
const pointer_list_1 = require("./pointer-list");
exports.AnyPointerList = pointer_list_1.PointerList(pointer_1.Pointer);

},{"./pointer":37,"./pointer-list":35}],19:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.BoolList = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class BoolList extends list_1.List {
    get(index) {
        const bitMask = 1 << index % 8;
        const byteOffset = index >>> 3;
        const c = pointer_1.getContent(this);
        const v = c.segment.getUint8(c.byteOffset + byteOffset);
        return (v & bitMask) !== 0;
    }
    set(index, value) {
        const bitMask = 1 << index % 8;
        const c = pointer_1.getContent(this);
        const byteOffset = c.byteOffset + (index >>> 3);
        const v = c.segment.getUint8(byteOffset);
        c.segment.setUint8(byteOffset, value ? v | bitMask : v & ~bitMask);
    }
    toString() {
        return `Bool_${super.toString()}`;
    }
}
exports.BoolList = BoolList;
BoolList._capnp = {
    displayName: "List<boolean>",
    size: list_element_size_1.ListElementSize.BIT
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],20:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.CompositeList = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
function CompositeList(CompositeClass) {
    var _a;
    return _a = class extends list_1.List {
            get(index) {
                return new CompositeClass(this.segment, this.byteOffset, this._capnp.depthLimit - 1, index);
            }
            set(index, value) {
                pointer_1.copyFrom(value, this.get(index));
            }
            toString() {
                return `Composite_${super.toString()},cls:${CompositeClass.toString()}`;
            }
        },
        _a._capnp = {
            compositeSize: CompositeClass._capnp.size,
            displayName: `List<${CompositeClass._capnp.displayName}>`,
            size: list_element_size_1.ListElementSize.COMPOSITE,
        },
        _a;
}
exports.CompositeList = CompositeList;

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],21:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.DataList = void 0;
const data_1 = require("./data");
const pointer_list_1 = require("./pointer-list");
exports.DataList = pointer_list_1.PointerList(data_1.Data);

},{"./data":22,"./pointer-list":35}],22:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Data = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const pointer_type_1 = require("./pointer-type");
const trace = debug_1.default("capnp:data");
trace("load");
/**
 * A generic blob of bytes. Can be converted to a DataView or Uint8Array to access its contents using `toDataView()` and
 * `toUint8Array()`. Use `copyBuffer()` to copy an entire buffer at once.
 *
 * @export
 * @class Data
 * @extends {List<number>}
 */
class Data extends list_1.List {
    static fromPointer(pointer) {
        pointer_1.validate(pointer_type_1.PointerType.LIST, pointer, list_element_size_1.ListElementSize.BYTE);
        return this._fromPointerUnchecked(pointer);
    }
    static _fromPointerUnchecked(pointer) {
        return new this(pointer.segment, pointer.byteOffset, pointer._capnp.depthLimit);
    }
    /**
     * Copy the contents of `src` into this Data pointer. If `src` is smaller than the length of this pointer then the
     * remaining bytes will be zeroed out. Extra bytes in `src` are ignored.
     *
     * @param {(ArrayBuffer | ArrayBufferView)} src The source buffer.
     * @returns {void}
     */
    // TODO: Would be nice to have a way to zero-copy a buffer by allocating a new segment into the message with that
    // buffer data.
    copyBuffer(src) {
        const c = pointer_1.getContent(this);
        const dstLength = this.getLength();
        const srcLength = src.byteLength;
        const i = src instanceof ArrayBuffer
            ? new Uint8Array(src)
            : new Uint8Array(src.buffer, src.byteOffset, Math.min(dstLength, srcLength));
        const o = new Uint8Array(c.segment.buffer, c.byteOffset, this.getLength());
        o.set(i);
        if (dstLength > srcLength) {
            trace("Zeroing out remaining %d bytes after copy into %s.", dstLength - srcLength, this);
            o.fill(0, srcLength, dstLength);
        }
        else if (dstLength < srcLength) {
            trace("Truncated %d bytes from source buffer while copying to %s.", srcLength - dstLength, this);
        }
    }
    /**
     * Read a byte from the specified offset.
     *
     * @param {number} byteOffset The byte offset to read.
     * @returns {number} The byte value.
     */
    get(byteOffset) {
        const c = pointer_1.getContent(this);
        return c.segment.getUint8(c.byteOffset + byteOffset);
    }
    /**
     * Write a byte at the specified offset.
     *
     * @param {number} byteOffset The byte offset to set.
     * @param {number} value The byte value to set.
     * @returns {void}
     */
    set(byteOffset, value) {
        const c = pointer_1.getContent(this);
        c.segment.setUint8(c.byteOffset + byteOffset, value);
    }
    /**
     * Creates a **copy** of the underlying buffer data and returns it as an ArrayBuffer.
     *
     * To obtain a reference to the underlying buffer instead, use `toUint8Array()` or `toDataView()`.
     *
     * @returns {ArrayBuffer} A copy of this data buffer.
     */
    toArrayBuffer() {
        const c = pointer_1.getContent(this);
        return c.segment.buffer.slice(c.byteOffset, c.byteOffset + this.getLength());
    }
    /**
     * Convert this Data pointer to a DataView representing the pointer's contents.
     *
     * WARNING: The DataView references memory from a message segment, so do not venture outside the bounds of the
     * DataView or else BAD THINGS.
     *
     * @returns {DataView} A live reference to the underlying buffer.
     */
    toDataView() {
        const c = pointer_1.getContent(this);
        return new DataView(c.segment.buffer, c.byteOffset, this.getLength());
    }
    toString() {
        return `Data_${super.toString()}`;
    }
    /**
     * Convert this Data pointer to a Uint8Array representing the pointer's contents.
     *
     * WARNING: The Uint8Array references memory from a message segment, so do not venture outside the bounds of the
     * Uint8Array or else BAD THINGS.
     *
     * @returns {DataView} A live reference to the underlying buffer.
     */
    toUint8Array() {
        const c = pointer_1.getContent(this);
        return new Uint8Array(c.segment.buffer, c.byteOffset, this.getLength());
    }
}
exports.Data = Data;

},{"../list-element-size":13,"./list":32,"./pointer":37,"./pointer-type":36,"debug":52,"tslib":56}],23:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Float32List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Float32List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getFloat32(c.byteOffset + index * 4);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setFloat32(c.byteOffset + index * 4, value);
    }
    toString() {
        return `Float32_${super.toString()}`;
    }
}
exports.Float32List = Float32List;
Float32List._capnp = {
    displayName: "List<Float32>",
    size: list_element_size_1.ListElementSize.BYTE_4
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],24:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Float64List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Float64List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getFloat64(c.byteOffset + index * 8);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setFloat64(c.byteOffset + index * 8, value);
    }
    toString() {
        return `Float64_${super.toString()}`;
    }
}
exports.Float64List = Float64List;
Float64List._capnp = {
    displayName: "List<Float64>",
    size: list_element_size_1.ListElementSize.BYTE_8
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],25:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.VoidList = exports.VOID = exports.Void = exports.Uint64List = exports.Uint32List = exports.Uint16List = exports.Uint8List = exports.TextList = exports.Text = exports.Struct = exports.Pointer = exports.PointerType = exports.PointerList = exports.Orphan = exports.List = exports.InterfaceList = exports.Interface = exports.Int64List = exports.Int32List = exports.Int16List = exports.Int8List = exports.Float64List = exports.Float32List = exports.DataList = exports.Data = exports.CompositeList = exports.BoolList = exports.AnyPointerList = void 0;
var any_pointer_list_1 = require("./any-pointer-list");
Object.defineProperty(exports, "AnyPointerList", { enumerable: true, get: function () { return any_pointer_list_1.AnyPointerList; } });
var bool_list_1 = require("./bool-list");
Object.defineProperty(exports, "BoolList", { enumerable: true, get: function () { return bool_list_1.BoolList; } });
var composite_list_1 = require("./composite-list");
Object.defineProperty(exports, "CompositeList", { enumerable: true, get: function () { return composite_list_1.CompositeList; } });
var data_1 = require("./data");
Object.defineProperty(exports, "Data", { enumerable: true, get: function () { return data_1.Data; } });
var data_list_1 = require("./data-list");
Object.defineProperty(exports, "DataList", { enumerable: true, get: function () { return data_list_1.DataList; } });
var float32_list_1 = require("./float32-list");
Object.defineProperty(exports, "Float32List", { enumerable: true, get: function () { return float32_list_1.Float32List; } });
var float64_list_1 = require("./float64-list");
Object.defineProperty(exports, "Float64List", { enumerable: true, get: function () { return float64_list_1.Float64List; } });
var int8_list_1 = require("./int8-list");
Object.defineProperty(exports, "Int8List", { enumerable: true, get: function () { return int8_list_1.Int8List; } });
var int16_list_1 = require("./int16-list");
Object.defineProperty(exports, "Int16List", { enumerable: true, get: function () { return int16_list_1.Int16List; } });
var int32_list_1 = require("./int32-list");
Object.defineProperty(exports, "Int32List", { enumerable: true, get: function () { return int32_list_1.Int32List; } });
var int64_list_1 = require("./int64-list");
Object.defineProperty(exports, "Int64List", { enumerable: true, get: function () { return int64_list_1.Int64List; } });
var interface_1 = require("./interface");
Object.defineProperty(exports, "Interface", { enumerable: true, get: function () { return interface_1.Interface; } });
var interface_list_1 = require("./interface-list");
Object.defineProperty(exports, "InterfaceList", { enumerable: true, get: function () { return interface_list_1.InterfaceList; } });
var list_1 = require("./list");
Object.defineProperty(exports, "List", { enumerable: true, get: function () { return list_1.List; } });
var orphan_1 = require("./orphan");
Object.defineProperty(exports, "Orphan", { enumerable: true, get: function () { return orphan_1.Orphan; } });
var pointer_list_1 = require("./pointer-list");
Object.defineProperty(exports, "PointerList", { enumerable: true, get: function () { return pointer_list_1.PointerList; } });
var pointer_type_1 = require("./pointer-type");
Object.defineProperty(exports, "PointerType", { enumerable: true, get: function () { return pointer_type_1.PointerType; } });
var pointer_1 = require("./pointer");
Object.defineProperty(exports, "Pointer", { enumerable: true, get: function () { return pointer_1.Pointer; } });
var struct_1 = require("./struct");
Object.defineProperty(exports, "Struct", { enumerable: true, get: function () { return struct_1.Struct; } });
var text_1 = require("./text");
Object.defineProperty(exports, "Text", { enumerable: true, get: function () { return text_1.Text; } });
var text_list_1 = require("./text-list");
Object.defineProperty(exports, "TextList", { enumerable: true, get: function () { return text_list_1.TextList; } });
var uint8_list_1 = require("./uint8-list");
Object.defineProperty(exports, "Uint8List", { enumerable: true, get: function () { return uint8_list_1.Uint8List; } });
var uint16_list_1 = require("./uint16-list");
Object.defineProperty(exports, "Uint16List", { enumerable: true, get: function () { return uint16_list_1.Uint16List; } });
var uint32_list_1 = require("./uint32-list");
Object.defineProperty(exports, "Uint32List", { enumerable: true, get: function () { return uint32_list_1.Uint32List; } });
var uint64_list_1 = require("./uint64-list");
Object.defineProperty(exports, "Uint64List", { enumerable: true, get: function () { return uint64_list_1.Uint64List; } });
var void_1 = require("./void");
Object.defineProperty(exports, "Void", { enumerable: true, get: function () { return void_1.Void; } });
Object.defineProperty(exports, "VOID", { enumerable: true, get: function () { return void_1.VOID; } });
var void_list_1 = require("./void-list");
Object.defineProperty(exports, "VoidList", { enumerable: true, get: function () { return void_list_1.VoidList; } });

},{"./any-pointer-list":18,"./bool-list":19,"./composite-list":20,"./data":22,"./data-list":21,"./float32-list":23,"./float64-list":24,"./int16-list":26,"./int32-list":27,"./int64-list":28,"./int8-list":29,"./interface":31,"./interface-list":30,"./list":32,"./orphan":33,"./pointer":37,"./pointer-list":35,"./pointer-type":36,"./struct":38,"./text":40,"./text-list":39,"./uint16-list":41,"./uint32-list":42,"./uint64-list":43,"./uint8-list":44,"./void":46,"./void-list":45}],26:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Int16List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Int16List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getInt16(c.byteOffset + index * 2);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setInt16(c.byteOffset + index * 2, value);
    }
    toString() {
        return `Int16_${super.toString()}`;
    }
}
exports.Int16List = Int16List;
Int16List._capnp = {
    displayName: "List<Int16>",
    size: list_element_size_1.ListElementSize.BYTE_2
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],27:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Int32List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Int32List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getInt32(c.byteOffset + index * 4);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setInt32(c.byteOffset + index * 4, value);
    }
    toString() {
        return `Int32_${super.toString()}`;
    }
}
exports.Int32List = Int32List;
Int32List._capnp = {
    displayName: "List<Int32>",
    size: list_element_size_1.ListElementSize.BYTE_4
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],28:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Int64List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Int64List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getInt64(c.byteOffset + index * 8);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setInt64(c.byteOffset + index * 8, value);
    }
    toString() {
        return `Int64_${super.toString()}`;
    }
}
exports.Int64List = Int64List;
Int64List._capnp = {
    displayName: "List<Int64>",
    size: list_element_size_1.ListElementSize.BYTE_8,
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],29:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Int8List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Int8List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getInt8(c.byteOffset + index);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setInt8(c.byteOffset + index, value);
    }
    toString() {
        return `Int8_${super.toString()}`;
    }
}
exports.Int8List = Int8List;
Int8List._capnp = {
    displayName: "List<Int8>",
    size: list_element_size_1.ListElementSize.BYTE
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],30:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.InterfaceList = void 0;
const interface_1 = require("./interface");
const pointer_list_1 = require("./pointer-list");
exports.InterfaceList = pointer_list_1.PointerList(interface_1.Interface);

},{"./interface":31,"./pointer-list":35}],31:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Interface = void 0;
const constants_1 = require("../../constants");
const errors_1 = require("../../errors");
const util_1 = require("../../util");
const pointer_1 = require("./pointer");
class Interface extends pointer_1.Pointer {
    constructor(segment, byteOffset, depthLimit = constants_1.MAX_DEPTH) {
        super(segment, byteOffset, depthLimit);
        throw new Error(util_1.format(errors_1.NOT_IMPLEMENTED, "new Interface"));
    }
}
exports.Interface = Interface;

},{"../../constants":3,"../../errors":4,"../../util":51,"./pointer":37}],32:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.set = exports.get = exports.initList = exports.List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const errors_1 = require("../../errors");
const util_1 = require("../../util");
const list_element_size_1 = require("../list-element-size");
const object_size_1 = require("../object-size");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list");
trace("load");
/**
 * A generic list class. Implements Filterable,
 */
class List extends pointer_1.Pointer {
    static toString() {
        return this._capnp.displayName;
    }
    all(callbackfn) {
        const length = this.getLength();
        for (let i = 0; i < length; i++) {
            if (!callbackfn(this.get(i), i))
                return false;
        }
        return true;
    }
    any(callbackfn) {
        const length = this.getLength();
        for (let i = 0; i < length; i++) {
            if (callbackfn(this.get(i), i))
                return true;
        }
        return false;
    }
    ap(callbackfns) {
        const length = this.getLength();
        const res = [];
        for (let i = 0; i < length; i++) {
            res.push(...callbackfns.map((f) => f(this.get(i), i)));
        }
        return res;
    }
    concat(other) {
        const length = this.getLength();
        const otherLength = other.getLength();
        const res = new Array(length + otherLength);
        for (let i = 0; i < length; i++)
            res[i] = this.get(i);
        for (let i = 0; i < otherLength; i++)
            res[i + length] = other.get(i);
        return res;
    }
    drop(n) {
        const length = this.getLength();
        const res = new Array(length);
        for (let i = n; i < length; i++)
            res[i] = this.get(i);
        return res;
    }
    dropWhile(callbackfn) {
        const length = this.getLength();
        const res = [];
        let drop = true;
        for (let i = 0; i < length; i++) {
            const v = this.get(i);
            if (drop)
                drop = callbackfn(v, i);
            if (!drop)
                res.push(v);
        }
        return res;
    }
    empty() {
        return [];
    }
    every(callbackfn) {
        return this.all(callbackfn);
    }
    filter(callbackfn) {
        const length = this.getLength();
        const res = [];
        for (let i = 0; i < length; i++) {
            const value = this.get(i);
            if (callbackfn(value, i))
                res.push(value);
        }
        return res;
    }
    find(callbackfn) {
        const length = this.getLength();
        for (let i = 0; i < length; i++) {
            const value = this.get(i);
            if (callbackfn(value, i))
                return value;
        }
        return undefined;
    }
    findIndex(callbackfn) {
        const length = this.getLength();
        for (let i = 0; i < length; i++) {
            const value = this.get(i);
            if (callbackfn(value, i))
                return i;
        }
        return -1;
    }
    forEach(callbackfn) {
        const length = this.getLength();
        for (let i = 0; i < length; i++)
            callbackfn(this.get(i), i);
    }
    get(_index) {
        return get(_index, this);
    }
    /**
     * Get the length of this list.
     *
     * @returns {number} The number of elements in this list.
     */
    getLength() {
        return pointer_1.getTargetListLength(this);
    }
    groupBy(callbackfn) {
        const length = this.getLength();
        const res = {};
        for (let i = 0; i < length; i++) {
            const v = this.get(i);
            res[callbackfn(v, i)] = v;
        }
        return res;
    }
    intersperse(sep) {
        const length = this.getLength();
        const res = new Array(length);
        for (let i = 0; i < length; i++) {
            if (i > 0)
                res.push(sep);
            res.push(this.get(i));
        }
        return res;
    }
    map(callbackfn) {
        const length = this.getLength();
        const res = new Array(length);
        for (let i = 0; i < length; i++)
            res[i] = callbackfn(this.get(i), i);
        return res;
    }
    reduce(callbackfn, initialValue) {
        let i = 0;
        let res;
        if (initialValue === undefined) {
            res = this.get(0);
            i++;
        }
        else {
            res = initialValue;
        }
        for (; i < this.getLength(); i++)
            res = callbackfn(res, this.get(i), i);
        return res;
    }
    set(_index, _value) {
        set(_index, _value, this);
    }
    slice(start = 0, end) {
        const length = end ? Math.min(this.getLength(), end) : this.getLength();
        const res = new Array(length - start);
        for (let i = start; i < length; i++)
            res[i] = this.get(i);
        return res;
    }
    some(callbackfn) {
        return this.any(callbackfn);
    }
    take(n) {
        const length = Math.min(this.getLength(), n);
        const res = new Array(length);
        for (let i = 0; i < length; i++)
            res[i] = this.get(i);
        return res;
    }
    takeWhile(callbackfn) {
        const length = this.getLength();
        const res = [];
        let take;
        for (let i = 0; i < length; i++) {
            const v = this.get(i);
            take = callbackfn(v, i);
            if (!take)
                return res;
            res.push(v);
        }
        return res;
    }
    toArray() {
        return this.map(util_1.identity);
    }
    toString() {
        return `List_${super.toString()}`;
    }
}
exports.List = List;
List._capnp = {
    displayName: "List<Generic>",
    size: list_element_size_1.ListElementSize.VOID,
};
List.get = get;
List.initList = initList;
List.set = set;
/**
 * Initialize the list with the given element size and length. This will allocate new space for the list, ideally in
 * the same segment as this pointer.
 *
 * @param {ListElementSize} elementSize The size of each element in the list.
 * @param {number} length The number of elements in the list.
 * @param {List<T>} l The list to initialize.
 * @param {ObjectSize} [compositeSize] The size of each element in a composite list. This value is required for
 * composite lists.
 * @returns {void}
 */
function initList(elementSize, length, l, compositeSize) {
    let c;
    switch (elementSize) {
        case list_element_size_1.ListElementSize.BIT:
            c = l.segment.allocate(Math.ceil(length / 8));
            break;
        case list_element_size_1.ListElementSize.BYTE:
        case list_element_size_1.ListElementSize.BYTE_2:
        case list_element_size_1.ListElementSize.BYTE_4:
        case list_element_size_1.ListElementSize.BYTE_8:
        case list_element_size_1.ListElementSize.POINTER:
            c = l.segment.allocate(length * pointer_1.getListElementByteLength(elementSize));
            break;
        case list_element_size_1.ListElementSize.COMPOSITE: {
            if (compositeSize === undefined) {
                throw new Error(util_1.format(errors_1.PTR_COMPOSITE_SIZE_UNDEFINED));
            }
            compositeSize = object_size_1.padToWord(compositeSize);
            const byteLength = object_size_1.getByteLength(compositeSize) * length;
            // We need to allocate an extra 8 bytes for the tag word, then make sure we write the length to it. We advance
            // the content pointer by 8 bytes so that it then points to the first list element as intended. Everything
            // starts off zeroed out so these nested structs don't need to be initialized in any way.
            c = l.segment.allocate(byteLength + 8);
            pointer_1.setStructPointer(length, compositeSize, c);
            trace("Wrote composite tag word %s for %s.", c, l);
            break;
        }
        case list_element_size_1.ListElementSize.VOID:
            // No need to allocate anything, we can write the list pointer right here.
            pointer_1.setListPointer(0, elementSize, length, l);
            return;
        default:
            throw new Error(util_1.format(errors_1.PTR_INVALID_LIST_SIZE, elementSize));
    }
    const res = pointer_1.initPointer(c.segment, c.byteOffset, l);
    pointer_1.setListPointer(res.offsetWords, elementSize, length, res.pointer, compositeSize);
}
exports.initList = initList;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function get(_index, _l) {
    throw new TypeError();
}
exports.get = get;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function set(_index, _value, _l) {
    throw new TypeError();
}
exports.set = set;

},{"../../errors":4,"../../util":51,"../list-element-size":13,"../object-size":16,"./pointer":37,"debug":52,"tslib":56}],33:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Orphan = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const errors_1 = require("../../errors");
const util_1 = require("../../util");
const list_element_size_1 = require("../list-element-size");
const object_size_1 = require("../object-size");
const pointer_1 = require("./pointer");
const pointer_type_1 = require("./pointer-type");
const trace = debug_1.default("capnp:orphan");
trace("load");
// Technically speaking this class doesn't need to be generic, but the extra type checking enforced by this helps to
// make sure you don't accidentally adopt a pointer of the wrong type.
/**
 * An orphaned pointer. This object itself is technically a pointer to the original pointer's content, which was left
 * untouched in its original message. The original pointer data is encoded as attributes on the Orphan object, ready to
 * be reconstructed once another pointer is ready to adopt it.
 *
 * @export
 * @class Orphan
 * @extends {Pointer}
 * @template T
 */
class Orphan {
    constructor(src) {
        const c = pointer_1.getContent(src);
        this.segment = c.segment;
        this.byteOffset = c.byteOffset;
        this._capnp = {};
        // Read vital info from the src pointer so we can reconstruct it during adoption.
        this._capnp.type = pointer_1.getTargetPointerType(src);
        switch (this._capnp.type) {
            case pointer_type_1.PointerType.STRUCT:
                this._capnp.size = pointer_1.getTargetStructSize(src);
                break;
            case pointer_type_1.PointerType.LIST:
                this._capnp.length = pointer_1.getTargetListLength(src);
                this._capnp.elementSize = pointer_1.getTargetListElementSize(src);
                if (this._capnp.elementSize === list_element_size_1.ListElementSize.COMPOSITE) {
                    this._capnp.size = pointer_1.getTargetCompositeListSize(src);
                }
                break;
            case pointer_type_1.PointerType.OTHER:
                this._capnp.capId = pointer_1.getCapabilityId(src);
                break;
            default:
                // COVERAGE: Unreachable code.
                /* istanbul ignore next */
                throw new Error(errors_1.PTR_INVALID_POINTER_TYPE);
        }
        // Zero out the source pointer (but not the contents!).
        pointer_1.erasePointer(src);
    }
    /**
     * Adopt (move) this orphan into the target pointer location. This will allocate far pointers in `dst` as needed.
     *
     * @param {T} dst The destination pointer.
     * @returns {void}
     */
    _moveTo(dst) {
        if (this._capnp === undefined) {
            throw new Error(util_1.format(errors_1.PTR_ALREADY_ADOPTED, this));
        }
        // TODO: Implement copy semantics when this happens.
        if (this.segment.message !== dst.segment.message) {
            throw new Error(util_1.format(errors_1.PTR_ADOPT_WRONG_MESSAGE, this, dst));
        }
        // Recursively wipe out the destination pointer first.
        pointer_1.erase(dst);
        const res = pointer_1.initPointer(this.segment, this.byteOffset, dst);
        switch (this._capnp.type) {
            case pointer_type_1.PointerType.STRUCT:
                pointer_1.setStructPointer(res.offsetWords, this._capnp.size, res.pointer);
                break;
            case pointer_type_1.PointerType.LIST: {
                let offsetWords = res.offsetWords;
                if (this._capnp.elementSize === list_element_size_1.ListElementSize.COMPOSITE) {
                    offsetWords--; // The tag word gets skipped.
                }
                pointer_1.setListPointer(offsetWords, this._capnp.elementSize, this._capnp.length, res.pointer, this._capnp.size);
                break;
            }
            case pointer_type_1.PointerType.OTHER:
                pointer_1.setInterfacePointer(this._capnp.capId, res.pointer);
                break;
            /* istanbul ignore next */
            default:
                throw new Error(errors_1.PTR_INVALID_POINTER_TYPE);
        }
        this._capnp = undefined;
    }
    dispose() {
        // FIXME: Should this throw?
        if (this._capnp === undefined) {
            trace("not disposing an already disposed orphan", this);
            return;
        }
        switch (this._capnp.type) {
            case pointer_type_1.PointerType.STRUCT:
                this.segment.fillZeroWords(this.byteOffset, object_size_1.getWordLength(this._capnp.size));
                break;
            case pointer_type_1.PointerType.LIST: {
                const byteLength = pointer_1.getListByteLength(this._capnp.elementSize, this._capnp.length, this._capnp.size);
                this.segment.fillZeroWords(this.byteOffset, byteLength);
                break;
            }
            default:
                // Other pointer types don't actually have any content.
                break;
        }
        this._capnp = undefined;
    }
    toString() {
        return util_1.format("Orphan_%d@%a,type:%s", this.segment.id, this.byteOffset, this._capnp && this._capnp.type);
    }
}
exports.Orphan = Orphan;

},{"../../errors":4,"../../util":51,"../list-element-size":13,"../object-size":16,"./pointer":37,"./pointer-type":36,"debug":52,"tslib":56}],34:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PointerAllocationResult = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const trace = debug_1.default("capnp:pointer-allocation-result");
trace("load");
/**
 * This is used as the return value for `Pointer.prototype.initPointer`. Turns out using a class in V8 for multiple
 * return values is faster than using an array or anonymous object.
 *
 * http://jsben.ch/#/zTdbD
 *
 * @export
 * @class PointerAllocationResult
 */
class PointerAllocationResult {
    constructor(pointer, offsetWords) {
        this.pointer = pointer;
        this.offsetWords = offsetWords;
    }
}
exports.PointerAllocationResult = PointerAllocationResult;

},{"debug":52,"tslib":56}],35:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PointerList = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
function PointerList(PointerClass) {
    var _a;
    return _a = class extends list_1.List {
            get(index) {
                const c = pointer_1.getContent(this);
                return new PointerClass(c.segment, c.byteOffset + index * 8, this._capnp.depthLimit - 1);
            }
            set(index, value) {
                pointer_1.copyFrom(value, this.get(index));
            }
            toString() {
                return `Pointer_${super.toString()},cls:${PointerClass.toString()}`;
            }
        },
        _a._capnp = {
            displayName: `List<${PointerClass._capnp.displayName}>`,
            size: list_element_size_1.ListElementSize.POINTER,
        },
        _a;
}
exports.PointerList = PointerList;

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],36:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.PointerType = void 0;
var PointerType;
(function (PointerType) {
    PointerType[PointerType["STRUCT"] = 0] = "STRUCT";
    PointerType[PointerType["LIST"] = 1] = "LIST";
    PointerType[PointerType["FAR"] = 2] = "FAR";
    PointerType[PointerType["OTHER"] = 3] = "OTHER";
})(PointerType = exports.PointerType || (exports.PointerType = {}));

},{}],37:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.trackPointerAllocation = exports.copyFromStruct = exports.copyFromList = exports.validate = exports.setStructPointer = exports.setListPointer = exports.setInterfacePointer = exports.setFarPointer = exports.relocateTo = exports.isNull = exports.isDoubleFar = exports.initPointer = exports.getTargetStructSize = exports.getTargetPointerType = exports.getTargetListLength = exports.getTargetListElementSize = exports.getTargetCompositeListSize = exports.getTargetCompositeListTag = exports.getStructSize = exports.getStructPointerLength = exports.getStructDataWords = exports.getPointerType = exports.getOffsetWords = exports.getListLength = exports.getListElementSize = exports.getFarSegmentId = exports.getContent = exports.getCapabilityId = exports.followFars = exports.followFar = exports.erasePointer = exports.erase = exports.copyFrom = exports.add = exports.getListElementByteLength = exports.getListByteLength = exports.dump = exports.disown = exports.adopt = exports.Pointer = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("../../constants");
const util_1 = require("../../util");
const list_element_size_1 = require("../list-element-size");
const object_size_1 = require("../object-size");
const orphan_1 = require("./orphan");
const pointer_allocation_result_1 = require("./pointer-allocation-result");
const pointer_type_1 = require("./pointer-type");
const errors_1 = require("../../errors");
const trace = debug_1.default("capnp:pointer");
trace("load");
/**
 * A pointer referencing a single byte location in a segment. This is typically used for Cap'n Proto pointers, but is
 * also sometimes used to reference an offset to a pointer's content or tag words.
 *
 * @export
 * @class Pointer
 */
class Pointer {
    constructor(segment, byteOffset, depthLimit = constants_1.MAX_DEPTH) {
        this._capnp = { compositeList: false, depthLimit };
        this.segment = segment;
        this.byteOffset = byteOffset;
        if (depthLimit === 0) {
            throw new Error(util_1.format(errors_1.PTR_DEPTH_LIMIT_EXCEEDED, this));
        }
        // Make sure we keep track of all pointer allocations; there's a limit per message (prevent DoS).
        trackPointerAllocation(segment.message, this);
        // NOTE: It's okay to have a pointer to the end of the segment; you'll see this when creating pointers to the
        // beginning of the content of a newly-allocated composite list with zero elements. Unlike other language
        // implementations buffer over/underflows are not a big issue since all buffer access is bounds checked in native
        // code anyway.
        if (byteOffset < 0 || byteOffset > segment.byteLength) {
            throw new Error(util_1.format(errors_1.PTR_OFFSET_OUT_OF_BOUNDS, byteOffset));
        }
        trace("new %s", this);
    }
    toString() {
        return util_1.format("Pointer_%d@%a,%s,limit:%x", this.segment.id, this.byteOffset, dump(this), this._capnp.depthLimit);
    }
}
exports.Pointer = Pointer;
Pointer.adopt = adopt;
Pointer.copyFrom = copyFrom;
Pointer.disown = disown;
Pointer.dump = dump;
Pointer.isNull = isNull;
Pointer._capnp = {
    displayName: "Pointer",
};
/**
 * Adopt an orphaned pointer, making the pointer point to the orphaned content without copying it.
 *
 * @param {Orphan<Pointer>} src The orphan to adopt.
 * @param {Pointer} p The the pointer to adopt into.
 * @returns {void}
 */
function adopt(src, p) {
    src._moveTo(p);
}
exports.adopt = adopt;
/**
 * Convert a pointer to an Orphan, zeroing out the pointer and leaving its content untouched. If the content is no
 * longer needed, call `disown()` on the orphaned pointer to erase the contents as well.
 *
 * Call `adopt()` on the orphan with the new target pointer location to move it back into the message; the orphan
 * object is then invalidated after adoption (can only adopt once!).
 *
 * @param {T} p The pointer to turn into an Orphan.
 * @returns {Orphan<T>} An orphaned pointer.
 */
function disown(p) {
    return new orphan_1.Orphan(p);
}
exports.disown = disown;
function dump(p) {
    return util_1.bufferToHex(p.segment.buffer.slice(p.byteOffset, p.byteOffset + 8));
}
exports.dump = dump;
/**
 * Get the total number of bytes required to hold a list of the provided size with the given length, rounded up to the
 * nearest word.
 *
 * @param {ListElementSize} elementSize A number describing the size of the list elements.
 * @param {number} length The length of the list.
 * @param {ObjectSize} [compositeSize] The size of each element in a composite list; required if
 * `elementSize === ListElementSize.COMPOSITE`.
 * @returns {number} The number of bytes required to hold an element of that size, or `NaN` if that is undefined.
 */
function getListByteLength(elementSize, length, compositeSize) {
    switch (elementSize) {
        case list_element_size_1.ListElementSize.BIT:
            return util_1.padToWord((length + 7) >>> 3);
        case list_element_size_1.ListElementSize.BYTE:
        case list_element_size_1.ListElementSize.BYTE_2:
        case list_element_size_1.ListElementSize.BYTE_4:
        case list_element_size_1.ListElementSize.BYTE_8:
        case list_element_size_1.ListElementSize.POINTER:
        case list_element_size_1.ListElementSize.VOID:
            return util_1.padToWord(getListElementByteLength(elementSize) * length);
        /* istanbul ignore next */
        case list_element_size_1.ListElementSize.COMPOSITE:
            if (compositeSize === undefined) {
                throw new Error(util_1.format(errors_1.PTR_INVALID_LIST_SIZE, NaN));
            }
            return length * util_1.padToWord(object_size_1.getByteLength(compositeSize));
        /* istanbul ignore next */
        default:
            throw new Error(errors_1.PTR_INVALID_LIST_SIZE);
    }
}
exports.getListByteLength = getListByteLength;
/**
 * Get the number of bytes required to hold a list element of the provided size. `COMPOSITE` elements do not have a
 * fixed size, and `BIT` elements are packed into exactly a single bit, so these both return `NaN`.
 *
 * @param {ListElementSize} elementSize A number describing the size of the list elements.
 * @returns {number} The number of bytes required to hold an element of that size, or `NaN` if that is undefined.
 */
function getListElementByteLength(elementSize) {
    switch (elementSize) {
        /* istanbul ignore next */
        case list_element_size_1.ListElementSize.BIT:
            return NaN;
        case list_element_size_1.ListElementSize.BYTE:
            return 1;
        case list_element_size_1.ListElementSize.BYTE_2:
            return 2;
        case list_element_size_1.ListElementSize.BYTE_4:
            return 4;
        case list_element_size_1.ListElementSize.BYTE_8:
        case list_element_size_1.ListElementSize.POINTER:
            return 8;
        /* istanbul ignore next */
        case list_element_size_1.ListElementSize.COMPOSITE:
            // Caller has to figure it out based on the tag word.
            return NaN;
        /* istanbul ignore next */
        case list_element_size_1.ListElementSize.VOID:
            return 0;
        /* istanbul ignore next */
        default:
            throw new Error(util_1.format(errors_1.PTR_INVALID_LIST_SIZE, elementSize));
    }
}
exports.getListElementByteLength = getListElementByteLength;
/**
 * Add an offset to the pointer's offset and return a new Pointer for that address.
 *
 * @param {number} offset The number of bytes to add to the offset.
 * @param {Pointer} p The pointer to add from.
 * @returns {Pointer} A new pointer to the address.
 */
function add(offset, p) {
    return new Pointer(p.segment, p.byteOffset + offset, p._capnp.depthLimit);
}
exports.add = add;
/**
 * Replace a pointer with a deep copy of the pointer at `src` and all of its contents.
 *
 * @param {Pointer} src The pointer to copy.
 * @param {Pointer} p The pointer to copy into.
 * @returns {void}
 */
function copyFrom(src, p) {
    // If the pointer is the same then this is a noop.
    if (p.segment === src.segment && p.byteOffset === src.byteOffset) {
        trace("ignoring copy operation from identical pointer %s", src);
        return;
    }
    // Make sure we erase this pointer's contents before moving on. If src is null, that's all we do.
    erase(p); // noop if null
    if (isNull(src))
        return;
    switch (getTargetPointerType(src)) {
        case pointer_type_1.PointerType.STRUCT:
            copyFromStruct(src, p);
            break;
        case pointer_type_1.PointerType.LIST:
            copyFromList(src, p);
            break;
        /* istanbul ignore next */
        default:
            throw new Error(util_1.format(errors_1.PTR_INVALID_POINTER_TYPE, getTargetPointerType(p)));
    }
}
exports.copyFrom = copyFrom;
/**
 * Recursively erase a pointer, any far pointers/landing pads/tag words, and the content it points to.
 *
 * Note that this will leave "holes" of zeroes in the message, since the space cannot be reclaimed. With packing this
 * will have a negligible effect on the final message size.
 *
 * FIXME: This may need protection against infinite recursion...
 *
 * @param {Pointer} p The pointer to erase.
 * @returns {void}
 */
function erase(p) {
    if (isNull(p))
        return;
    // First deal with the contents.
    let c;
    switch (getTargetPointerType(p)) {
        case pointer_type_1.PointerType.STRUCT: {
            const size = getTargetStructSize(p);
            c = getContent(p);
            // Wipe the data section.
            c.segment.fillZeroWords(c.byteOffset, size.dataByteLength / 8);
            // Iterate over all the pointers and nuke them.
            for (let i = 0; i < size.pointerLength; i++) {
                erase(add(i * 8, c));
            }
            break;
        }
        case pointer_type_1.PointerType.LIST: {
            const elementSize = getTargetListElementSize(p);
            const length = getTargetListLength(p);
            let contentWords = util_1.padToWord(length * getListElementByteLength(elementSize));
            c = getContent(p);
            if (elementSize === list_element_size_1.ListElementSize.POINTER) {
                for (let i = 0; i < length; i++) {
                    erase(new Pointer(c.segment, c.byteOffset + i * 8, p._capnp.depthLimit - 1));
                }
                // Calling erase on each pointer takes care of the content, nothing left to do here.
                break;
            }
            else if (elementSize === list_element_size_1.ListElementSize.COMPOSITE) {
                // Read some stuff from the tag word.
                const tag = add(-8, c);
                const compositeSize = getStructSize(tag);
                const compositeByteLength = object_size_1.getByteLength(compositeSize);
                contentWords = getOffsetWords(tag);
                // Kill the tag word.
                c.segment.setWordZero(c.byteOffset - 8);
                // Recursively erase each pointer.
                for (let i = 0; i < length; i++) {
                    for (let j = 0; j < compositeSize.pointerLength; j++) {
                        erase(new Pointer(c.segment, c.byteOffset + i * compositeByteLength + j * 8, p._capnp.depthLimit - 1));
                    }
                }
            }
            c.segment.fillZeroWords(c.byteOffset, contentWords);
            break;
        }
        case pointer_type_1.PointerType.OTHER:
            // No content.
            break;
        default:
            throw new Error(util_1.format(errors_1.PTR_INVALID_POINTER_TYPE, getTargetPointerType(p)));
    }
    erasePointer(p);
}
exports.erase = erase;
/**
 * Set the pointer (and far pointer landing pads, if applicable) to zero. Does not touch the pointer's content.
 *
 * @param {Pointer} p The pointer to erase.
 * @returns {void}
 */
function erasePointer(p) {
    if (getPointerType(p) === pointer_type_1.PointerType.FAR) {
        const landingPad = followFar(p);
        if (isDoubleFar(p)) {
            // Kill the double-far tag word.
            landingPad.segment.setWordZero(landingPad.byteOffset + 8);
        }
        // Kill the landing pad.
        landingPad.segment.setWordZero(landingPad.byteOffset);
    }
    // Finally! Kill the pointer itself...
    p.segment.setWordZero(p.byteOffset);
}
exports.erasePointer = erasePointer;
/**
 * Interpret the pointer as a far pointer, returning its target segment and offset.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {Pointer} A pointer to the far target.
 */
function followFar(p) {
    const targetSegment = p.segment.message.getSegment(p.segment.getUint32(p.byteOffset + 4));
    const targetWordOffset = p.segment.getUint32(p.byteOffset) >>> 3;
    return new Pointer(targetSegment, targetWordOffset * 8, p._capnp.depthLimit - 1);
}
exports.followFar = followFar;
/**
 * If the pointer address references a far pointer, follow it to the location where the actual pointer data is written.
 * Otherwise, returns the pointer unmodified.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {Pointer} A new pointer representing the target location, or `p` if it is not a far pointer.
 */
function followFars(p) {
    if (getPointerType(p) === pointer_type_1.PointerType.FAR) {
        const landingPad = followFar(p);
        if (isDoubleFar(p))
            landingPad.byteOffset += 8;
        return landingPad;
    }
    return p;
}
exports.followFars = followFars;
function getCapabilityId(p) {
    return p.segment.getUint32(p.byteOffset + 4);
}
exports.getCapabilityId = getCapabilityId;
function isCompositeList(p) {
    return getTargetPointerType(p) === pointer_type_1.PointerType.LIST && getTargetListElementSize(p) === list_element_size_1.ListElementSize.COMPOSITE;
}
/**
 * Obtain the location of the pointer's content, following far pointers as needed.
 * If the pointer is a struct pointer and `compositeIndex` is set, it will be offset by a multiple of the struct's size.
 *
 * @param {Pointer} p The pointer to read from.
 * @param {boolean} [ignoreCompositeIndex] If true, will not follow the composite struct pointer's composite index and
 * instead return a pointer to the parent list's contents (also the beginning of the first struct).
 * @returns {Pointer} A pointer to the beginning of the pointer's content.
 */
function getContent(p, ignoreCompositeIndex) {
    let c;
    if (isDoubleFar(p)) {
        const landingPad = followFar(p);
        c = new Pointer(p.segment.message.getSegment(getFarSegmentId(landingPad)), getOffsetWords(landingPad) * 8);
    }
    else {
        const target = followFars(p);
        c = new Pointer(target.segment, target.byteOffset + 8 + getOffsetWords(target) * 8);
    }
    if (isCompositeList(p))
        c.byteOffset += 8;
    if (!ignoreCompositeIndex && p._capnp.compositeIndex !== undefined) {
        // Seek backwards by one word so we can read the struct size off the tag word.
        c.byteOffset -= 8;
        // Seek ahead by `compositeIndex` multiples of the struct's total size.
        c.byteOffset += 8 + p._capnp.compositeIndex * object_size_1.getByteLength(object_size_1.padToWord(getStructSize(c)));
    }
    return c;
}
exports.getContent = getContent;
/**
 * Read the target segment ID from a far pointer.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {number} The target segment ID.
 */
function getFarSegmentId(p) {
    return p.segment.getUint32(p.byteOffset + 4);
}
exports.getFarSegmentId = getFarSegmentId;
/**
 * Get a number indicating the size of the list's elements.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {ListElementSize} The size of the list's elements.
 */
function getListElementSize(p) {
    return p.segment.getUint32(p.byteOffset + 4) & constants_1.LIST_SIZE_MASK;
}
exports.getListElementSize = getListElementSize;
/**
 * Get the number of elements in a list pointer. For composite lists, it instead represents the total number of words in
 * the list (not counting the tag word).
 *
 * This method does **not** attempt to distinguish between composite and non-composite lists. To get the correct
 * length for composite lists use `getTargetListLength()` instead.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {number} The length of the list, or total number of words for composite lists.
 */
function getListLength(p) {
    return p.segment.getUint32(p.byteOffset + 4) >>> 3;
}
exports.getListLength = getListLength;
/**
 * Get the offset (in words) from the end of a pointer to the start of its content. For struct pointers, this is the
 * beginning of the data section, and for list pointers it is the location of the first element. The value should
 * always be zero for interface pointers.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {number} The offset, in words, from the end of the pointer to the start of the data section.
 */
function getOffsetWords(p) {
    const o = p.segment.getInt32(p.byteOffset);
    // Far pointers only have 29 offset bits.
    return o & 2 ? o >> 3 : o >> 2;
}
exports.getOffsetWords = getOffsetWords;
/**
 * Look up the pointer's type.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {PointerType} The type of pointer.
 */
function getPointerType(p) {
    return p.segment.getUint32(p.byteOffset) & constants_1.POINTER_TYPE_MASK;
}
exports.getPointerType = getPointerType;
/**
 * Read the number of data words from this struct pointer.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {number} The number of data words in the struct.
 */
function getStructDataWords(p) {
    return p.segment.getUint16(p.byteOffset + 4);
}
exports.getStructDataWords = getStructDataWords;
/**
 * Read the number of pointers contained in this struct pointer.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {number} The number of pointers in this struct.
 */
function getStructPointerLength(p) {
    return p.segment.getUint16(p.byteOffset + 6);
}
exports.getStructPointerLength = getStructPointerLength;
/**
 * Get an object describing this struct pointer's size.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {ObjectSize} The size of the struct.
 */
function getStructSize(p) {
    return new object_size_1.ObjectSize(getStructDataWords(p) * 8, getStructPointerLength(p));
}
exports.getStructSize = getStructSize;
/**
 * Get a pointer to this pointer's composite list tag word, following far pointers as needed.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {Pointer} A pointer to the list's composite tag word.
 */
function getTargetCompositeListTag(p) {
    const c = getContent(p);
    // The composite list tag is always one word before the content.
    c.byteOffset -= 8;
    return c;
}
exports.getTargetCompositeListTag = getTargetCompositeListTag;
/**
 * Get the object size for the target composite list, following far pointers as needed.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {ObjectSize} An object describing the size of each struct in the list.
 */
function getTargetCompositeListSize(p) {
    return getStructSize(getTargetCompositeListTag(p));
}
exports.getTargetCompositeListSize = getTargetCompositeListSize;
/**
 * Get the size of the list elements referenced by this pointer, following far pointers if necessary.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {ListElementSize} The size of the elements in the list.
 */
function getTargetListElementSize(p) {
    return getListElementSize(followFars(p));
}
exports.getTargetListElementSize = getTargetListElementSize;
/**
 * Get the length of the list referenced by this pointer, following far pointers if necessary. If the list is a
 * composite list, it will look up the tag word and read the length from there.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {number} The number of elements in the list.
 */
function getTargetListLength(p) {
    const t = followFars(p);
    if (getListElementSize(t) === list_element_size_1.ListElementSize.COMPOSITE) {
        // The content is prefixed by a tag word; it's a struct pointer whose offset contains the list's length.
        return getOffsetWords(getTargetCompositeListTag(p));
    }
    return getListLength(t);
}
exports.getTargetListLength = getTargetListLength;
/**
 * Get the type of a pointer, following far pointers if necessary. For non-far pointers this is equivalent to calling
 * `getPointerType()`.
 *
 * The target of a far pointer can never be another far pointer, and this method will throw if such a situation is
 * encountered.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {PointerType} The type of pointer referenced by this pointer.
 */
function getTargetPointerType(p) {
    const t = getPointerType(followFars(p));
    if (t === pointer_type_1.PointerType.FAR)
        throw new Error(util_1.format(errors_1.PTR_INVALID_FAR_TARGET, p));
    return t;
}
exports.getTargetPointerType = getTargetPointerType;
/**
 * Get the size of the struct referenced by a pointer, following far pointers if necessary.
 *
 * @param {Pointer} p The poiner to read from.
 * @returns {ObjectSize} The size of the struct referenced by this pointer.
 */
function getTargetStructSize(p) {
    return getStructSize(followFars(p));
}
exports.getTargetStructSize = getTargetStructSize;
/**
 * Initialize a pointer to point at the data in the content segment. If the content segment is not the same as the
 * pointer's segment, this will allocate and write far pointers as needed. Nothing is written otherwise.
 *
 * The return value includes a pointer to write the pointer's actual data to (the eventual far target), and the offset
 * value (in words) to use for that pointer. In the case of double-far pointers this offset will always be zero.
 *
 * @param {Segment} contentSegment The segment containing this pointer's content.
 * @param {number} contentOffset The offset within the content segment for the beginning of this pointer's content.
 * @param {Pointer} p The pointer to initialize.
 * @returns {PointerAllocationResult} An object containing a pointer (where the pointer data should be written), and
 * the value to use as the offset for that pointer.
 */
function initPointer(contentSegment, contentOffset, p) {
    if (p.segment !== contentSegment) {
        // Need a far pointer.
        trace("Initializing far pointer %s -> %s.", p, contentSegment);
        if (!contentSegment.hasCapacity(8)) {
            // GAH! Not enough space in the content segment for a landing pad so we need a double far pointer.
            const landingPad = p.segment.allocate(16);
            trace("GAH! Initializing double-far pointer in %s from %s -> %s.", p, contentSegment, landingPad);
            setFarPointer(true, landingPad.byteOffset / 8, landingPad.segment.id, p);
            setFarPointer(false, contentOffset / 8, contentSegment.id, landingPad);
            landingPad.byteOffset += 8;
            return new pointer_allocation_result_1.PointerAllocationResult(landingPad, 0);
        }
        // Allocate a far pointer landing pad in the target segment.
        const landingPad = contentSegment.allocate(8);
        if (landingPad.segment.id !== contentSegment.id) {
            throw new Error(errors_1.INVARIANT_UNREACHABLE_CODE);
        }
        setFarPointer(false, landingPad.byteOffset / 8, landingPad.segment.id, p);
        return new pointer_allocation_result_1.PointerAllocationResult(landingPad, (contentOffset - landingPad.byteOffset - 8) / 8);
    }
    trace("Initializing intra-segment pointer %s -> %a.", p, contentOffset);
    return new pointer_allocation_result_1.PointerAllocationResult(p, (contentOffset - p.byteOffset - 8) / 8);
}
exports.initPointer = initPointer;
/**
 * Check if the pointer is a double-far pointer.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {boolean} `true` if it is a double-far pointer, `false` otherwise.
 */
function isDoubleFar(p) {
    return getPointerType(p) === pointer_type_1.PointerType.FAR && (p.segment.getUint32(p.byteOffset) & constants_1.POINTER_DOUBLE_FAR_MASK) !== 0;
}
exports.isDoubleFar = isDoubleFar;
/**
 * Quickly check to see if the pointer is "null". A "null" pointer is a zero word, equivalent to an empty struct
 * pointer.
 *
 * @param {Pointer} p The pointer to read from.
 * @returns {boolean} `true` if the pointer is "null".
 */
function isNull(p) {
    return p.segment.isWordZero(p.byteOffset);
}
exports.isNull = isNull;
/**
 * Relocate a pointer to the given destination, ensuring that it points to the same content. This will create far
 * pointers as needed if the content is in a different segment than the destination. After the relocation the source
 * pointer will be erased and is no longer valid.
 *
 * @param {Pointer} dst The desired location for the `src` pointer. Any existing contents will be erased before
 * relocating!
 * @param {Pointer} src The pointer to relocate.
 * @returns {void}
 */
function relocateTo(dst, src) {
    const t = followFars(src);
    const lo = t.segment.getUint8(t.byteOffset) & 0x03; // discard the offset
    const hi = t.segment.getUint32(t.byteOffset + 4);
    // Make sure anything dst was pointing to is wiped out.
    erase(dst);
    const res = initPointer(t.segment, t.byteOffset + 8 + getOffsetWords(t) * 8, dst);
    // Keep the low 2 bits and write the new offset.
    res.pointer.segment.setUint32(res.pointer.byteOffset, lo | (res.offsetWords << 2));
    // Keep the high 32 bits intact.
    res.pointer.segment.setUint32(res.pointer.byteOffset + 4, hi);
    erasePointer(src);
}
exports.relocateTo = relocateTo;
/**
 * Write a far pointer.
 *
 * @param {boolean} doubleFar Set to `true` if this is a double far pointer.
 * @param {number} offsetWords The offset, in words, to the target pointer.
 * @param {number} segmentId The segment the target pointer is located in.
 * @param {Pointer} p The pointer to write to.
 * @returns {void}
 */
function setFarPointer(doubleFar, offsetWords, segmentId, p) {
    const A = pointer_type_1.PointerType.FAR;
    const B = doubleFar ? 1 : 0;
    const C = offsetWords;
    const D = segmentId;
    p.segment.setUint32(p.byteOffset, A | (B << 2) | (C << 3));
    p.segment.setUint32(p.byteOffset + 4, D);
}
exports.setFarPointer = setFarPointer;
/**
 * Write a raw interface pointer.
 *
 * @param {number} capId The capability ID.
 * @param {Pointer} p The pointer to write to.
 * @returns {void}
 */
function setInterfacePointer(capId, p) {
    p.segment.setUint32(p.byteOffset, pointer_type_1.PointerType.OTHER);
    p.segment.setUint32(p.byteOffset + 4, capId);
}
exports.setInterfacePointer = setInterfacePointer;
/**
 * Write a raw list pointer.
 *
 * @param {number} offsetWords The number of words from the end of this pointer to the beginning of the list content.
 * @param {ListElementSize} size The size of each element in the list.
 * @param {number} length The number of elements in the list.
 * @param {Pointer} p The pointer to write to.
 * @param {ObjectSize} [compositeSize] For composite lists this describes the size of each element in this list. This
 * is required for composite lists.
 * @returns {void}
 */
function setListPointer(offsetWords, size, length, p, compositeSize) {
    const A = pointer_type_1.PointerType.LIST;
    const B = offsetWords;
    const C = size;
    let D = length;
    if (size === list_element_size_1.ListElementSize.COMPOSITE) {
        if (compositeSize === undefined) {
            throw new TypeError(errors_1.TYPE_COMPOSITE_SIZE_UNDEFINED);
        }
        D *= object_size_1.getWordLength(compositeSize);
    }
    p.segment.setUint32(p.byteOffset, A | (B << 2));
    p.segment.setUint32(p.byteOffset + 4, C | (D << 3));
}
exports.setListPointer = setListPointer;
/**
 * Write a raw struct pointer.
 *
 * @param {number} offsetWords The number of words from the end of this pointer to the beginning of the struct's data
 * section.
 * @param {ObjectSize} size An object describing the size of the struct.
 * @param {Pointer} p The pointer to write to.
 * @returns {void}
 */
function setStructPointer(offsetWords, size, p) {
    const A = pointer_type_1.PointerType.STRUCT;
    const B = offsetWords;
    const C = object_size_1.getDataWordLength(size);
    const D = size.pointerLength;
    p.segment.setUint32(p.byteOffset, A | (B << 2));
    p.segment.setUint16(p.byteOffset + 4, C);
    p.segment.setUint16(p.byteOffset + 6, D);
}
exports.setStructPointer = setStructPointer;
/**
 * Read some bits off a pointer to make sure it has the right pointer data.
 *
 * @param {PointerType} pointerType The expected pointer type.
 * @param {Pointer} p The pointer to validate.
 * @param {ListElementSize} [elementSize] For list pointers, the expected element size. Leave this
 * undefined for struct pointers.
 * @returns {void}
 */
function validate(pointerType, p, elementSize) {
    if (isNull(p))
        return;
    const t = followFars(p);
    // Check the pointer type.
    const A = t.segment.getUint32(t.byteOffset) & constants_1.POINTER_TYPE_MASK;
    if (A !== pointerType) {
        throw new Error(util_1.format(errors_1.PTR_WRONG_POINTER_TYPE, p, pointerType));
    }
    // Check the list element size, if provided.
    if (elementSize !== undefined) {
        const C = t.segment.getUint32(t.byteOffset + 4) & constants_1.LIST_SIZE_MASK;
        if (C !== elementSize) {
            throw new Error(util_1.format(errors_1.PTR_WRONG_LIST_TYPE, p, list_element_size_1.ListElementSize[elementSize]));
        }
    }
}
exports.validate = validate;
function copyFromList(src, dst) {
    if (dst._capnp.depthLimit <= 0)
        throw new Error(errors_1.PTR_DEPTH_LIMIT_EXCEEDED);
    const srcContent = getContent(src);
    const srcElementSize = getTargetListElementSize(src);
    const srcLength = getTargetListLength(src);
    let srcCompositeSize;
    let srcStructByteLength;
    let dstContent;
    if (srcElementSize === list_element_size_1.ListElementSize.POINTER) {
        dstContent = dst.segment.allocate(srcLength << 3);
        // Recursively copy each pointer in the list.
        for (let i = 0; i < srcLength; i++) {
            const srcPtr = new Pointer(srcContent.segment, srcContent.byteOffset + (i << 3), src._capnp.depthLimit - 1);
            const dstPtr = new Pointer(dstContent.segment, dstContent.byteOffset + (i << 3), dst._capnp.depthLimit - 1);
            copyFrom(srcPtr, dstPtr);
        }
    }
    else if (srcElementSize === list_element_size_1.ListElementSize.COMPOSITE) {
        srcCompositeSize = object_size_1.padToWord(getTargetCompositeListSize(src));
        srcStructByteLength = object_size_1.getByteLength(srcCompositeSize);
        dstContent = dst.segment.allocate(object_size_1.getByteLength(srcCompositeSize) * srcLength + 8);
        // Copy the tag word.
        dstContent.segment.copyWord(dstContent.byteOffset, srcContent.segment, srcContent.byteOffset - 8);
        // Copy the entire contents, including all pointers. This should be more efficient than making `srcLength`
        // copies to skip the pointer sections, and we're about to rewrite all those pointers anyway.
        // PERF: Skip this step if the composite struct only contains pointers.
        if (srcCompositeSize.dataByteLength > 0) {
            const wordLength = object_size_1.getWordLength(srcCompositeSize) * srcLength;
            dstContent.segment.copyWords(dstContent.byteOffset + 8, srcContent.segment, srcContent.byteOffset, wordLength);
        }
        // Recursively copy all the pointers in each struct.
        for (let i = 0; i < srcLength; i++) {
            for (let j = 0; j < srcCompositeSize.pointerLength; j++) {
                const offset = i * srcStructByteLength + srcCompositeSize.dataByteLength + (j << 3);
                const srcPtr = new Pointer(srcContent.segment, srcContent.byteOffset + offset, src._capnp.depthLimit - 1);
                const dstPtr = new Pointer(dstContent.segment, dstContent.byteOffset + offset + 8, dst._capnp.depthLimit - 1);
                copyFrom(srcPtr, dstPtr);
            }
        }
    }
    else {
        const byteLength = util_1.padToWord(srcElementSize === list_element_size_1.ListElementSize.BIT
            ? (srcLength + 7) >>> 3
            : getListElementByteLength(srcElementSize) * srcLength);
        const wordLength = byteLength >>> 3;
        dstContent = dst.segment.allocate(byteLength);
        // Copy all of the list contents word-by-word.
        dstContent.segment.copyWords(dstContent.byteOffset, srcContent.segment, srcContent.byteOffset, wordLength);
    }
    // Initialize the list pointer.
    const res = initPointer(dstContent.segment, dstContent.byteOffset, dst);
    setListPointer(res.offsetWords, srcElementSize, srcLength, res.pointer, srcCompositeSize);
}
exports.copyFromList = copyFromList;
function copyFromStruct(src, dst) {
    if (dst._capnp.depthLimit <= 0)
        throw new Error(errors_1.PTR_DEPTH_LIMIT_EXCEEDED);
    const srcContent = getContent(src);
    const srcSize = getTargetStructSize(src);
    const srcDataWordLength = object_size_1.getDataWordLength(srcSize);
    // Allocate space for the destination content.
    const dstContent = dst.segment.allocate(object_size_1.getByteLength(srcSize));
    // Copy the data section.
    dstContent.segment.copyWords(dstContent.byteOffset, srcContent.segment, srcContent.byteOffset, srcDataWordLength);
    // Copy the pointer section.
    for (let i = 0; i < srcSize.pointerLength; i++) {
        const offset = srcSize.dataByteLength + i * 8;
        const srcPtr = new Pointer(srcContent.segment, srcContent.byteOffset + offset, src._capnp.depthLimit - 1);
        const dstPtr = new Pointer(dstContent.segment, dstContent.byteOffset + offset, dst._capnp.depthLimit - 1);
        copyFrom(srcPtr, dstPtr);
    }
    // Don't touch dst if it's already initialized as a composite list pointer. With composite struct pointers there's
    // no pointer to copy here and we've already copied the contents.
    if (dst._capnp.compositeList)
        return;
    // Initialize the struct pointer.
    const res = initPointer(dstContent.segment, dstContent.byteOffset, dst);
    setStructPointer(res.offsetWords, srcSize, res.pointer);
}
exports.copyFromStruct = copyFromStruct;
/**
 * Track the allocation of a new Pointer object.
 *
 * This will decrement an internal counter tracking how many bytes have been traversed in the message so far. After
 * a certain limit, this method will throw an error in order to prevent a certain class of DoS attacks.
 *
 * @param {Message} message The message the pointer belongs to.
 * @param {Pointer} p The pointer being allocated.
 * @returns {void}
 */
function trackPointerAllocation(message, p) {
    message._capnp.traversalLimit -= 8;
    if (message._capnp.traversalLimit <= 0) {
        throw new Error(util_1.format(errors_1.PTR_TRAVERSAL_LIMIT_EXCEEDED, p));
    }
}
exports.trackPointerAllocation = trackPointerAllocation;

},{"../../constants":3,"../../errors":4,"../../util":51,"../list-element-size":13,"../object-size":16,"./orphan":33,"./pointer-allocation-result":34,"./pointer-type":36,"debug":52,"tslib":56}],38:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.checkPointerBounds = exports.checkDataBounds = exports.testWhich = exports.setVoid = exports.setUint8 = exports.setUint64 = exports.setUint32 = exports.setUint16 = exports.setText = exports.setPointer = exports.setInt8 = exports.setInt64 = exports.setInt32 = exports.setInt16 = exports.setFloat64 = exports.setFloat32 = exports.setBit = exports.initList = exports.initData = exports.getVoid = exports.getUint8 = exports.getUint64 = exports.getUint32 = exports.getUint16 = exports.getText = exports.getStruct = exports.getSize = exports.getPointerSection = exports.getPointerAs = exports.getPointer = exports.getList = exports.getInt8 = exports.getInt64 = exports.getInt32 = exports.getInt16 = exports.getFloat64 = exports.getFloat32 = exports.getDataSection = exports.getData = exports.getBit = exports.getAs = exports.disown = exports.adopt = exports.resize = exports.initStructAt = exports.initStruct = exports.Struct = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("../../constants");
const index_1 = require("../../types/index");
const util_1 = require("../../util");
const list_element_size_1 = require("../list-element-size");
const object_size_1 = require("../object-size");
const data_1 = require("./data");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const pointer_type_1 = require("./pointer-type");
const text_1 = require("./text");
const errors_1 = require("../../errors");
const trace = debug_1.default("capnp:struct");
trace("load");
// Used to apply bit masks (default values).
const TMP_WORD = new DataView(new ArrayBuffer(8));
class Struct extends pointer_1.Pointer {
    /**
     * Create a new pointer to a struct.
     *
     * @constructor {Struct}
     * @param {Segment} segment The segment the pointer resides in.
     * @param {number} byteOffset The offset from the beginning of the segment to the beginning of the pointer data.
     * @param {any} [depthLimit=MAX_DEPTH] The nesting depth limit for this object.
     * @param {number} [compositeIndex] If set, then this pointer is actually a reference to a composite list
     * (`this._getPointerTargetType() === PointerType.LIST`), and this number is used as the index of the struct within
     * the list. It is not valid to call `initStruct()` on a composite struct – the struct contents are initialized when
     * the list pointer is initialized.
     */
    constructor(segment, byteOffset, depthLimit = constants_1.MAX_DEPTH, compositeIndex) {
        super(segment, byteOffset, depthLimit);
        this._capnp.compositeIndex = compositeIndex;
        this._capnp.compositeList = compositeIndex !== undefined;
    }
    static toString() {
        return this._capnp.displayName;
    }
    toString() {
        return (`Struct_${super.toString()}` +
            `${this._capnp.compositeIndex === undefined ? "" : `,ci:${this._capnp.compositeIndex}`}`);
    }
}
exports.Struct = Struct;
Struct._capnp = {
    displayName: "Struct",
};
Struct.getAs = getAs;
Struct.getBit = getBit;
Struct.getData = getData;
Struct.getFloat32 = getFloat32;
Struct.getFloat64 = getFloat64;
Struct.getUint8 = getUint8;
Struct.getUint16 = getUint16;
Struct.getUint32 = getUint32;
Struct.getUint64 = getUint64;
Struct.getInt8 = getInt8;
Struct.getInt16 = getInt16;
Struct.getInt32 = getInt32;
Struct.getInt64 = getInt64;
Struct.getList = getList;
Struct.getPointer = getPointer;
Struct.getPointerAs = getPointerAs;
Struct.getStruct = getStruct;
Struct.getText = getText;
Struct.initData = initData;
Struct.initList = initList;
Struct.initStruct = initStruct;
Struct.initStructAt = initStructAt;
Struct.setBit = setBit;
Struct.setFloat32 = setFloat32;
Struct.setFloat64 = setFloat64;
Struct.setUint8 = setUint8;
Struct.setUint16 = setUint16;
Struct.setUint32 = setUint32;
Struct.setUint64 = setUint64;
Struct.setInt8 = setInt8;
Struct.setInt16 = setInt16;
Struct.setInt32 = setInt32;
Struct.setInt64 = setInt64;
Struct.setText = setText;
Struct.testWhich = testWhich;
/**
 * Initialize a struct with the provided object size. This will allocate new space for the struct contents, ideally in
 * the same segment as this pointer.
 *
 * @param {ObjectSize} size An object describing the size of the struct's data and pointer sections.
 * @param {Struct} s The struct to initialize.
 * @returns {void}
 */
function initStruct(size, s) {
    if (s._capnp.compositeIndex !== undefined) {
        throw new Error(util_1.format(errors_1.PTR_INIT_COMPOSITE_STRUCT, s));
    }
    // Make sure to clear existing contents before overwriting the pointer data (erase is a noop if already empty).
    pointer_1.erase(s);
    const c = s.segment.allocate(object_size_1.getByteLength(size));
    const res = pointer_1.initPointer(c.segment, c.byteOffset, s);
    pointer_1.setStructPointer(res.offsetWords, size, res.pointer);
}
exports.initStruct = initStruct;
function initStructAt(index, StructClass, p) {
    const s = getPointerAs(index, StructClass, p);
    initStruct(StructClass._capnp.size, s);
    return s;
}
exports.initStructAt = initStructAt;
/**
 * Make a shallow copy of a struct's contents and update the pointer to point to the new content. The data and pointer
 * sections will be resized to the provided size.
 *
 * WARNING: This method can cause data loss if `dstSize` is smaller than the original size!
 *
 * @param {ObjectSize} dstSize The desired size for the struct contents.
 * @param {Struct} s The struct to resize.
 * @returns {void}
 */
function resize(dstSize, s) {
    const srcSize = getSize(s);
    const srcContent = pointer_1.getContent(s);
    const dstContent = s.segment.allocate(object_size_1.getByteLength(dstSize));
    // Only copy the data section for now. The pointer section will need to be rewritten.
    dstContent.segment.copyWords(dstContent.byteOffset, srcContent.segment, srcContent.byteOffset, Math.min(object_size_1.getDataWordLength(srcSize), object_size_1.getDataWordLength(dstSize)));
    const res = pointer_1.initPointer(dstContent.segment, dstContent.byteOffset, s);
    pointer_1.setStructPointer(res.offsetWords, dstSize, res.pointer);
    // Iterate through the new pointer section and update the offsets so they point to the right place. This is a bit
    // more complicated than it appears due to the fact that the original pointers could have been far pointers, and
    // the new pointers might need to be allocated as far pointers if the segment is full.
    for (let i = 0; i < Math.min(srcSize.pointerLength, dstSize.pointerLength); i++) {
        const srcPtr = new pointer_1.Pointer(srcContent.segment, srcContent.byteOffset + srcSize.dataByteLength + i * 8);
        if (pointer_1.isNull(srcPtr)) {
            // If source pointer is null, leave the destination pointer as default null.
            continue;
        }
        const srcPtrTarget = pointer_1.followFars(srcPtr);
        const srcPtrContent = pointer_1.getContent(srcPtr);
        const dstPtr = new pointer_1.Pointer(dstContent.segment, dstContent.byteOffset + dstSize.dataByteLength + i * 8);
        // For composite lists the offset needs to point to the tag word, not the first element which is what getContent
        // returns.
        if (pointer_1.getTargetPointerType(srcPtr) === pointer_type_1.PointerType.LIST &&
            pointer_1.getTargetListElementSize(srcPtr) === list_element_size_1.ListElementSize.COMPOSITE) {
            srcPtrContent.byteOffset -= 8;
        }
        const r = pointer_1.initPointer(srcPtrContent.segment, srcPtrContent.byteOffset, dstPtr);
        // Read the old pointer data, but discard the original offset.
        const a = srcPtrTarget.segment.getUint8(srcPtrTarget.byteOffset) & 0x03;
        const b = srcPtrTarget.segment.getUint32(srcPtrTarget.byteOffset + 4);
        r.pointer.segment.setUint32(r.pointer.byteOffset, a | (r.offsetWords << 2));
        r.pointer.segment.setUint32(r.pointer.byteOffset + 4, b);
    }
    // Zero out the old data and pointer sections.
    srcContent.segment.fillZeroWords(srcContent.byteOffset, object_size_1.getWordLength(srcSize));
}
exports.resize = resize;
function adopt(src, s) {
    if (s._capnp.compositeIndex !== undefined) {
        throw new Error(util_1.format(errors_1.PTR_ADOPT_COMPOSITE_STRUCT, s));
    }
    pointer_1.Pointer.adopt(src, s);
}
exports.adopt = adopt;
function disown(s) {
    if (s._capnp.compositeIndex !== undefined) {
        throw new Error(util_1.format(errors_1.PTR_DISOWN_COMPOSITE_STRUCT, s));
    }
    return pointer_1.Pointer.disown(s);
}
exports.disown = disown;
/**
 * Convert a struct to a struct of the provided class. Particularly useful when casting to nested group types.
 *
 * @protected
 * @template T
 * @param {StructCtor<T>} StructClass The struct class to convert to. Not particularly useful if `Struct`.
 * @param {Struct} s The struct to convert.
 * @returns {T} A new instance of the desired struct class pointing to the same location.
 */
function getAs(StructClass, s) {
    return new StructClass(s.segment, s.byteOffset, s._capnp.depthLimit, s._capnp.compositeIndex);
}
exports.getAs = getAs;
/**
 * Read a boolean (bit) value out of a struct.
 *
 * @protected
 * @param {number} bitOffset The offset in **bits** from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {boolean} The value.
 */
function getBit(bitOffset, s, defaultMask) {
    const byteOffset = Math.floor(bitOffset / 8);
    const bitMask = 1 << bitOffset % 8;
    checkDataBounds(byteOffset, 1, s);
    const ds = getDataSection(s);
    const v = ds.segment.getUint8(ds.byteOffset + byteOffset);
    if (defaultMask === undefined)
        return (v & bitMask) !== 0;
    const defaultValue = defaultMask.getUint8(0);
    return ((v ^ defaultValue) & bitMask) !== 0;
}
exports.getBit = getBit;
function getData(index, s, defaultValue) {
    checkPointerBounds(index, s);
    const ps = getPointerSection(s);
    ps.byteOffset += index * 8;
    const l = new data_1.Data(ps.segment, ps.byteOffset, s._capnp.depthLimit - 1);
    if (pointer_1.isNull(l)) {
        if (defaultValue) {
            pointer_1.Pointer.copyFrom(defaultValue, l);
        }
        else {
            list_1.List.initList(list_element_size_1.ListElementSize.BYTE, 0, l);
        }
    }
    return l;
}
exports.getData = getData;
function getDataSection(s) {
    return pointer_1.getContent(s);
}
exports.getDataSection = getDataSection;
/**
 * Read a float32 value out of a struct.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getFloat32(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 4, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getFloat32(ds.byteOffset + byteOffset);
    }
    const v = ds.segment.getUint32(ds.byteOffset + byteOffset) ^ defaultMask.getUint32(0, true);
    TMP_WORD.setUint32(0, v, constants_1.NATIVE_LITTLE_ENDIAN);
    return TMP_WORD.getFloat32(0, constants_1.NATIVE_LITTLE_ENDIAN);
}
exports.getFloat32 = getFloat32;
/**
 * Read a float64 value out of this segment.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getFloat64(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 8, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined) {
        const lo = ds.segment.getUint32(ds.byteOffset + byteOffset) ^ defaultMask.getUint32(0, true);
        const hi = ds.segment.getUint32(ds.byteOffset + byteOffset + 4) ^ defaultMask.getUint32(4, true);
        TMP_WORD.setUint32(0, lo, constants_1.NATIVE_LITTLE_ENDIAN);
        TMP_WORD.setUint32(4, hi, constants_1.NATIVE_LITTLE_ENDIAN);
        return TMP_WORD.getFloat64(0, constants_1.NATIVE_LITTLE_ENDIAN);
    }
    return ds.segment.getFloat64(ds.byteOffset + byteOffset);
}
exports.getFloat64 = getFloat64;
/**
 * Read an int16 value out of this segment.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getInt16(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 2, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getInt16(ds.byteOffset + byteOffset);
    }
    const v = ds.segment.getUint16(ds.byteOffset + byteOffset) ^ defaultMask.getUint16(0, true);
    TMP_WORD.setUint16(0, v, constants_1.NATIVE_LITTLE_ENDIAN);
    return TMP_WORD.getInt16(0, constants_1.NATIVE_LITTLE_ENDIAN);
}
exports.getInt16 = getInt16;
/**
 * Read an int32 value out of this segment.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getInt32(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 4, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getInt32(ds.byteOffset + byteOffset);
    }
    const v = ds.segment.getUint32(ds.byteOffset + byteOffset) ^ defaultMask.getUint16(0, true);
    TMP_WORD.setUint32(0, v, constants_1.NATIVE_LITTLE_ENDIAN);
    return TMP_WORD.getInt32(0, constants_1.NATIVE_LITTLE_ENDIAN);
}
exports.getInt32 = getInt32;
/**
 * Read an int64 value out of this segment.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getInt64(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 8, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getInt64(ds.byteOffset + byteOffset);
    }
    const lo = ds.segment.getUint32(ds.byteOffset + byteOffset) ^ defaultMask.getUint32(0, true);
    const hi = ds.segment.getUint32(ds.byteOffset + byteOffset + 4) ^ defaultMask.getUint32(4, true);
    TMP_WORD.setUint32(0, lo, constants_1.NATIVE_LITTLE_ENDIAN);
    TMP_WORD.setUint32(4, hi, constants_1.NATIVE_LITTLE_ENDIAN);
    return new index_1.Int64(new Uint8Array(TMP_WORD.buffer.slice(0)));
}
exports.getInt64 = getInt64;
/**
 * Read an int8 value out of this segment.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getInt8(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 1, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getInt8(ds.byteOffset + byteOffset);
    }
    const v = ds.segment.getUint8(ds.byteOffset + byteOffset) ^ defaultMask.getUint8(0);
    TMP_WORD.setUint8(0, v);
    return TMP_WORD.getInt8(0);
}
exports.getInt8 = getInt8;
function getList(index, ListClass, s, defaultValue) {
    checkPointerBounds(index, s);
    const ps = getPointerSection(s);
    ps.byteOffset += index * 8;
    const l = new ListClass(ps.segment, ps.byteOffset, s._capnp.depthLimit - 1);
    if (pointer_1.isNull(l)) {
        if (defaultValue) {
            pointer_1.Pointer.copyFrom(defaultValue, l);
        }
        else {
            list_1.List.initList(ListClass._capnp.size, 0, l, ListClass._capnp.compositeSize);
        }
    }
    else if (ListClass._capnp.compositeSize !== undefined) {
        // If this is a composite list we need to be sure the composite elements are big enough to hold everything as
        // specified in the schema. If the new schema has added fields we'll need to "resize" (shallow-copy) the list so
        // it has room for the new fields.
        const srcSize = pointer_1.getTargetCompositeListSize(l);
        const dstSize = ListClass._capnp.compositeSize;
        if (dstSize.dataByteLength > srcSize.dataByteLength || dstSize.pointerLength > srcSize.pointerLength) {
            const srcContent = pointer_1.getContent(l);
            const srcLength = pointer_1.getTargetListLength(l);
            trace("resizing composite list %s due to protocol upgrade, new size: %d", l, object_size_1.getByteLength(dstSize) * srcLength);
            // Allocate an extra 8 bytes for the tag.
            const dstContent = l.segment.allocate(object_size_1.getByteLength(dstSize) * srcLength + 8);
            const res = pointer_1.initPointer(dstContent.segment, dstContent.byteOffset, l);
            pointer_1.setListPointer(res.offsetWords, ListClass._capnp.size, srcLength, res.pointer, dstSize);
            // Write the new tag word.
            pointer_1.setStructPointer(srcLength, dstSize, dstContent);
            // Seek ahead past the tag word before copying the content.
            dstContent.byteOffset += 8;
            for (let i = 0; i < srcLength; i++) {
                const srcElementOffset = srcContent.byteOffset + i * object_size_1.getByteLength(srcSize);
                const dstElementOffset = dstContent.byteOffset + i * object_size_1.getByteLength(dstSize);
                // Copy the data section.
                dstContent.segment.copyWords(dstElementOffset, srcContent.segment, srcElementOffset, object_size_1.getWordLength(srcSize));
                // Iterate through the pointers and update the offsets so they point to the right place.
                for (let j = 0; j < srcSize.pointerLength; j++) {
                    const srcPtr = new pointer_1.Pointer(srcContent.segment, srcElementOffset + srcSize.dataByteLength + j * 8);
                    const dstPtr = new pointer_1.Pointer(dstContent.segment, dstElementOffset + dstSize.dataByteLength + j * 8);
                    const srcPtrTarget = pointer_1.followFars(srcPtr);
                    const srcPtrContent = pointer_1.getContent(srcPtr);
                    if (pointer_1.getTargetPointerType(srcPtr) === pointer_type_1.PointerType.LIST &&
                        pointer_1.getTargetListElementSize(srcPtr) === list_element_size_1.ListElementSize.COMPOSITE) {
                        srcPtrContent.byteOffset -= 8;
                    }
                    const r = pointer_1.initPointer(srcPtrContent.segment, srcPtrContent.byteOffset, dstPtr);
                    // Read the old pointer data, but discard the original offset.
                    const a = srcPtrTarget.segment.getUint8(srcPtrTarget.byteOffset) & 0x03;
                    const b = srcPtrTarget.segment.getUint32(srcPtrTarget.byteOffset + 4);
                    r.pointer.segment.setUint32(r.pointer.byteOffset, a | (r.offsetWords << 2));
                    r.pointer.segment.setUint32(r.pointer.byteOffset + 4, b);
                }
            }
            // Zero out the old content.
            srcContent.segment.fillZeroWords(srcContent.byteOffset, object_size_1.getWordLength(srcSize) * srcLength);
        }
    }
    return l;
}
exports.getList = getList;
function getPointer(index, s) {
    checkPointerBounds(index, s);
    const ps = getPointerSection(s);
    ps.byteOffset += index * 8;
    return new pointer_1.Pointer(ps.segment, ps.byteOffset, s._capnp.depthLimit - 1);
}
exports.getPointer = getPointer;
function getPointerAs(index, PointerClass, s) {
    checkPointerBounds(index, s);
    const ps = getPointerSection(s);
    ps.byteOffset += index * 8;
    return new PointerClass(ps.segment, ps.byteOffset, s._capnp.depthLimit - 1);
}
exports.getPointerAs = getPointerAs;
function getPointerSection(s) {
    const ps = pointer_1.getContent(s);
    ps.byteOffset += util_1.padToWord(getSize(s).dataByteLength);
    return ps;
}
exports.getPointerSection = getPointerSection;
function getSize(s) {
    if (s._capnp.compositeIndex !== undefined) {
        // For composite lists the object size is stored in a tag word right before the content.
        const c = pointer_1.getContent(s, true);
        c.byteOffset -= 8;
        return pointer_1.getStructSize(c);
    }
    return pointer_1.getTargetStructSize(s);
}
exports.getSize = getSize;
function getStruct(index, StructClass, s, defaultValue) {
    const t = getPointerAs(index, StructClass, s);
    if (pointer_1.isNull(t)) {
        if (defaultValue) {
            pointer_1.Pointer.copyFrom(defaultValue, t);
        }
        else {
            initStruct(StructClass._capnp.size, t);
        }
    }
    else {
        pointer_1.validate(pointer_type_1.PointerType.STRUCT, t);
        const ts = pointer_1.getTargetStructSize(t);
        // This can happen when reading a struct that was constructed with an older version of the same schema, and new
        // fields were added to the struct. A shallow copy of the struct will be made so that there's enough room for the
        // data and pointer sections. This will unfortunately leave a "hole" of zeroes in the message, but that hole will
        // at least compress well.
        if (ts.dataByteLength < StructClass._capnp.size.dataByteLength ||
            ts.pointerLength < StructClass._capnp.size.pointerLength) {
            trace("need to resize child struct %s", t);
            resize(StructClass._capnp.size, t);
        }
    }
    return t;
}
exports.getStruct = getStruct;
function getText(index, s, defaultValue) {
    const t = text_1.Text.fromPointer(getPointer(index, s));
    // FIXME: This will perform an unnecessary string<>ArrayBuffer roundtrip.
    if (pointer_1.isNull(t) && defaultValue)
        t.set(0, defaultValue);
    return t.get(0);
}
exports.getText = getText;
/**
 * Read an uint16 value out of a struct..
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getUint16(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 2, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getUint16(ds.byteOffset + byteOffset);
    }
    return ds.segment.getUint16(ds.byteOffset + byteOffset) ^ defaultMask.getUint16(0, true);
}
exports.getUint16 = getUint16;
/**
 * Read an uint32 value out of a struct.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getUint32(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 4, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getUint32(ds.byteOffset + byteOffset);
    }
    return ds.segment.getUint32(ds.byteOffset + byteOffset) ^ defaultMask.getUint32(0, true);
}
exports.getUint32 = getUint32;
/**
 * Read an uint64 value out of a struct.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getUint64(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 8, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getUint64(ds.byteOffset + byteOffset);
    }
    const lo = ds.segment.getUint32(ds.byteOffset + byteOffset) ^ defaultMask.getUint32(0, true);
    const hi = ds.segment.getUint32(ds.byteOffset + byteOffset + 4) ^ defaultMask.getUint32(4, true);
    TMP_WORD.setUint32(0, lo, constants_1.NATIVE_LITTLE_ENDIAN);
    TMP_WORD.setUint32(4, hi, constants_1.NATIVE_LITTLE_ENDIAN);
    return new index_1.Uint64(new Uint8Array(TMP_WORD.buffer.slice(0)));
}
exports.getUint64 = getUint64;
/**
 * Read an uint8 value out of a struct.
 *
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {Struct} s The struct to read from.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {number} The value.
 */
function getUint8(byteOffset, s, defaultMask) {
    checkDataBounds(byteOffset, 1, s);
    const ds = getDataSection(s);
    if (defaultMask === undefined) {
        return ds.segment.getUint8(ds.byteOffset + byteOffset);
    }
    return ds.segment.getUint8(ds.byteOffset + byteOffset) ^ defaultMask.getUint8(0);
}
exports.getUint8 = getUint8;
function getVoid() {
    throw new Error(errors_1.INVARIANT_UNREACHABLE_CODE);
}
exports.getVoid = getVoid;
function initData(index, length, s) {
    checkPointerBounds(index, s);
    const ps = getPointerSection(s);
    ps.byteOffset += index * 8;
    const l = new data_1.Data(ps.segment, ps.byteOffset, s._capnp.depthLimit - 1);
    pointer_1.erase(l);
    list_1.List.initList(list_element_size_1.ListElementSize.BYTE, length, l);
    return l;
}
exports.initData = initData;
function initList(index, ListClass, length, s) {
    checkPointerBounds(index, s);
    const ps = getPointerSection(s);
    ps.byteOffset += index * 8;
    const l = new ListClass(ps.segment, ps.byteOffset, s._capnp.depthLimit - 1);
    pointer_1.erase(l);
    list_1.List.initList(ListClass._capnp.size, length, l, ListClass._capnp.compositeSize);
    return l;
}
exports.initList = initList;
/**
 * Write a boolean (bit) value to the struct.
 *
 * @protected
 * @param {number} bitOffset The offset in **bits** from the start of the data section.
 * @param {boolean} value The value to write (writes a 0 for `false`, 1 for `true`).
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setBit(bitOffset, value, s, defaultMask) {
    const byteOffset = Math.floor(bitOffset / 8);
    const bitMask = 1 << bitOffset % 8;
    checkDataBounds(byteOffset, 1, s);
    const ds = getDataSection(s);
    const b = ds.segment.getUint8(ds.byteOffset + byteOffset);
    // If the default mask bit is set, that means `true` values are actually written as `0`.
    if (defaultMask !== undefined) {
        value = (defaultMask.getUint8(0) & bitMask) !== 0 ? !value : value;
    }
    ds.segment.setUint8(ds.byteOffset + byteOffset, value ? b | bitMask : b & ~bitMask);
}
exports.setBit = setBit;
/**
 * Write a primitive float32 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setFloat32(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 4, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined) {
        TMP_WORD.setFloat32(0, value, constants_1.NATIVE_LITTLE_ENDIAN);
        const v = TMP_WORD.getUint32(0, constants_1.NATIVE_LITTLE_ENDIAN) ^ defaultMask.getUint32(0, true);
        ds.segment.setUint32(ds.byteOffset + byteOffset, v);
        return;
    }
    ds.segment.setFloat32(ds.byteOffset + byteOffset, value);
}
exports.setFloat32 = setFloat32;
/**
 * Write a primitive float64 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setFloat64(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 8, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined) {
        TMP_WORD.setFloat64(0, value, constants_1.NATIVE_LITTLE_ENDIAN);
        const lo = TMP_WORD.getUint32(0, constants_1.NATIVE_LITTLE_ENDIAN) ^ defaultMask.getUint32(0, true);
        const hi = TMP_WORD.getUint32(4, constants_1.NATIVE_LITTLE_ENDIAN) ^ defaultMask.getUint32(4, true);
        ds.segment.setUint32(ds.byteOffset + byteOffset, lo);
        ds.segment.setUint32(ds.byteOffset + byteOffset + 4, hi);
        return;
    }
    ds.segment.setFloat64(ds.byteOffset + byteOffset, value);
}
exports.setFloat64 = setFloat64;
/**
 * Write a primitive int16 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setInt16(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 2, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined) {
        TMP_WORD.setInt16(0, value, constants_1.NATIVE_LITTLE_ENDIAN);
        const v = TMP_WORD.getUint16(0, constants_1.NATIVE_LITTLE_ENDIAN) ^ defaultMask.getUint16(0, true);
        ds.segment.setUint16(ds.byteOffset + byteOffset, v);
        return;
    }
    ds.segment.setInt16(ds.byteOffset + byteOffset, value);
}
exports.setInt16 = setInt16;
/**
 * Write a primitive int32 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setInt32(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 4, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined) {
        TMP_WORD.setInt32(0, value, constants_1.NATIVE_LITTLE_ENDIAN);
        const v = TMP_WORD.getUint32(0, constants_1.NATIVE_LITTLE_ENDIAN) ^ defaultMask.getUint32(0, true);
        ds.segment.setUint32(ds.byteOffset + byteOffset, v);
        return;
    }
    ds.segment.setInt32(ds.byteOffset + byteOffset, value);
}
exports.setInt32 = setInt32;
/**
 * Write a primitive int64 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setInt64(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 8, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined) {
        // PERF: We could cast the Int64 to a DataView to apply the mask using four 32-bit reads, but we already have a
        // typed array so avoiding the object allocation turns out to be slightly faster. Int64 is guaranteed to be in
        // little-endian format by design.
        for (let i = 0; i < 8; i++) {
            ds.segment.setUint8(ds.byteOffset + byteOffset + i, value.buffer[i] ^ defaultMask.getUint8(i));
        }
        return;
    }
    ds.segment.setInt64(ds.byteOffset + byteOffset, value);
}
exports.setInt64 = setInt64;
/**
 * Write a primitive int8 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setInt8(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 1, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined) {
        TMP_WORD.setInt8(0, value);
        const v = TMP_WORD.getUint8(0) ^ defaultMask.getUint8(0);
        ds.segment.setUint8(ds.byteOffset + byteOffset, v);
        return;
    }
    ds.segment.setInt8(ds.byteOffset + byteOffset, value);
}
exports.setInt8 = setInt8;
function setPointer(index, value, s) {
    pointer_1.copyFrom(value, getPointer(index, s));
}
exports.setPointer = setPointer;
function setText(index, value, s) {
    text_1.Text.fromPointer(getPointer(index, s)).set(0, value);
}
exports.setText = setText;
/**
 * Write a primitive uint16 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setUint16(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 2, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined)
        value ^= defaultMask.getUint16(0, true);
    ds.segment.setUint16(ds.byteOffset + byteOffset, value);
}
exports.setUint16 = setUint16;
/**
 * Write a primitive uint32 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setUint32(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 4, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined)
        value ^= defaultMask.getUint32(0, true);
    ds.segment.setUint32(ds.byteOffset + byteOffset, value);
}
exports.setUint32 = setUint32;
/**
 * Write a primitive uint64 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setUint64(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 8, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined) {
        // PERF: We could cast the Uint64 to a DataView to apply the mask using four 32-bit reads, but we already have a
        // typed array so avoiding the object allocation turns out to be slightly faster. Uint64 is guaranteed to be in
        // little-endian format by design.
        for (let i = 0; i < 8; i++) {
            ds.segment.setUint8(ds.byteOffset + byteOffset + i, value.buffer[i] ^ defaultMask.getUint8(i));
        }
        return;
    }
    ds.segment.setUint64(ds.byteOffset + byteOffset, value);
}
exports.setUint64 = setUint64;
/**
 * Write a primitive uint8 value to the struct.
 *
 * @protected
 * @param {number} byteOffset The offset in bytes from the start of the data section.
 * @param {number} value The value to write.
 * @param {Struct} s The struct to write to.
 * @param {DataView} [defaultMask] The default value as a DataView.
 * @returns {void}
 */
function setUint8(byteOffset, value, s, defaultMask) {
    checkDataBounds(byteOffset, 1, s);
    const ds = getDataSection(s);
    if (defaultMask !== undefined)
        value ^= defaultMask.getUint8(0);
    ds.segment.setUint8(ds.byteOffset + byteOffset, value);
}
exports.setUint8 = setUint8;
function setVoid() {
    throw new Error(errors_1.INVARIANT_UNREACHABLE_CODE);
}
exports.setVoid = setVoid;
function testWhich(name, found, wanted, s) {
    if (found !== wanted) {
        throw new Error(util_1.format(errors_1.PTR_INVALID_UNION_ACCESS, s, name, found, wanted));
    }
}
exports.testWhich = testWhich;
function checkDataBounds(byteOffset, byteLength, s) {
    const dataByteLength = getSize(s).dataByteLength;
    if (byteOffset < 0 || byteLength < 0 || byteOffset + byteLength > dataByteLength) {
        throw new Error(util_1.format(errors_1.PTR_STRUCT_DATA_OUT_OF_BOUNDS, s, byteLength, byteOffset, dataByteLength));
    }
}
exports.checkDataBounds = checkDataBounds;
function checkPointerBounds(index, s) {
    const pointerLength = getSize(s).pointerLength;
    if (index < 0 || index >= pointerLength) {
        throw new Error(util_1.format(errors_1.PTR_STRUCT_POINTER_OUT_OF_BOUNDS, s, index, pointerLength));
    }
}
exports.checkPointerBounds = checkPointerBounds;

},{"../../constants":3,"../../errors":4,"../../types/index":48,"../../util":51,"../list-element-size":13,"../object-size":16,"./data":22,"./list":32,"./pointer":37,"./pointer-type":36,"./text":40,"debug":52,"tslib":56}],39:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.TextList = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const text_1 = require("./text");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class TextList extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        c.byteOffset += index * 8;
        return text_1.Text.fromPointer(c).get(0);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.byteOffset += index * 8;
        text_1.Text.fromPointer(c).set(0, value);
    }
    toString() {
        return `Text_${super.toString()}`;
    }
}
exports.TextList = TextList;
TextList._capnp = {
    displayName: "List<Text>",
    size: list_element_size_1.ListElementSize.POINTER
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"./text":40,"debug":52,"tslib":56}],40:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Text = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const util_1 = require("../../util");
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const pointer_type_1 = require("./pointer-type");
const trace = debug_1.default("capnp:text");
trace("load");
class Text extends list_1.List {
    static fromPointer(pointer) {
        pointer_1.validate(pointer_type_1.PointerType.LIST, pointer, list_element_size_1.ListElementSize.BYTE);
        return textFromPointerUnchecked(pointer);
    }
    /**
     * Read a utf-8 encoded string value from this pointer.
     *
     * @param {number} [index] The index at which to start reading; defaults to zero.
     * @returns {string} The string value.
     */
    get(index = 0) {
        if (index !== 0) {
            trace("Called get() on %s with a strange index (%d).", this, index);
        }
        if (pointer_1.isNull(this))
            return "";
        const c = pointer_1.getContent(this);
        // Remember to exclude the NUL byte.
        return util_1.decodeUtf8(new Uint8Array(c.segment.buffer, c.byteOffset + index, this.getLength() - index));
    }
    /**
     * Get the number of utf-8 encoded bytes in this text. This does **not** include the NUL byte.
     *
     * @returns {number} The number of bytes allocated for the text.
     */
    getLength() {
        return super.getLength() - 1;
    }
    /**
     * Write a utf-8 encoded string value starting at the specified index.
     *
     * @param {number} index The index at which to start copying the string. Note that if this is not zero the bytes
     * before `index` will be left as-is. All bytes after `index` will be overwritten.
     * @param {string} value The string value to set.
     * @returns {void}
     */
    set(index, value) {
        if (index !== 0) {
            trace("Called set() on %s with a strange index (%d).", this, index);
        }
        const src = util_1.encodeUtf8(value);
        const dstLength = src.byteLength + index;
        let c;
        let original;
        // TODO: Consider reusing existing space if list is already initialized and there's enough room for the value.
        if (!pointer_1.isNull(this)) {
            c = pointer_1.getContent(this);
            // Only copy bytes that will remain after copying. Everything after `index` should end up truncated.
            let originalLength = this.getLength();
            if (originalLength >= index) {
                originalLength = index;
            }
            else {
                trace("%d byte gap exists between original text and new text in %s.", index - originalLength, this);
            }
            original = new Uint8Array(c.segment.buffer.slice(c.byteOffset, c.byteOffset + Math.min(originalLength, index)));
            pointer_1.erase(this);
        }
        // Always allocate an extra byte for the NUL byte.
        list_1.initList(list_element_size_1.ListElementSize.BYTE, dstLength + 1, this);
        c = pointer_1.getContent(this);
        const dst = new Uint8Array(c.segment.buffer, c.byteOffset, dstLength);
        if (original)
            dst.set(original);
        dst.set(src, index);
    }
    toString() {
        return `Text_${super.toString()}`;
    }
}
exports.Text = Text;
function textFromPointerUnchecked(pointer) {
    return new Text(pointer.segment, pointer.byteOffset, pointer._capnp.depthLimit);
}

},{"../../util":51,"../list-element-size":13,"./list":32,"./pointer":37,"./pointer-type":36,"debug":52,"tslib":56}],41:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Uint16List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Uint16List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getUint16(c.byteOffset + index * 2);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setUint16(c.byteOffset + index * 2, value);
    }
    toString() {
        return `Uint16_${super.toString()}`;
    }
}
exports.Uint16List = Uint16List;
Uint16List._capnp = {
    displayName: "List<Uint16>",
    size: list_element_size_1.ListElementSize.BYTE_2
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],42:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Uint32List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Uint32List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getUint32(c.byteOffset + index * 4);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setUint32(c.byteOffset + index * 4, value);
    }
    toString() {
        return `Uint32_${super.toString()}`;
    }
}
exports.Uint32List = Uint32List;
Uint32List._capnp = {
    displayName: "List<Uint32>",
    size: list_element_size_1.ListElementSize.BYTE_4
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],43:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Uint64List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Uint64List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getUint64(c.byteOffset + index * 8);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setUint64(c.byteOffset + index * 8, value);
    }
    toString() {
        return `Uint64_${super.toString()}`;
    }
}
exports.Uint64List = Uint64List;
Uint64List._capnp = {
    displayName: "List<Uint64>",
    size: list_element_size_1.ListElementSize.BYTE_8,
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],44:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Uint8List = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const list_element_size_1 = require("../list-element-size");
const list_1 = require("./list");
const pointer_1 = require("./pointer");
const trace = debug_1.default("capnp:list:composite");
trace("load");
class Uint8List extends list_1.List {
    get(index) {
        const c = pointer_1.getContent(this);
        return c.segment.getUint8(c.byteOffset + index);
    }
    set(index, value) {
        const c = pointer_1.getContent(this);
        c.segment.setUint8(c.byteOffset + index, value);
    }
    toString() {
        return `Uint8_${super.toString()}`;
    }
}
exports.Uint8List = Uint8List;
Uint8List._capnp = {
    displayName: "List<Uint8>",
    size: list_element_size_1.ListElementSize.BYTE
};

},{"../list-element-size":13,"./list":32,"./pointer":37,"debug":52,"tslib":56}],45:[function(require,module,exports){
"use strict";
/**
 * Why would anyone **SANE** ever use this!?
 *
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.VoidList = void 0;
const pointer_list_1 = require("./pointer-list");
const void_1 = require("./void");
exports.VoidList = pointer_list_1.PointerList(void_1.Void);

},{"./pointer-list":35,"./void":46}],46:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.VOID = exports.Void = void 0;
const object_size_1 = require("../object-size");
const struct_1 = require("./struct");
class Void extends struct_1.Struct {
}
exports.Void = Void;
Void._capnp = {
    displayName: "Void",
    id: "0",
    size: new object_size_1.ObjectSize(0, 0)
};
// This following line makes a mysterious "whooshing" sound when it runs.
exports.VOID = undefined;

},{"../object-size":16,"./struct":38}],47:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
var _a;
Object.defineProperty(exports, "__esModule", { value: true });
exports.Segment = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("../constants");
const errors_1 = require("../errors");
const types_1 = require("../types");
const util_1 = require("../util");
const pointers_1 = require("./pointers");
const trace = debug_1.default("capnp:segment");
trace("load");
class Segment {
    constructor(id, message, buffer, byteLength = 0) {
        this[_a] = "Segment";
        this.id = id;
        this.message = message;
        this.buffer = buffer;
        this._dv = new DataView(buffer);
        this.byteOffset = 0;
        this.byteLength = byteLength;
    }
    /**
     * Attempt to allocate the requested number of bytes in this segment. If this segment is full this method will return
     * a pointer to freshly allocated space in another segment from the same message.
     *
     * @param {number} byteLength The number of bytes to allocate, will be rounded up to the nearest word.
     * @returns {Pointer} A pointer to the newly allocated space.
     */
    allocate(byteLength) {
        trace("allocate(%d)", byteLength);
        // eslint-disable-next-line @typescript-eslint/no-this-alias
        let segment = this;
        byteLength = util_1.padToWord(byteLength);
        if (byteLength > constants_1.MAX_SEGMENT_LENGTH - 8) {
            throw new Error(util_1.format(errors_1.SEG_SIZE_OVERFLOW, byteLength));
        }
        if (!segment.hasCapacity(byteLength)) {
            segment = segment.message.allocateSegment(byteLength);
        }
        const byteOffset = segment.byteLength;
        segment.byteLength = segment.byteLength + byteLength;
        trace("Allocated %x bytes in %s (requested segment: %s).", byteLength, this, segment);
        return new pointers_1.Pointer(segment, byteOffset);
    }
    /**
     * Quickly copy a word (8 bytes) from `srcSegment` into this one at the given offset.
     *
     * @param {number} byteOffset The offset to write the word to.
     * @param {Segment} srcSegment The segment to copy the word from.
     * @param {number} srcByteOffset The offset from the start of `srcSegment` to copy from.
     * @returns {void}
     */
    copyWord(byteOffset, srcSegment, srcByteOffset) {
        const value = srcSegment._dv.getFloat64(srcByteOffset, constants_1.NATIVE_LITTLE_ENDIAN);
        this._dv.setFloat64(byteOffset, value, constants_1.NATIVE_LITTLE_ENDIAN);
    }
    /**
     * Quickly copy words from `srcSegment` into this one.
     *
     * @param {number} byteOffset The offset to start copying into.
     * @param {Segment} srcSegment The segment to copy from.
     * @param {number} srcByteOffset The start offset to copy from.
     * @param {number} wordLength The number of words to copy.
     * @returns {void}
     */
    copyWords(byteOffset, srcSegment, srcByteOffset, wordLength) {
        const dst = new Float64Array(this.buffer, byteOffset, wordLength);
        const src = new Float64Array(srcSegment.buffer, srcByteOffset, wordLength);
        dst.set(src);
    }
    /**
     * Quickly fill a number of words in the buffer with zeroes.
     *
     * @param {number} byteOffset The first byte to set to zero.
     * @param {number} wordLength The number of words (not bytes!) to zero out.
     * @returns {void}
     */
    fillZeroWords(byteOffset, wordLength) {
        new Float64Array(this.buffer, byteOffset, wordLength).fill(0);
    }
    /** WARNING: This function is not yet implemented. */
    getBigInt64(byteOffset, littleEndian) {
        throw new Error(util_1.format(errors_1.NOT_IMPLEMENTED, byteOffset, littleEndian));
    }
    /** WARNING: This function is not yet implemented. */
    getBigUint64(byteOffset, littleEndian) {
        throw new Error(util_1.format(errors_1.NOT_IMPLEMENTED, byteOffset, littleEndian));
    }
    /**
     * Get the total number of bytes available in this segment (the size of its underlying buffer).
     *
     * @returns {number} The total number of bytes this segment can hold.
     */
    getCapacity() {
        return this.buffer.byteLength;
    }
    /**
     * Read a float32 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getFloat32(byteOffset) {
        return this._dv.getFloat32(byteOffset, true);
    }
    /**
     * Read a float64 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getFloat64(byteOffset) {
        return this._dv.getFloat64(byteOffset, true);
    }
    /**
     * Read an int16 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getInt16(byteOffset) {
        return this._dv.getInt16(byteOffset, true);
    }
    /**
     * Read an int32 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getInt32(byteOffset) {
        return this._dv.getInt32(byteOffset, true);
    }
    /**
     * Read an int64 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getInt64(byteOffset) {
        return new types_1.Int64(new Uint8Array(this.buffer.slice(byteOffset, byteOffset + 8)));
    }
    /**
     * Read an int8 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getInt8(byteOffset) {
        return this._dv.getInt8(byteOffset);
    }
    /**
     * Read a uint16 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getUint16(byteOffset) {
        return this._dv.getUint16(byteOffset, true);
    }
    /**
     * Read a uint32 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getUint32(byteOffset) {
        return this._dv.getUint32(byteOffset, true);
    }
    /**
     * Read a uint8 value out of this segment.
     * NOTE: this does not copy the memory region, so updates to the underlying buffer will affect the Uint64 value!
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getUint64(byteOffset) {
        return new types_1.Uint64(new Uint8Array(this.buffer.slice(byteOffset, byteOffset + 8)));
    }
    /**
     * Read a uint8 value out of this segment.
     *
     * @param {number} byteOffset The offset in bytes to the value.
     * @returns {number} The value.
     */
    getUint8(byteOffset) {
        return this._dv.getUint8(byteOffset);
    }
    hasCapacity(byteLength) {
        trace("hasCapacity(%d)", byteLength);
        // capacity - allocated >= requested
        return this.buffer.byteLength - this.byteLength >= byteLength;
    }
    /**
     * Quickly check the word at the given offset to see if it is equal to zero.
     *
     * PERF_V8: Fastest way to do this is by reading the whole word as a `number` (float64) in the _native_ endian format
     * and see if it's zero.
     *
     * Benchmark: http://jsben.ch/#/Pjooc
     *
     * @param {number} byteOffset The offset to the word.
     * @returns {boolean} `true` if the word is zero.
     */
    isWordZero(byteOffset) {
        return this._dv.getFloat64(byteOffset, constants_1.NATIVE_LITTLE_ENDIAN) === 0;
    }
    /**
     * Swap out this segment's underlying buffer with a new one. It's assumed that the new buffer has the same content but
     * more free space, otherwise all existing pointers to this segment will be hilariously broken.
     *
     * @param {ArrayBuffer} buffer The new buffer to use.
     * @returns {void}
     */
    replaceBuffer(buffer) {
        trace("replaceBuffer(%p)", buffer);
        if (this.buffer === buffer)
            return;
        if (buffer.byteLength < this.byteLength) {
            throw new Error(errors_1.SEG_REPLACEMENT_BUFFER_TOO_SMALL);
        }
        this._dv = new DataView(buffer);
        this.buffer = buffer;
    }
    /** WARNING: This function is not yet implemented.  */
    setBigInt64(byteOffset, value, littleEndian) {
        throw new Error(util_1.format(errors_1.NOT_IMPLEMENTED, byteOffset, value, littleEndian));
    }
    /** WARNING: This function is not yet implemented.  */
    setBigUint64(byteOffset, value, littleEndian) {
        throw new Error(util_1.format(errors_1.NOT_IMPLEMENTED, byteOffset, value, littleEndian));
    }
    /**
     * Write a float32 value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {number} val The value to store.
     * @returns {void}
     */
    setFloat32(byteOffset, val) {
        this._dv.setFloat32(byteOffset, val, true);
    }
    /**
     * Write an float64 value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {number} val The value to store.
     * @returns {void}
     */
    setFloat64(byteOffset, val) {
        this._dv.setFloat64(byteOffset, val, true);
    }
    /**
     * Write an int16 value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {number} val The value to store.
     * @returns {void}
     */
    setInt16(byteOffset, val) {
        this._dv.setInt16(byteOffset, val, true);
    }
    /**
     * Write an int32 value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {number} val The value to store.
     * @returns {void}
     */
    setInt32(byteOffset, val) {
        this._dv.setInt32(byteOffset, val, true);
    }
    /**
     * Write an int8 value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {number} val The value to store.
     * @returns {void}
     */
    setInt8(byteOffset, val) {
        this._dv.setInt8(byteOffset, val);
    }
    /**
     * Write an int64 value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {Int64} val The value to store.
     * @returns {void}
     */
    setInt64(byteOffset, val) {
        this._dv.setUint8(byteOffset, val.buffer[0]);
        this._dv.setUint8(byteOffset + 1, val.buffer[1]);
        this._dv.setUint8(byteOffset + 2, val.buffer[2]);
        this._dv.setUint8(byteOffset + 3, val.buffer[3]);
        this._dv.setUint8(byteOffset + 4, val.buffer[4]);
        this._dv.setUint8(byteOffset + 5, val.buffer[5]);
        this._dv.setUint8(byteOffset + 6, val.buffer[6]);
        this._dv.setUint8(byteOffset + 7, val.buffer[7]);
    }
    /**
     * Write a uint16 value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {number} val The value to store.
     * @returns {void}
     */
    setUint16(byteOffset, val) {
        this._dv.setUint16(byteOffset, val, true);
    }
    /**
     * Write a uint32 value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {number} val The value to store.
     * @returns {void}
     */
    setUint32(byteOffset, val) {
        this._dv.setUint32(byteOffset, val, true);
    }
    /**
     * Write a uint64 value to the specified offset.
     * TODO: benchmark other ways to perform this write operation.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {Uint64} val The value to store.
     * @returns {void}
     */
    setUint64(byteOffset, val) {
        this._dv.setUint8(byteOffset + 0, val.buffer[0]);
        this._dv.setUint8(byteOffset + 1, val.buffer[1]);
        this._dv.setUint8(byteOffset + 2, val.buffer[2]);
        this._dv.setUint8(byteOffset + 3, val.buffer[3]);
        this._dv.setUint8(byteOffset + 4, val.buffer[4]);
        this._dv.setUint8(byteOffset + 5, val.buffer[5]);
        this._dv.setUint8(byteOffset + 6, val.buffer[6]);
        this._dv.setUint8(byteOffset + 7, val.buffer[7]);
    }
    /**
     * Write a uint8 (byte) value to the specified offset.
     *
     * @param {number} byteOffset The offset from the beginning of the buffer.
     * @param {number} val The value to store.
     * @returns {void}
     */
    setUint8(byteOffset, val) {
        this._dv.setUint8(byteOffset, val);
    }
    /**
     * Write a zero word (8 bytes) to the specified offset. This is slightly faster than calling `setUint64` or
     * `setFloat64` with a zero value.
     *
     * Benchmark: http://jsben.ch/#/dUdPI
     *
     * @param {number} byteOffset The offset of the word to set to zero.
     * @returns {void}
     */
    setWordZero(byteOffset) {
        this._dv.setFloat64(byteOffset, 0, constants_1.NATIVE_LITTLE_ENDIAN);
    }
    toString() {
        return util_1.format("Segment_id:%d,off:%a,len:%a,cap:%a", this.id, this.byteLength, this.byteOffset, this.buffer.byteLength);
    }
}
exports.Segment = Segment;
_a = Symbol.toStringTag;

},{"../constants":3,"../errors":4,"../types":48,"../util":51,"./pointers":25,"debug":52,"tslib":56}],48:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Uint64 = exports.Int64 = void 0;
var int64_1 = require("./int64");
Object.defineProperty(exports, "Int64", { enumerable: true, get: function () { return int64_1.Int64; } });
var uint64_1 = require("./uint64");
Object.defineProperty(exports, "Uint64", { enumerable: true, get: function () { return uint64_1.Uint64; } });

},{"./int64":49,"./uint64":50}],49:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Int64 = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("../constants");
const util_1 = require("../util");
const uint64_1 = require("./uint64");
const trace = debug_1.default("capnp:int64");
trace("load");
/**
 * Represents a signed 64-bit integer stored using a Uint8Array in little-endian format.
 *
 * You may convert this to a primitive number by calling `toNumber()` but be wary of precision loss!
 *
 * The value passed in as the source buffer is expected to be in little-endian format.
 */
class Int64 extends uint64_1.Uint64 {
    static fromArrayBuffer(source, offset = 0, noCopy = false) {
        if (noCopy)
            return new this(new Uint8Array(source, offset, 8));
        return new this(new Uint8Array(source.slice(offset, offset + 8)));
    }
    static fromDataView(source, offset = 0, noCopy = false) {
        if (noCopy) {
            return new this(new Uint8Array(source.buffer, source.byteOffset + offset, 8));
        }
        return new this(new Uint8Array(source.buffer.slice(source.byteOffset + offset, source.byteLength + offset + 8)));
    }
    static fromNumber(source) {
        const ret = new this(new Uint8Array(8));
        ret.setValue(source);
        return ret;
    }
    /**
     * Parse a hexadecimal string in **big endian format** as an Int64 value.
     *
     * The value will be negative if the string is either preceded with a `-` sign, or already in the negative 2's
     * complement form.
     *
     * @static
     * @param {string} source The source string.
     * @returns {Int64} The string parsed as a 64-bit signed integer.
     */
    static fromHexString(source) {
        if (source.substr(0, 2) === "0x")
            source = source.substr(2);
        if (source.length < 1)
            return Int64.fromNumber(0);
        const neg = source[0] === "-";
        if (neg)
            source = source.substr(1);
        source = util_1.pad(source, 16);
        if (source.length !== 16) {
            throw new RangeError("Source string must contain at most 16 hexadecimal digits.");
        }
        const bytes = source.toLowerCase().replace(/[^\da-f]/g, "");
        const buf = new Uint8Array(new ArrayBuffer(8));
        for (let i = 0; i < 8; i++) {
            buf[7 - i] = parseInt(bytes.substr(i * 2, 2), 16);
        }
        const val = new Int64(buf);
        if (neg)
            val.negate();
        return val;
    }
    static fromUint8Array(source, offset = 0, noCopy = false) {
        if (noCopy)
            return new this(source.subarray(offset, offset + 8));
        return new this(new Uint8Array(source.buffer.slice(source.byteOffset + offset, source.byteOffset + offset + 8)));
    }
    equals(other) {
        return super.equals(other);
    }
    inspect() {
        return `[Int64 ${this.toString(10)} 0x${this.toHexString()}]`;
    }
    negate() {
        for (let b = this.buffer, carry = 1, i = 0; i < 8; i++) {
            const v = (b[i] ^ 0xff) + carry;
            b[i] = v & 0xff;
            carry = v >> 8;
        }
    }
    setValue(loWord, hiWord) {
        let negate = false;
        let lo = loWord;
        let hi = hiWord;
        if (hi === undefined) {
            hi = lo;
            negate = hi < 0;
            hi = Math.abs(hi);
            lo = hi % constants_1.VAL32;
            hi = hi / constants_1.VAL32;
            if (hi > constants_1.VAL32)
                throw new RangeError(`${loWord} is outside Int64 range`);
            hi = hi >>> 0;
        }
        for (let i = 0; i < 8; i++) {
            this.buffer[i] = lo & 0xff;
            lo = i === 3 ? hi : lo >>> 8;
        }
        if (negate)
            this.negate();
    }
    toHexString() {
        const b = this.buffer;
        const negate = b[7] & 0x80;
        if (negate)
            this.negate();
        let hex = "";
        for (let i = 7; i >= 0; i--) {
            let v = b[i].toString(16);
            if (v.length === 1)
                v = "0" + v;
            hex += v;
        }
        if (negate) {
            this.negate();
            hex = "-" + hex;
        }
        return hex;
    }
    /**
     * Convert to a native javascript number.
     *
     * WARNING: do not expect this number to be accurate to integer precision for large (positive or negative) numbers!
     *
     * @param {boolean} allowImprecise If `true`, no check is performed to verify the returned value is accurate;
     * otherwise out-of-range values are clamped to +/-Infinity.
     * @returns {number} A numeric representation of this integer.
     */
    toNumber(allowImprecise) {
        const b = this.buffer;
        const negate = b[7] & 0x80;
        let x = 0;
        let carry = 1;
        let i = 0;
        let m = 1;
        while (i < 8) {
            let v = b[i];
            if (negate) {
                v = (v ^ 0xff) + carry;
                carry = v >> 8;
                v = v & 0xff;
            }
            x += v * m;
            m *= 256;
            i++;
        }
        if (!allowImprecise && x >= constants_1.MAX_SAFE_INTEGER) {
            trace("Coercing out of range value %d to Infinity.", x);
            return negate ? -Infinity : Infinity;
        }
        return negate ? -x : x;
    }
}
exports.Int64 = Int64;

},{"../constants":3,"../util":51,"./uint64":50,"debug":52,"tslib":56}],50:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Uint64 = void 0;
const tslib_1 = require("tslib");
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("../constants");
const errors_1 = require("../errors");
const util_1 = require("../util");
const trace = debug_1.default("capnp:uint64");
trace("load");
/**
 * Represents an unsigned 64-bit integer stored using a Uint8Array in little-endian format. It's a little bit faster
 * than int64 because we don't need to keep track of the sign bit or perform two's compliment operations on set.
 *
 * You may convert this to a primitive number by calling `toNumber()` but be wary of precision loss!
 *
 * Note that overflow is not implemented, so negative numbers passed into `setValue()` will be negated first.
 *
 * The value passed in as the source buffer is expected to be in little-endian format.
 */
class Uint64 {
    /**
     * Creates a new instance; this is a no-frills constructor for speed. Use the factory methods if you need to convert
     * from other types or use a different offset into the buffer.
     *
     * Will throw if the buffer is not at least 8 bytes long.
     *
     * @constructor
     * @param {Uint8Array} buffer The buffer to use for this 64-bit word; the bytes must be in little-endian order.
     */
    constructor(buffer) {
        if (buffer.byteLength < 8)
            throw new RangeError(errors_1.RANGE_INT64_UNDERFLOW);
        this.buffer = buffer;
    }
    static fromArrayBuffer(source, offset = 0, noCopy = false) {
        if (noCopy)
            return new this(new Uint8Array(source, offset, 8));
        return new this(new Uint8Array(source.slice(offset, offset + 8)));
    }
    static fromDataView(source, offset = 0, noCopy = false) {
        if (noCopy) {
            return new this(new Uint8Array(source.buffer, source.byteOffset + offset, 8));
        }
        return new this(new Uint8Array(source.buffer.slice(source.byteOffset + offset, source.byteLength + offset + 8)));
    }
    /**
     * Parse a hexadecimal string in **big endian format** as a Uint64 value.
     *
     * @static
     * @param {string} source The source string.
     * @returns {Uint64} The string parsed as a 64-bit unsigned integer.
     */
    static fromHexString(source) {
        if (source.substr(0, 2) === "0x")
            source = source.substr(2);
        if (source.length < 1)
            return Uint64.fromNumber(0);
        if (source[0] === "-")
            throw new RangeError("Source must not be negative.");
        source = util_1.pad(source, 16);
        if (source.length !== 16) {
            throw new RangeError("Source string must contain at most 16 hexadecimal digits.");
        }
        const bytes = source.toLowerCase().replace(/[^\da-f]/g, "");
        const buf = new Uint8Array(new ArrayBuffer(8));
        for (let i = 0; i < 8; i++) {
            buf[7 - i] = parseInt(bytes.substr(i * 2, 2), 16);
        }
        return new Uint64(buf);
    }
    static fromNumber(source) {
        const ret = new this(new Uint8Array(8));
        ret.setValue(source);
        return ret;
    }
    static fromUint8Array(source, offset = 0, noCopy = false) {
        if (noCopy)
            return new this(source.subarray(offset, offset + 8));
        return new this(new Uint8Array(source.buffer.slice(source.byteOffset + offset, source.byteOffset + offset + 8)));
    }
    equals(other) {
        for (let i = 0; i < 8; i++) {
            if (this.buffer[i] !== other.buffer[i])
                return false;
        }
        return true;
    }
    inspect() {
        return `[Uint64 ${this.toString(10)} 0x${this.toHexString()}]`;
    }
    /**
     * Faster way to check for zero values without converting to a number first.
     *
     * @returns {boolean} `true` if the contained value is zero.
     * @memberOf Uint64
     */
    isZero() {
        for (let i = 0; i < 8; i++) {
            if (this.buffer[i] !== 0)
                return false;
        }
        return true;
    }
    setValue(loWord, hiWord) {
        let lo = loWord;
        let hi = hiWord;
        if (hi === undefined) {
            hi = lo;
            hi = Math.abs(hi);
            lo = hi % constants_1.VAL32;
            hi = hi / constants_1.VAL32;
            if (hi > constants_1.VAL32)
                throw new RangeError(`${loWord} is outside Uint64 range`);
            hi = hi >>> 0;
        }
        for (let i = 0; i < 8; i++) {
            this.buffer[i] = lo & 0xff;
            lo = i === 3 ? hi : lo >>> 8;
        }
    }
    /**
     * Convert to a native javascript number.
     *
     * WARNING: do not expect this number to be accurate to integer precision for large (positive or negative) numbers!
     *
     * @param {boolean} allowImprecise If `true`, no check is performed to verify the returned value is accurate;
     * otherwise out-of-range values are clamped to +Infinity.
     * @returns {number} A numeric representation of this integer.
     */
    toNumber(allowImprecise) {
        const b = this.buffer;
        let x = 0;
        let i = 0;
        let m = 1;
        while (i < 8) {
            const v = b[i];
            x += v * m;
            m *= 256;
            i++;
        }
        if (!allowImprecise && x >= constants_1.MAX_SAFE_INTEGER) {
            trace("Coercing out of range value %d to Infinity.", x);
            return Infinity;
        }
        return x;
    }
    valueOf() {
        return this.toNumber(false);
    }
    toArrayBuffer() {
        return this.buffer.buffer;
    }
    toDataView() {
        return new DataView(this.buffer.buffer);
    }
    toHexString() {
        let hex = "";
        for (let i = 7; i >= 0; i--) {
            let v = this.buffer[i].toString(16);
            if (v.length === 1)
                v = "0" + v;
            hex += v;
        }
        return hex;
    }
    toString(radix) {
        return this.toNumber(true).toString(radix);
    }
    toUint8Array() {
        return this.buffer;
    }
}
exports.Uint64 = Uint64;

},{"../constants":3,"../errors":4,"../util":51,"debug":52,"tslib":56}],51:[function(require,module,exports){
"use strict";
/**
 * @author jdiaz5513
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.repeat = exports.padToWord = exports.pad = exports.identity = exports.format = exports.encodeUtf8 = exports.dumpBuffer = exports.decodeUtf8 = exports.checkUint32 = exports.checkInt32 = exports.bufferToHex = void 0;
const tslib_1 = require("tslib");
// LINT: a lot of the util functions need the any type.
/* tslint:disable:no-any no-unsafe-any */
const debug_1 = tslib_1.__importDefault(require("debug"));
const constants_1 = require("./constants");
const errors_1 = require("./errors");
const trace = debug_1.default("capnp:util");
trace("load");
/**
 * Dump a hex string from the given buffer.
 *
 * @export
 * @param {ArrayBuffer} buffer The buffer to convert.
 * @returns {string} A hexadecimal string representing the buffer.
 */
function bufferToHex(buffer) {
    const a = new Uint8Array(buffer);
    const h = [];
    for (let i = 0; i < a.byteLength; i++)
        h.push(pad(a[i].toString(16), 2));
    return `[${h.join(" ")}]`;
}
exports.bufferToHex = bufferToHex;
/**
 * Throw an error if the provided value cannot be represented as a 32-bit integer.
 *
 * @export
 * @param {number} value The number to check.
 * @returns {number} The same number if it is valid.
 */
function checkInt32(value) {
    if (value > constants_1.MAX_INT32 || value < -constants_1.MAX_INT32) {
        throw new RangeError(errors_1.RANGE_INT32_OVERFLOW);
    }
    return value;
}
exports.checkInt32 = checkInt32;
function checkUint32(value) {
    if (value < 0 || value > constants_1.MAX_UINT32) {
        throw new RangeError(errors_1.RANGE_UINT32_OVERFLOW);
    }
    return value;
}
exports.checkUint32 = checkUint32;
/**
 * Decode a UTF-8 encoded byte array into a JavaScript string (UCS-2).
 *
 * @export
 * @param {Uint8Array} src A utf-8 encoded byte array.
 * @returns {string} A string representation of the byte array.
 */
function decodeUtf8(src) {
    // This ain't for the faint of heart, kids. If you suffer from seizures, heart palpitations, or have had a history of
    // stroke you may want to look away now.
    const l = src.byteLength;
    let dst = "";
    let i = 0;
    let cp = 0;
    let a = 0;
    let b = 0;
    let c = 0;
    let d = 0;
    while (i < l) {
        a = src[i++];
        if ((a & 0b10000000) === 0) {
            cp = a;
        }
        else if ((a & 0b11100000) === 0b11000000) {
            if (i >= l)
                throw new RangeError(errors_1.RANGE_INVALID_UTF8);
            b = src[i++];
            cp = ((a & 0b00011111) << 6) | (b & 0b00111111);
        }
        else if ((a & 0b11110000) === 0b11100000) {
            if (i + 1 >= l)
                throw new RangeError(errors_1.RANGE_INVALID_UTF8);
            b = src[i++];
            c = src[i++];
            cp = ((a & 0b00001111) << 12) | ((b & 0b00111111) << 6) | (c & 0b00111111);
        }
        else if ((a & 0b11111000) === 0b11110000) {
            if (i + 2 >= l)
                throw new RangeError(errors_1.RANGE_INVALID_UTF8);
            b = src[i++];
            c = src[i++];
            d = src[i++];
            cp = ((a & 0b00000111) << 18) | ((b & 0b00111111) << 12) | ((c & 0b00111111) << 6) | (d & 0b00111111);
        }
        else {
            throw new RangeError(errors_1.RANGE_INVALID_UTF8);
        }
        if (cp <= 0xd7ff || (cp >= 0xe000 && cp <= 0xffff)) {
            dst += String.fromCharCode(cp);
        }
        else {
            // We must reach into the astral plane and construct the surrogate pair!
            cp -= 0x00010000;
            const hi = (cp >>> 10) + 0xd800;
            const lo = (cp & 0x03ff) + 0xdc00;
            if (hi < 0xd800 || hi > 0xdbff)
                throw new RangeError(errors_1.RANGE_INVALID_UTF8);
            dst += String.fromCharCode(hi, lo);
        }
    }
    return dst;
}
exports.decodeUtf8 = decodeUtf8;
function dumpBuffer(buffer) {
    const b = buffer instanceof ArrayBuffer
        ? new Uint8Array(buffer)
        : new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    const byteLength = Math.min(b.byteLength, constants_1.MAX_BUFFER_DUMP_BYTES);
    let r = format("\n=== buffer[%d] ===", byteLength);
    for (let j = 0; j < byteLength; j += 16) {
        r += `\n${pad(j.toString(16), 8)}: `;
        let s = "";
        let k;
        for (k = 0; k < 16 && j + k < b.byteLength; k++) {
            const v = b[j + k];
            r += `${pad(v.toString(16), 2)} `;
            // Printable ASCII range.
            s += v > 31 && v < 255 ? String.fromCharCode(v) : "·";
            if (k === 7)
                r += " ";
        }
        r += `${repeat((17 - k) * 3, " ")}${s}`;
    }
    r += "\n";
    if (byteLength !== b.byteLength) {
        r += format("=== (truncated %d bytes) ===\n", b.byteLength - byteLength);
    }
    return r;
}
exports.dumpBuffer = dumpBuffer;
/**
 * Encode a JavaScript string (UCS-2) to a UTF-8 encoded string inside a Uint8Array.
 *
 * Note that the underlying buffer for the array will likely be larger than the actual contents; ignore the extra bytes.
 *
 * @export
 * @param {string} src The input string.
 * @returns {Uint8Array} A UTF-8 encoded buffer with the string's contents.
 */
function encodeUtf8(src) {
    const l = src.length;
    const dst = new Uint8Array(new ArrayBuffer(l * 4));
    let j = 0;
    for (let i = 0; i < l; i++) {
        const c = src.charCodeAt(i);
        if (c <= 0x7f) {
            dst[j++] = c;
        }
        else if (c <= 0x07ff) {
            dst[j++] = 0b11000000 | (c >>> 6);
            dst[j++] = 0b10000000 | ((c >>> 0) & 0b00111111);
        }
        else if (c <= 0xd7ff || c >= 0xe000) {
            dst[j++] = 0b11100000 | (c >>> 12);
            dst[j++] = 0b10000000 | ((c >>> 6) & 0b00111111);
            dst[j++] = 0b10000000 | ((c >>> 0) & 0b00111111);
        }
        else {
            // Make sure the surrogate pair is complete.
            /* istanbul ignore next */
            if (i + 1 >= l)
                throw new RangeError(errors_1.RANGE_INVALID_UTF8);
            // I cast thee back into the astral plane.
            const hi = c - 0xd800;
            const lo = src.charCodeAt(++i) - 0xdc00;
            const cp = ((hi << 10) | lo) + 0x00010000;
            dst[j++] = 0b11110000 | (cp >>> 18);
            dst[j++] = 0b10000000 | ((cp >>> 12) & 0b00111111);
            dst[j++] = 0b10000000 | ((cp >>> 6) & 0b00111111);
            dst[j++] = 0b10000000 | ((cp >>> 0) & 0b00111111);
        }
    }
    return dst.subarray(0, j);
}
exports.encodeUtf8 = encodeUtf8;
/**
 * Produce a `printf`-style string. Nice for providing arguments to `assert` without paying the cost for string
 * concatenation up front. Precision is supported for floating point numbers.
 *
 * @param {string} s The format string. Supported format specifiers: b, c, d, f, j, o, s, x, and X.
 * @param {...any} args Values to be formatted in the string. Arguments beyond what are consumed by the format string
 * are ignored.
 * @returns {string} The formatted string.
 */
function format(s, ...args) {
    const n = s.length;
    let arg;
    let argIndex = 0;
    let c;
    let escaped = false;
    let i = 0;
    let leadingZero = false;
    let precision;
    let result = "";
    function nextArg() {
        return args[argIndex++];
    }
    function slurpNumber() {
        let digits = "";
        while (/\d/.test(s[i])) {
            digits += s[i++];
            c = s[i];
        }
        return digits.length > 0 ? parseInt(digits, 10) : null;
    }
    for (; i < n; ++i) {
        c = s[i];
        if (escaped) {
            escaped = false;
            if (c === ".") {
                leadingZero = false;
                c = s[++i];
            }
            else if (c === "0" && s[i + 1] === ".") {
                leadingZero = true;
                i += 2;
                c = s[i];
            }
            else {
                leadingZero = true;
            }
            precision = slurpNumber();
            switch (c) {
                case "a": // number in hex with padding
                    result += "0x" + pad(parseInt(String(nextArg()), 10).toString(16), 8);
                    break;
                case "b": // number in binary
                    result += parseInt(String(nextArg()), 10).toString(2);
                    break;
                case "c": // character
                    arg = nextArg();
                    if (typeof arg === "string" || arg instanceof String) {
                        result += arg;
                    }
                    else {
                        result += String.fromCharCode(parseInt(String(arg), 10));
                    }
                    break;
                case "d": // number in decimal
                    result += parseInt(String(nextArg()), 10);
                    break;
                case "f": {
                    // floating point number
                    const tmp = parseFloat(String(nextArg())).toFixed(precision || 6);
                    result += leadingZero ? tmp : tmp.replace(/^0/, "");
                    break;
                }
                case "j": // JSON
                    result += JSON.stringify(nextArg());
                    break;
                case "o": // number in octal
                    result += "0" + parseInt(String(nextArg()), 10).toString(8);
                    break;
                case "s": // string
                    result += nextArg();
                    break;
                case "x": // lowercase hexadecimal
                    result += "0x" + parseInt(String(nextArg()), 10).toString(16);
                    break;
                case "X": // uppercase hexadecimal
                    result += "0x" + parseInt(String(nextArg()), 10).toString(16).toUpperCase();
                    break;
                default:
                    result += c;
                    break;
            }
        }
        else if (c === "%") {
            escaped = true;
        }
        else {
            result += c;
        }
    }
    return result;
}
exports.format = format;
/**
 * Return the thing that was passed in. Yaaaaawn.
 *
 * @export
 * @template T
 * @param {T} x A thing.
 * @returns {T} The same thing.
 */
function identity(x) {
    return x;
}
exports.identity = identity;
function pad(v, width, pad = "0") {
    return v.length >= width ? v : new Array(width - v.length + 1).join(pad) + v;
}
exports.pad = pad;
/**
 * Add padding to a number to make it divisible by 8. Typically used to pad byte sizes so they align to a word boundary.
 *
 * @export
 * @param {number} size The number to pad.
 * @returns {number} The padded number.
 */
function padToWord(size) {
    return (size + 7) & ~7;
}
exports.padToWord = padToWord;
/**
 * Repeat a string n times. Shamelessly copied from lodash.repeat.
 *
 * @param {number} times Number of times to repeat.
 * @param {string} str The string to repeat.
 * @returns {string} The repeated string.
 */
function repeat(times, str) {
    let out = "";
    let n = times;
    let s = str;
    if (n < 1 || n > Number.MAX_VALUE)
        return out;
    // https://en.wikipedia.org/wiki/Exponentiation_by_squaring
    do {
        if (n % 2)
            out += s;
        n = Math.floor(n / 2);
        if (n)
            s += s;
    } while (n);
    return out;
}
exports.repeat = repeat;
const hex = (v) => parseInt(String(v)).toString(16);
// Set up custom debug formatters.
/* istanbul ignore next */
debug_1.default.formatters["h"] = hex;
/* istanbul ignore next */
debug_1.default.formatters["x"] = (v) => `0x${hex(v)}`;
/* istanbul ignore next */
debug_1.default.formatters["a"] = (v) => `0x${pad(hex(v), 8)}`;
/* istanbul ignore next */
debug_1.default.formatters["X"] = (v) => `0x${hex(v).toUpperCase()}`;

},{"./constants":3,"./errors":4,"debug":52,"tslib":56}],52:[function(require,module,exports){
(function (process){(function (){
/* eslint-env browser */

/**
 * This is the web browser implementation of `debug()`.
 */

exports.formatArgs = formatArgs;
exports.save = save;
exports.load = load;
exports.useColors = useColors;
exports.storage = localstorage();
exports.destroy = (() => {
	let warned = false;

	return () => {
		if (!warned) {
			warned = true;
			console.warn('Instance method `debug.destroy()` is deprecated and no longer does anything. It will be removed in the next major version of `debug`.');
		}
	};
})();

/**
 * Colors.
 */

exports.colors = [
	'#0000CC',
	'#0000FF',
	'#0033CC',
	'#0033FF',
	'#0066CC',
	'#0066FF',
	'#0099CC',
	'#0099FF',
	'#00CC00',
	'#00CC33',
	'#00CC66',
	'#00CC99',
	'#00CCCC',
	'#00CCFF',
	'#3300CC',
	'#3300FF',
	'#3333CC',
	'#3333FF',
	'#3366CC',
	'#3366FF',
	'#3399CC',
	'#3399FF',
	'#33CC00',
	'#33CC33',
	'#33CC66',
	'#33CC99',
	'#33CCCC',
	'#33CCFF',
	'#6600CC',
	'#6600FF',
	'#6633CC',
	'#6633FF',
	'#66CC00',
	'#66CC33',
	'#9900CC',
	'#9900FF',
	'#9933CC',
	'#9933FF',
	'#99CC00',
	'#99CC33',
	'#CC0000',
	'#CC0033',
	'#CC0066',
	'#CC0099',
	'#CC00CC',
	'#CC00FF',
	'#CC3300',
	'#CC3333',
	'#CC3366',
	'#CC3399',
	'#CC33CC',
	'#CC33FF',
	'#CC6600',
	'#CC6633',
	'#CC9900',
	'#CC9933',
	'#CCCC00',
	'#CCCC33',
	'#FF0000',
	'#FF0033',
	'#FF0066',
	'#FF0099',
	'#FF00CC',
	'#FF00FF',
	'#FF3300',
	'#FF3333',
	'#FF3366',
	'#FF3399',
	'#FF33CC',
	'#FF33FF',
	'#FF6600',
	'#FF6633',
	'#FF9900',
	'#FF9933',
	'#FFCC00',
	'#FFCC33'
];

/**
 * Currently only WebKit-based Web Inspectors, Firefox >= v31,
 * and the Firebug extension (any Firefox version) are known
 * to support "%c" CSS customizations.
 *
 * TODO: add a `localStorage` variable to explicitly enable/disable colors
 */

// eslint-disable-next-line complexity
function useColors() {
	// NB: In an Electron preload script, document will be defined but not fully
	// initialized. Since we know we're in Chrome, we'll just detect this case
	// explicitly
	if (typeof window !== 'undefined' && window.process && (window.process.type === 'renderer' || window.process.__nwjs)) {
		return true;
	}

	// Internet Explorer and Edge do not support colors.
	if (typeof navigator !== 'undefined' && navigator.userAgent && navigator.userAgent.toLowerCase().match(/(edge|trident)\/(\d+)/)) {
		return false;
	}

	// Is webkit? http://stackoverflow.com/a/16459606/376773
	// document is undefined in react-native: https://github.com/facebook/react-native/pull/1632
	return (typeof document !== 'undefined' && document.documentElement && document.documentElement.style && document.documentElement.style.WebkitAppearance) ||
		// Is firebug? http://stackoverflow.com/a/398120/376773
		(typeof window !== 'undefined' && window.console && (window.console.firebug || (window.console.exception && window.console.table))) ||
		// Is firefox >= v31?
		// https://developer.mozilla.org/en-US/docs/Tools/Web_Console#Styling_messages
		(typeof navigator !== 'undefined' && navigator.userAgent && navigator.userAgent.toLowerCase().match(/firefox\/(\d+)/) && parseInt(RegExp.$1, 10) >= 31) ||
		// Double check webkit in userAgent just in case we are in a worker
		(typeof navigator !== 'undefined' && navigator.userAgent && navigator.userAgent.toLowerCase().match(/applewebkit\/(\d+)/));
}

/**
 * Colorize log arguments if enabled.
 *
 * @api public
 */

function formatArgs(args) {
	args[0] = (this.useColors ? '%c' : '') +
		this.namespace +
		(this.useColors ? ' %c' : ' ') +
		args[0] +
		(this.useColors ? '%c ' : ' ') +
		'+' + module.exports.humanize(this.diff);

	if (!this.useColors) {
		return;
	}

	const c = 'color: ' + this.color;
	args.splice(1, 0, c, 'color: inherit');

	// The final "%c" is somewhat tricky, because there could be other
	// arguments passed either before or after the %c, so we need to
	// figure out the correct index to insert the CSS into
	let index = 0;
	let lastC = 0;
	args[0].replace(/%[a-zA-Z%]/g, match => {
		if (match === '%%') {
			return;
		}
		index++;
		if (match === '%c') {
			// We only are interested in the *last* %c
			// (the user may have provided their own)
			lastC = index;
		}
	});

	args.splice(lastC, 0, c);
}

/**
 * Invokes `console.debug()` when available.
 * No-op when `console.debug` is not a "function".
 * If `console.debug` is not available, falls back
 * to `console.log`.
 *
 * @api public
 */
exports.log = console.debug || console.log || (() => {});

/**
 * Save `namespaces`.
 *
 * @param {String} namespaces
 * @api private
 */
function save(namespaces) {
	try {
		if (namespaces) {
			exports.storage.setItem('debug', namespaces);
		} else {
			exports.storage.removeItem('debug');
		}
	} catch (error) {
		// Swallow
		// XXX (@Qix-) should we be logging these?
	}
}

/**
 * Load `namespaces`.
 *
 * @return {String} returns the previously persisted debug modes
 * @api private
 */
function load() {
	let r;
	try {
		r = exports.storage.getItem('debug');
	} catch (error) {
		// Swallow
		// XXX (@Qix-) should we be logging these?
	}

	// If debug isn't set in LS, and we're in Electron, try to load $DEBUG
	if (!r && typeof process !== 'undefined' && 'env' in process) {
		r = process.env.DEBUG;
	}

	return r;
}

/**
 * Localstorage attempts to return the localstorage.
 *
 * This is necessary because safari throws
 * when a user disables cookies/localstorage
 * and you attempt to access it.
 *
 * @return {LocalStorage}
 * @api private
 */

function localstorage() {
	try {
		// TVMLKit (Apple TV JS Runtime) does not have a window object, just localStorage in the global context
		// The Browser also has localStorage in the global context.
		return localStorage;
	} catch (error) {
		// Swallow
		// XXX (@Qix-) should we be logging these?
	}
}

module.exports = require('./common')(exports);

const {formatters} = module.exports;

/**
 * Map %j to `JSON.stringify()`, since no Web Inspectors do that by default.
 */

formatters.j = function (v) {
	try {
		return JSON.stringify(v);
	} catch (error) {
		return '[UnexpectedJSONParseError]: ' + error.message;
	}
};

}).call(this)}).call(this,require('_process'))
},{"./common":53,"_process":55}],53:[function(require,module,exports){

/**
 * This is the common logic for both the Node.js and web browser
 * implementations of `debug()`.
 */

function setup(env) {
	createDebug.debug = createDebug;
	createDebug.default = createDebug;
	createDebug.coerce = coerce;
	createDebug.disable = disable;
	createDebug.enable = enable;
	createDebug.enabled = enabled;
	createDebug.humanize = require('ms');
	createDebug.destroy = destroy;

	Object.keys(env).forEach(key => {
		createDebug[key] = env[key];
	});

	/**
	* The currently active debug mode names, and names to skip.
	*/

	createDebug.names = [];
	createDebug.skips = [];

	/**
	* Map of special "%n" handling functions, for the debug "format" argument.
	*
	* Valid key names are a single, lower or upper-case letter, i.e. "n" and "N".
	*/
	createDebug.formatters = {};

	/**
	* Selects a color for a debug namespace
	* @param {String} namespace The namespace string for the debug instance to be colored
	* @return {Number|String} An ANSI color code for the given namespace
	* @api private
	*/
	function selectColor(namespace) {
		let hash = 0;

		for (let i = 0; i < namespace.length; i++) {
			hash = ((hash << 5) - hash) + namespace.charCodeAt(i);
			hash |= 0; // Convert to 32bit integer
		}

		return createDebug.colors[Math.abs(hash) % createDebug.colors.length];
	}
	createDebug.selectColor = selectColor;

	/**
	* Create a debugger with the given `namespace`.
	*
	* @param {String} namespace
	* @return {Function}
	* @api public
	*/
	function createDebug(namespace) {
		let prevTime;
		let enableOverride = null;
		let namespacesCache;
		let enabledCache;

		function debug(...args) {
			// Disabled?
			if (!debug.enabled) {
				return;
			}

			const self = debug;

			// Set `diff` timestamp
			const curr = Number(new Date());
			const ms = curr - (prevTime || curr);
			self.diff = ms;
			self.prev = prevTime;
			self.curr = curr;
			prevTime = curr;

			args[0] = createDebug.coerce(args[0]);

			if (typeof args[0] !== 'string') {
				// Anything else let's inspect with %O
				args.unshift('%O');
			}

			// Apply any `formatters` transformations
			let index = 0;
			args[0] = args[0].replace(/%([a-zA-Z%])/g, (match, format) => {
				// If we encounter an escaped % then don't increase the array index
				if (match === '%%') {
					return '%';
				}
				index++;
				const formatter = createDebug.formatters[format];
				if (typeof formatter === 'function') {
					const val = args[index];
					match = formatter.call(self, val);

					// Now we need to remove `args[index]` since it's inlined in the `format`
					args.splice(index, 1);
					index--;
				}
				return match;
			});

			// Apply env-specific formatting (colors, etc.)
			createDebug.formatArgs.call(self, args);

			const logFn = self.log || createDebug.log;
			logFn.apply(self, args);
		}

		debug.namespace = namespace;
		debug.useColors = createDebug.useColors();
		debug.color = createDebug.selectColor(namespace);
		debug.extend = extend;
		debug.destroy = createDebug.destroy; // XXX Temporary. Will be removed in the next major release.

		Object.defineProperty(debug, 'enabled', {
			enumerable: true,
			configurable: false,
			get: () => {
				if (enableOverride !== null) {
					return enableOverride;
				}
				if (namespacesCache !== createDebug.namespaces) {
					namespacesCache = createDebug.namespaces;
					enabledCache = createDebug.enabled(namespace);
				}

				return enabledCache;
			},
			set: v => {
				enableOverride = v;
			}
		});

		// Env-specific initialization logic for debug instances
		if (typeof createDebug.init === 'function') {
			createDebug.init(debug);
		}

		return debug;
	}

	function extend(namespace, delimiter) {
		const newDebug = createDebug(this.namespace + (typeof delimiter === 'undefined' ? ':' : delimiter) + namespace);
		newDebug.log = this.log;
		return newDebug;
	}

	/**
	* Enables a debug mode by namespaces. This can include modes
	* separated by a colon and wildcards.
	*
	* @param {String} namespaces
	* @api public
	*/
	function enable(namespaces) {
		createDebug.save(namespaces);
		createDebug.namespaces = namespaces;

		createDebug.names = [];
		createDebug.skips = [];

		let i;
		const split = (typeof namespaces === 'string' ? namespaces : '').split(/[\s,]+/);
		const len = split.length;

		for (i = 0; i < len; i++) {
			if (!split[i]) {
				// ignore empty strings
				continue;
			}

			namespaces = split[i].replace(/\*/g, '.*?');

			if (namespaces[0] === '-') {
				createDebug.skips.push(new RegExp('^' + namespaces.slice(1) + '$'));
			} else {
				createDebug.names.push(new RegExp('^' + namespaces + '$'));
			}
		}
	}

	/**
	* Disable debug output.
	*
	* @return {String} namespaces
	* @api public
	*/
	function disable() {
		const namespaces = [
			...createDebug.names.map(toNamespace),
			...createDebug.skips.map(toNamespace).map(namespace => '-' + namespace)
		].join(',');
		createDebug.enable('');
		return namespaces;
	}

	/**
	* Returns true if the given mode name is enabled, false otherwise.
	*
	* @param {String} name
	* @return {Boolean}
	* @api public
	*/
	function enabled(name) {
		if (name[name.length - 1] === '*') {
			return true;
		}

		let i;
		let len;

		for (i = 0, len = createDebug.skips.length; i < len; i++) {
			if (createDebug.skips[i].test(name)) {
				return false;
			}
		}

		for (i = 0, len = createDebug.names.length; i < len; i++) {
			if (createDebug.names[i].test(name)) {
				return true;
			}
		}

		return false;
	}

	/**
	* Convert regexp to namespace
	*
	* @param {RegExp} regxep
	* @return {String} namespace
	* @api private
	*/
	function toNamespace(regexp) {
		return regexp.toString()
			.substring(2, regexp.toString().length - 2)
			.replace(/\.\*\?$/, '*');
	}

	/**
	* Coerce `val`.
	*
	* @param {Mixed} val
	* @return {Mixed}
	* @api private
	*/
	function coerce(val) {
		if (val instanceof Error) {
			return val.stack || val.message;
		}
		return val;
	}

	/**
	* XXX DO NOT USE. This is a temporary stub function.
	* XXX It WILL be removed in the next major release.
	*/
	function destroy() {
		console.warn('Instance method `debug.destroy()` is deprecated and no longer does anything. It will be removed in the next major version of `debug`.');
	}

	createDebug.enable(createDebug.load());

	return createDebug;
}

module.exports = setup;

},{"ms":54}],54:[function(require,module,exports){
/**
 * Helpers.
 */

var s = 1000;
var m = s * 60;
var h = m * 60;
var d = h * 24;
var w = d * 7;
var y = d * 365.25;

/**
 * Parse or format the given `val`.
 *
 * Options:
 *
 *  - `long` verbose formatting [false]
 *
 * @param {String|Number} val
 * @param {Object} [options]
 * @throws {Error} throw an error if val is not a non-empty string or a number
 * @return {String|Number}
 * @api public
 */

module.exports = function(val, options) {
  options = options || {};
  var type = typeof val;
  if (type === 'string' && val.length > 0) {
    return parse(val);
  } else if (type === 'number' && isFinite(val)) {
    return options.long ? fmtLong(val) : fmtShort(val);
  }
  throw new Error(
    'val is not a non-empty string or a valid number. val=' +
      JSON.stringify(val)
  );
};

/**
 * Parse the given `str` and return milliseconds.
 *
 * @param {String} str
 * @return {Number}
 * @api private
 */

function parse(str) {
  str = String(str);
  if (str.length > 100) {
    return;
  }
  var match = /^(-?(?:\d+)?\.?\d+) *(milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w|years?|yrs?|y)?$/i.exec(
    str
  );
  if (!match) {
    return;
  }
  var n = parseFloat(match[1]);
  var type = (match[2] || 'ms').toLowerCase();
  switch (type) {
    case 'years':
    case 'year':
    case 'yrs':
    case 'yr':
    case 'y':
      return n * y;
    case 'weeks':
    case 'week':
    case 'w':
      return n * w;
    case 'days':
    case 'day':
    case 'd':
      return n * d;
    case 'hours':
    case 'hour':
    case 'hrs':
    case 'hr':
    case 'h':
      return n * h;
    case 'minutes':
    case 'minute':
    case 'mins':
    case 'min':
    case 'm':
      return n * m;
    case 'seconds':
    case 'second':
    case 'secs':
    case 'sec':
    case 's':
      return n * s;
    case 'milliseconds':
    case 'millisecond':
    case 'msecs':
    case 'msec':
    case 'ms':
      return n;
    default:
      return undefined;
  }
}

/**
 * Short format for `ms`.
 *
 * @param {Number} ms
 * @return {String}
 * @api private
 */

function fmtShort(ms) {
  var msAbs = Math.abs(ms);
  if (msAbs >= d) {
    return Math.round(ms / d) + 'd';
  }
  if (msAbs >= h) {
    return Math.round(ms / h) + 'h';
  }
  if (msAbs >= m) {
    return Math.round(ms / m) + 'm';
  }
  if (msAbs >= s) {
    return Math.round(ms / s) + 's';
  }
  return ms + 'ms';
}

/**
 * Long format for `ms`.
 *
 * @param {Number} ms
 * @return {String}
 * @api private
 */

function fmtLong(ms) {
  var msAbs = Math.abs(ms);
  if (msAbs >= d) {
    return plural(ms, msAbs, d, 'day');
  }
  if (msAbs >= h) {
    return plural(ms, msAbs, h, 'hour');
  }
  if (msAbs >= m) {
    return plural(ms, msAbs, m, 'minute');
  }
  if (msAbs >= s) {
    return plural(ms, msAbs, s, 'second');
  }
  return ms + ' ms';
}

/**
 * Pluralization helper.
 */

function plural(ms, msAbs, n, name) {
  var isPlural = msAbs >= n * 1.5;
  return Math.round(ms / n) + ' ' + name + (isPlural ? 's' : '');
}

},{}],55:[function(require,module,exports){
// shim for using process in browser
var process = module.exports = {};

// cached from whatever global is present so that test runners that stub it
// don't break things.  But we need to wrap it in a try catch in case it is
// wrapped in strict mode code which doesn't define any globals.  It's inside a
// function because try/catches deoptimize in certain engines.

var cachedSetTimeout;
var cachedClearTimeout;

function defaultSetTimout() {
    throw new Error('setTimeout has not been defined');
}
function defaultClearTimeout () {
    throw new Error('clearTimeout has not been defined');
}
(function () {
    try {
        if (typeof setTimeout === 'function') {
            cachedSetTimeout = setTimeout;
        } else {
            cachedSetTimeout = defaultSetTimout;
        }
    } catch (e) {
        cachedSetTimeout = defaultSetTimout;
    }
    try {
        if (typeof clearTimeout === 'function') {
            cachedClearTimeout = clearTimeout;
        } else {
            cachedClearTimeout = defaultClearTimeout;
        }
    } catch (e) {
        cachedClearTimeout = defaultClearTimeout;
    }
} ())
function runTimeout(fun) {
    if (cachedSetTimeout === setTimeout) {
        //normal enviroments in sane situations
        return setTimeout(fun, 0);
    }
    // if setTimeout wasn't available but was latter defined
    if ((cachedSetTimeout === defaultSetTimout || !cachedSetTimeout) && setTimeout) {
        cachedSetTimeout = setTimeout;
        return setTimeout(fun, 0);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedSetTimeout(fun, 0);
    } catch(e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't trust the global object when called normally
            return cachedSetTimeout.call(null, fun, 0);
        } catch(e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error
            return cachedSetTimeout.call(this, fun, 0);
        }
    }


}
function runClearTimeout(marker) {
    if (cachedClearTimeout === clearTimeout) {
        //normal enviroments in sane situations
        return clearTimeout(marker);
    }
    // if clearTimeout wasn't available but was latter defined
    if ((cachedClearTimeout === defaultClearTimeout || !cachedClearTimeout) && clearTimeout) {
        cachedClearTimeout = clearTimeout;
        return clearTimeout(marker);
    }
    try {
        // when when somebody has screwed with setTimeout but no I.E. maddness
        return cachedClearTimeout(marker);
    } catch (e){
        try {
            // When we are in I.E. but the script has been evaled so I.E. doesn't  trust the global object when called normally
            return cachedClearTimeout.call(null, marker);
        } catch (e){
            // same as above but when it's a version of I.E. that must have the global object for 'this', hopfully our context correct otherwise it will throw a global error.
            // Some versions of I.E. have different rules for clearTimeout vs setTimeout
            return cachedClearTimeout.call(this, marker);
        }
    }



}
var queue = [];
var draining = false;
var currentQueue;
var queueIndex = -1;

function cleanUpNextTick() {
    if (!draining || !currentQueue) {
        return;
    }
    draining = false;
    if (currentQueue.length) {
        queue = currentQueue.concat(queue);
    } else {
        queueIndex = -1;
    }
    if (queue.length) {
        drainQueue();
    }
}

function drainQueue() {
    if (draining) {
        return;
    }
    var timeout = runTimeout(cleanUpNextTick);
    draining = true;

    var len = queue.length;
    while(len) {
        currentQueue = queue;
        queue = [];
        while (++queueIndex < len) {
            if (currentQueue) {
                currentQueue[queueIndex].run();
            }
        }
        queueIndex = -1;
        len = queue.length;
    }
    currentQueue = null;
    draining = false;
    runClearTimeout(timeout);
}

process.nextTick = function (fun) {
    var args = new Array(arguments.length - 1);
    if (arguments.length > 1) {
        for (var i = 1; i < arguments.length; i++) {
            args[i - 1] = arguments[i];
        }
    }
    queue.push(new Item(fun, args));
    if (queue.length === 1 && !draining) {
        runTimeout(drainQueue);
    }
};

// v8 likes predictible objects
function Item(fun, array) {
    this.fun = fun;
    this.array = array;
}
Item.prototype.run = function () {
    this.fun.apply(null, this.array);
};
process.title = 'browser';
process.browser = true;
process.env = {};
process.argv = [];
process.version = ''; // empty string to avoid regexp issues
process.versions = {};

function noop() {}

process.on = noop;
process.addListener = noop;
process.once = noop;
process.off = noop;
process.removeListener = noop;
process.removeAllListeners = noop;
process.emit = noop;
process.prependListener = noop;
process.prependOnceListener = noop;

process.listeners = function (name) { return [] }

process.binding = function (name) {
    throw new Error('process.binding is not supported');
};

process.cwd = function () { return '/' };
process.chdir = function (dir) {
    throw new Error('process.chdir is not supported');
};
process.umask = function() { return 0; };

},{}],56:[function(require,module,exports){
(function (global){(function (){
/******************************************************************************
Copyright (c) Microsoft Corporation.

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
***************************************************************************** */
/* global global, define, System, Reflect, Promise */
var __extends;
var __assign;
var __rest;
var __decorate;
var __param;
var __metadata;
var __awaiter;
var __generator;
var __exportStar;
var __values;
var __read;
var __spread;
var __spreadArrays;
var __spreadArray;
var __await;
var __asyncGenerator;
var __asyncDelegator;
var __asyncValues;
var __makeTemplateObject;
var __importStar;
var __importDefault;
var __classPrivateFieldGet;
var __classPrivateFieldSet;
var __classPrivateFieldIn;
var __createBinding;
(function (factory) {
    var root = typeof global === "object" ? global : typeof self === "object" ? self : typeof this === "object" ? this : {};
    if (typeof define === "function" && define.amd) {
        define("tslib", ["exports"], function (exports) { factory(createExporter(root, createExporter(exports))); });
    }
    else if (typeof module === "object" && typeof module.exports === "object") {
        factory(createExporter(root, createExporter(module.exports)));
    }
    else {
        factory(createExporter(root));
    }
    function createExporter(exports, previous) {
        if (exports !== root) {
            if (typeof Object.create === "function") {
                Object.defineProperty(exports, "__esModule", { value: true });
            }
            else {
                exports.__esModule = true;
            }
        }
        return function (id, v) { return exports[id] = previous ? previous(id, v) : v; };
    }
})
(function (exporter) {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };

    __extends = function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };

    __assign = Object.assign || function (t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p)) t[p] = s[p];
        }
        return t;
    };

    __rest = function (s, e) {
        var t = {};
        for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p) && e.indexOf(p) < 0)
            t[p] = s[p];
        if (s != null && typeof Object.getOwnPropertySymbols === "function")
            for (var i = 0, p = Object.getOwnPropertySymbols(s); i < p.length; i++) {
                if (e.indexOf(p[i]) < 0 && Object.prototype.propertyIsEnumerable.call(s, p[i]))
                    t[p[i]] = s[p[i]];
            }
        return t;
    };

    __decorate = function (decorators, target, key, desc) {
        var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
        if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
        else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
        return c > 3 && r && Object.defineProperty(target, key, r), r;
    };

    __param = function (paramIndex, decorator) {
        return function (target, key) { decorator(target, key, paramIndex); }
    };

    __metadata = function (metadataKey, metadataValue) {
        if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(metadataKey, metadataValue);
    };

    __awaiter = function (thisArg, _arguments, P, generator) {
        function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
        return new (P || (P = Promise))(function (resolve, reject) {
            function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
            function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
            function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
            step((generator = generator.apply(thisArg, _arguments || [])).next());
        });
    };

    __generator = function (thisArg, body) {
        var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
        return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
        function verb(n) { return function (v) { return step([n, v]); }; }
        function step(op) {
            if (f) throw new TypeError("Generator is already executing.");
            while (g && (g = 0, op[0] && (_ = 0)), _) try {
                if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
                if (y = 0, t) op = [op[0] & 2, t.value];
                switch (op[0]) {
                    case 0: case 1: t = op; break;
                    case 4: _.label++; return { value: op[1], done: false };
                    case 5: _.label++; y = op[1]; op = [0]; continue;
                    case 7: op = _.ops.pop(); _.trys.pop(); continue;
                    default:
                        if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                        if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                        if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                        if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                        if (t[2]) _.ops.pop();
                        _.trys.pop(); continue;
                }
                op = body.call(thisArg, _);
            } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
            if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
        }
    };

    __exportStar = function(m, o) {
        for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(o, p)) __createBinding(o, m, p);
    };

    __createBinding = Object.create ? (function(o, m, k, k2) {
        if (k2 === undefined) k2 = k;
        var desc = Object.getOwnPropertyDescriptor(m, k);
        if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
            desc = { enumerable: true, get: function() { return m[k]; } };
        }
        Object.defineProperty(o, k2, desc);
    }) : (function(o, m, k, k2) {
        if (k2 === undefined) k2 = k;
        o[k2] = m[k];
    });

    __values = function (o) {
        var s = typeof Symbol === "function" && Symbol.iterator, m = s && o[s], i = 0;
        if (m) return m.call(o);
        if (o && typeof o.length === "number") return {
            next: function () {
                if (o && i >= o.length) o = void 0;
                return { value: o && o[i++], done: !o };
            }
        };
        throw new TypeError(s ? "Object is not iterable." : "Symbol.iterator is not defined.");
    };

    __read = function (o, n) {
        var m = typeof Symbol === "function" && o[Symbol.iterator];
        if (!m) return o;
        var i = m.call(o), r, ar = [], e;
        try {
            while ((n === void 0 || n-- > 0) && !(r = i.next()).done) ar.push(r.value);
        }
        catch (error) { e = { error: error }; }
        finally {
            try {
                if (r && !r.done && (m = i["return"])) m.call(i);
            }
            finally { if (e) throw e.error; }
        }
        return ar;
    };

    /** @deprecated */
    __spread = function () {
        for (var ar = [], i = 0; i < arguments.length; i++)
            ar = ar.concat(__read(arguments[i]));
        return ar;
    };

    /** @deprecated */
    __spreadArrays = function () {
        for (var s = 0, i = 0, il = arguments.length; i < il; i++) s += arguments[i].length;
        for (var r = Array(s), k = 0, i = 0; i < il; i++)
            for (var a = arguments[i], j = 0, jl = a.length; j < jl; j++, k++)
                r[k] = a[j];
        return r;
    };

    __spreadArray = function (to, from, pack) {
        if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
            if (ar || !(i in from)) {
                if (!ar) ar = Array.prototype.slice.call(from, 0, i);
                ar[i] = from[i];
            }
        }
        return to.concat(ar || Array.prototype.slice.call(from));
    };

    __await = function (v) {
        return this instanceof __await ? (this.v = v, this) : new __await(v);
    };

    __asyncGenerator = function (thisArg, _arguments, generator) {
        if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
        var g = generator.apply(thisArg, _arguments || []), i, q = [];
        return i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i;
        function verb(n) { if (g[n]) i[n] = function (v) { return new Promise(function (a, b) { q.push([n, v, a, b]) > 1 || resume(n, v); }); }; }
        function resume(n, v) { try { step(g[n](v)); } catch (e) { settle(q[0][3], e); } }
        function step(r) { r.value instanceof __await ? Promise.resolve(r.value.v).then(fulfill, reject) : settle(q[0][2], r);  }
        function fulfill(value) { resume("next", value); }
        function reject(value) { resume("throw", value); }
        function settle(f, v) { if (f(v), q.shift(), q.length) resume(q[0][0], q[0][1]); }
    };

    __asyncDelegator = function (o) {
        var i, p;
        return i = {}, verb("next"), verb("throw", function (e) { throw e; }), verb("return"), i[Symbol.iterator] = function () { return this; }, i;
        function verb(n, f) { i[n] = o[n] ? function (v) { return (p = !p) ? { value: __await(o[n](v)), done: n === "return" } : f ? f(v) : v; } : f; }
    };

    __asyncValues = function (o) {
        if (!Symbol.asyncIterator) throw new TypeError("Symbol.asyncIterator is not defined.");
        var m = o[Symbol.asyncIterator], i;
        return m ? m.call(o) : (o = typeof __values === "function" ? __values(o) : o[Symbol.iterator](), i = {}, verb("next"), verb("throw"), verb("return"), i[Symbol.asyncIterator] = function () { return this; }, i);
        function verb(n) { i[n] = o[n] && function (v) { return new Promise(function (resolve, reject) { v = o[n](v), settle(resolve, reject, v.done, v.value); }); }; }
        function settle(resolve, reject, d, v) { Promise.resolve(v).then(function(v) { resolve({ value: v, done: d }); }, reject); }
    };

    __makeTemplateObject = function (cooked, raw) {
        if (Object.defineProperty) { Object.defineProperty(cooked, "raw", { value: raw }); } else { cooked.raw = raw; }
        return cooked;
    };

    var __setModuleDefault = Object.create ? (function(o, v) {
        Object.defineProperty(o, "default", { enumerable: true, value: v });
    }) : function(o, v) {
        o["default"] = v;
    };

    __importStar = function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
        __setModuleDefault(result, mod);
        return result;
    };

    __importDefault = function (mod) {
        return (mod && mod.__esModule) ? mod : { "default": mod };
    };

    __classPrivateFieldGet = function (receiver, state, kind, f) {
        if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a getter");
        if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot read private member from an object whose class did not declare it");
        return kind === "m" ? f : kind === "a" ? f.call(receiver) : f ? f.value : state.get(receiver);
    };

    __classPrivateFieldSet = function (receiver, state, value, kind, f) {
        if (kind === "m") throw new TypeError("Private method is not writable");
        if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a setter");
        if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot write private member to an object whose class did not declare it");
        return (kind === "a" ? f.call(receiver, value) : f ? f.value = value : state.set(receiver, value)), value;
    };

    __classPrivateFieldIn = function (state, receiver) {
        if (receiver === null || (typeof receiver !== "object" && typeof receiver !== "function")) throw new TypeError("Cannot use 'in' operator on non-object");
        return typeof state === "function" ? receiver === state : state.has(receiver);
    };

    exporter("__extends", __extends);
    exporter("__assign", __assign);
    exporter("__rest", __rest);
    exporter("__decorate", __decorate);
    exporter("__param", __param);
    exporter("__metadata", __metadata);
    exporter("__awaiter", __awaiter);
    exporter("__generator", __generator);
    exporter("__exportStar", __exportStar);
    exporter("__createBinding", __createBinding);
    exporter("__values", __values);
    exporter("__read", __read);
    exporter("__spread", __spread);
    exporter("__spreadArrays", __spreadArrays);
    exporter("__spreadArray", __spreadArray);
    exporter("__await", __await);
    exporter("__asyncGenerator", __asyncGenerator);
    exporter("__asyncDelegator", __asyncDelegator);
    exporter("__asyncValues", __asyncValues);
    exporter("__makeTemplateObject", __makeTemplateObject);
    exporter("__importStar", __importStar);
    exporter("__importDefault", __importDefault);
    exporter("__classPrivateFieldGet", __classPrivateFieldGet);
    exporter("__classPrivateFieldSet", __classPrivateFieldSet);
    exporter("__classPrivateFieldIn", __classPrivateFieldIn);
});

}).call(this)}).call(this,typeof global !== "undefined" ? global : typeof self !== "undefined" ? self : typeof window !== "undefined" ? window : {})
},{}]},{},[2])(2)
});
