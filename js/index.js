const Buffer = require('buffer/').Buffer
const capnp = require("capnp-ts");

const Iterable = require("./capnp/iterable.capnp.js").Iterable;
const KVIterable = require("./capnp/kv_iterable.capnp.js").KVIterable;
const RecursiveSerde = require("./capnp/recursive_serde.capnp.js").RecursiveSerde;




// lets us play in the browser console
window.capnp = capnp

window.KVIterable = KVIterable
window.Iterable = Iterable
window.RecursiveSerde = RecursiveSerde

const API_URL = "http://127.0.0.1:8081/api/v1/syft/js"



// ======= start simple <grid.api.syft.syft.SimpleObject object at 0x7f910902b3a0>
// calling rs_object2proto with <class 'grid.api.syft.syft.SimpleObject'>
// get fqn grid.api.syft.syft.SimpleObject <function rs_object2proto at 0x7f90bdb309d0>
// initializing fields with list of attribute length 1
// get key and value first 1
// calling rs_object2proto with <class 'int'>
// get fqn builtins.int <function rs_object2proto at 0x7f90bdb309d0>
// at a non recursive level
// calling nonrecursiveBlob serialize on self with  <function <lambda> at 0x7f90bdb30f70>
// serialize the sub type first b'\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x05\x00\x00\x00j\x00\x00\x00\t\x00\x00\x00\x12\x00\x00\x00builtins.int\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00'
// pack name and bytes at idx 0


// deserializing a packed recursive serde object <class 'bytes'>
// calling rs_proto2object with <class 'capnp.lib.capnp._DynamicStructReader'>
// get fqn ['grid', 'api', 'syft', 'syft', 'SimpleObject']
// get klass SimpleObject
// recovered type info from type bank (False, <function rs_object2proto at 0x7f90bdb309d0>, <function rs_proto2object at 0x7f90bdb30af0>, None, {})
// nonrecursive False
// recurisve attrs
// attr_name first
// deserializing a packed recursive serde object <class 'bytes'>
// calling rs_proto2object with <class 'capnp.lib.capnp._DynamicStructReader'>
// get fqn ['builtins', 'int']
// get klass int
// recovered type info from type bank (True, <function <lambda> at 0x7f90bdb30f70>, <function <lambda> at 0x7f90bdb31000>, None, {})
// nonrecursive True
// non recursive so deserialize the nonrecurisveBlob
// how do we deserialize a non recursive leaf blob <class 'bytes'>
// deserialize 1
// ======= finished simple


// recursive_serde_register(
//     int,
//     serialize=lambda x: x.to_bytes((x.bit_length() + 7) // 8 + 1, "big", signed=True),
//     deserialize=lambda x_bytes: int.from_bytes(x_bytes, "big", signed=True),
// )



keys = new Map();
keys.set("first", "builtins.str")
keys.set("second", "builtins.str")
keys.set("third", "builtins.str")

function createProxyConfig(map) {
    return {
        ownKeys(target) {
          return Array.from(map.keys());
        },
        getOwnPropertyDescriptor(target, prop) {
          console.log("get prop", target, prop)
          return {
            enumerable: true,
            configurable: true,
          };
        },
        get(target, name) {
            if (name in target) {
                return target[name]
            }
            console.log("accessing other property", name)
            if (map.has(name)) {
                return map.get(name)
            }
            return undefined
        },
        // getPrototypeOf(target) {
        //     console.log("afdsa", target)
        //     return this;
        // }
    }
}

// playing with this:
// https://stackoverflow.com/questions/24902061/is-there-an-repr-equivalent-for-javascript

function classFactory(className, keys) {
    const proxyConfig = createProxyConfig(keys)
    var parent = class {
        constructor() {
            return new Proxy(this, proxyConfig);
        }
    }

    const child = class extends parent {
        static name = className;
    }
    return child
}


const SyftBase = classFactory("SyftBase", keys)
window.SyftBase = SyftBase

window.syft = new SyftBase()

// const SyftBase = new Proxy({}, handler);


TYPE_BANK = new Map();


window.Buffer = Buffer

// 🟡 TODO 24: fix int decoding and add tests
function int_proto2object(buffer) {
    const view = new DataView(buffer)
    var result = null
    window.dv = view
    window.leaf = buffer

    const slice = view.byteLength % 2
    console.log("got buffer of length", view.byteLength)
    console.log("slice is", slice)

    // this doesnt work reliably the packing on the server side seems slightly off?
    if (view.byteLength < 2) {
        console.log("invalid int encoding")
    } else if (view.byteLength >= 2 && view.byteLength < 4) {
        console.log("int16")
        result = view.getInt16(slice, false)
    } else if (view.byteLength <= 5) {
        console.log("int32")
        result = view.getInt32(slice, false)
    } else {
        console.log("int64")
        result = view.getBigInt64(slice, false)
    }
    return result
}

TYPE_BANK.set("builtins.int", int_proto2object)

// we should replace these with capnp primitives
function float_proto2object(buffer) {
    const view = new DataView(buffer)
    return view.getFloat64() // this is wrong
}

TYPE_BANK.set("builtins.float", float_proto2object)

function str_proto2object(buffer) {
    return new TextDecoder().decode(buffer);
}

TYPE_BANK.set("builtins.str", str_proto2object)

function rs_proto2object(buffer) {
    console.log("rs_proto2object", buffer)
    const message = new capnp.Message(buffer, false);
    console.log("rs_proto2object2")
    window.message = message
    const rs = message.getRoot(RecursiveSerde);
    const fieldsName = rs.getFieldsName()
    const size = fieldsName.getLength()
    const fqn = rs.getFullyQualifiedName()
    console.log("fqn", fqn)
    console.log("rs_proto2object3")
    if (size < 1) {
        console.log("terminal leaf node")
        deserialize_func = TYPE_BANK.get(fqn)
        if (typeof deserialize_func === "undefined") {
            console.log("Type: ", fqn, "Not in TYPE_BANK")
        }
        const nonrecurisveBlob = rs.getNonrecursiveBlob()
        return deserialize_func(nonrecurisveBlob.toArrayBuffer())
    } else {
        console.log("rs_proto2object4")
        kv_iterable = new Map()
        const fieldsData = rs.getFieldsData()
        if (fieldsData.getLength() != fieldsName.getLength()) {
            console.log("fields sizes dont match")
        }
        for (let index = 0; index < size; index++) {
            const key = fieldsName.get(index)
            const bytes = fieldsData.get(index)
            console.log(key, bytes);
            window.rs_bytes = bytes
            window.rs_buffer = buffer
            const obj = rs_proto2object(bytes.toArrayBuffer())
            console.log("got back recursive obj", obj)
            kv_iterable.set(key, obj)
        }
        console.log("returning a kv_iterable", kv_iterable)
        return kv_iterable
    }
}

function rs_object2proto() {

}

function deserialize(buffer) {
  window.buffer = buffer
  const obj = rs_proto2object(buffer)
  console.log("trying to deserialize", obj)
  window.obj = obj
  return obj;
}

async function run() {
    // get protobuf message bytes from server
    const response = await fetch(`${API_URL}`)
    const bytes = await response.arrayBuffer()
    console.log(
        "bytes", bytes
    )

    const output = deserialize(bytes)
    console.log("output", output)

    // const client_foo = make_foo(`${server_foo.getBar()} updated`)
    // console.log(client_foo, client_foo.bar)

    // // lets us play around in the browser console
    // window.foo = client_foo

    // client_bytes = client_foo.segment.message.toArrayBuffer()
    // console.log("client_bytes", client_bytes)
    // // capnp.util.dumpBuffer(client_foo.segment)

    // const response2 = await fetch(`${API_URL}/send`, {
    //     method: "POST",
    //     headers: {"content-type": "application/octect-stream"},
    //     body: client_bytes,
    // })
    // console.log("finished posting", response2)

}
run()
