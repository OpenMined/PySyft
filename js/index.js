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
