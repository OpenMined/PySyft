import * as capnp from "capnp-ts";
import { RecursiveSerde } from "./capnp/recursive_serde.capnp.cjs";
import { KVIterable } from "./capnp/kv_iterable.capnp.cjs";
import { Iterable } from "./capnp/iterable.capnp.cjs";

class JSSerde {

	constructor(url) {
		this.url = url;
	}

	async loadTypeBank() {
		var _this = this
		this.type_bank = await fetch(this.url)
				 .then(response => response.json())
				 .then(function(response) {return response})
		
				 this.type_bank['builtins.int'] = [ true,
					  function (number) { return capnp.Int64.fromNumber(number)},
					  function (buffer) { 
						var buffer_array = new Uint8Array(buffer.slice(1)) // Not sure why but first byte is always zero, so we need to remove it.
						buffer_array.reverse()  // Little endian / big endian
						if (buffer.byteLength < 8){
							var array64 = new Uint8Array(8)
							array64.set(new Uint8Array(buffer_array))
							buffer = array64.buffer
						} else {
							buffer = buffer_array.buffer
						}
						return capnp.Int64.fromArrayBuffer(buffer).toNumber(false)},
					  null,
					  {}
					]
				this.type_bank['builtins.float'] = [ true,
					function (number) { return capnp.Int64.fromNumber(number)},
					function (buffer) { 
					const hex_str = new TextDecoder().decode(buffer)
					
					var aggr = 0
					console.log(hex_str)
					var [signal,int_n,hex_dec_n,exp] = hex_str.replaceAll('.', ' ').replaceAll('0x', ' ').replaceAll('p', ' ').split(' ')
					aggr += parseInt(int_n, 16)
					
					if (signal){
						signal = -1
					} else {
						signal = 1
					}
					// bracket notation
					for (let i = 0; i < hex_dec_n.length; i++) {
						aggr += parseInt(hex_dec_n[i],16) / (16.0 ** (i + 1))
					}
					return ( aggr * (2 ** parseInt(exp,10)) * signal)
					},
					null,
					{}
					]
				this.type_bank['builtins.str'] = [ true, function (text) { return  new TextEncoder().encode(text)}, function (buffer) {return new TextDecoder().decode(buffer)}, null, {}];
				this.type_bank['builtins.bytes'] = [ true, function (bytes) { return  bytes.buffer}, function (buffer) {return new Uint8Array(buffer)}, null, {}];

				this.type_bank['builtins.bool'] = [true,
					function (boolean) { return ( (boolean) ? new Uint8Array([49]).buffer : new Uint8Array(48).buffer)},
					function (buffer) { return ( (new Uint8Array(buffer)[0] == 49) ? true : false)},
					null,
					{}
				]
				this.type_bank['builtins.list'] = [true, function (list) {}, function (buffer) {
					var iter = []
					const message = new capnp.Message(buffer, false);
					const rs = message.getRoot(Iterable);
					const values = rs.getValues();
					const size = values.getLength();
					for (let index = 0; index < size; index++) {
						const value = values.get(index)
						iter.push(_this.deserialize(value.toArrayBuffer()))
					}
					return iter;
				}, null, {}]
				this.type_bank['builtins.dict'] = [true, function (dict) {}, function (buffer) {
					var kv_iter = {}
					const message = new capnp.Message(buffer, false);
					const rs = message.getRoot(KVIterable);
					const values = rs.getValues();
					const keys = rs.getKeys();
					const size = values.getLength();
					for (let index = 0; index < size; index++) {
						const value = values.get(index)
						const key = keys.get(index);
						kv_iter[_this.deserialize(key.toArrayBuffer())] = _this.deserialize(value.toArrayBuffer())
					}
					return kv_iter;
				}, null, {}]				

	}
	
	serialize(obj) {

	}

	deserialize(buffer) {
		const message = new capnp.Message(buffer, false);
		const rs = message.getRoot(RecursiveSerde);
		const fieldsName = rs.getFieldsName()
		const size = fieldsName.getLength()
		const fqn = rs.getFullyQualifiedName()
		let obj_properties = this.type_bank[fqn]
		console.log('Deserializing : ', fqn)
		if (size < 1) {
			return this.type_bank[fqn][2](rs.getNonrecursiveBlob().toArrayBuffer())
		} else {
			let kv_iterable = new Map()
			const fieldsData = rs.getFieldsData()
			if (fieldsData.getLength() != fieldsName.getLength()) {
				console.log('Error !!')
			} else {
				for (let index = 0; index < size; index++) {
					const key = fieldsName.get(index)
					const bytes = fieldsData.get(index)
					const obj = this.deserialize(bytes.toArrayBuffer())
					kv_iterable.set(key, obj)
				}
			}
			return kv_iterable
		}
	}
}

const js = new JSSerde('http://localhost:8081/api/v1/syft/serde')
await js.loadTypeBank();
const response = await fetch('http://localhost:8081/api/v1/syft/js')
const bytes_arr = await response.arrayBuffer()
console.log(js.deserialize(bytes_arr))

//console.log(new Uint8Array([32]))
//var x = [1,2,3,4]
//var data_list = new capnp.Message();
//data_list.allocateSegment(4*8)
//var data_list_root = data_list.initRoot(capnp.DataList)
//data_list_root.segment.allocate((x.length-1)*8)

//for (let index = 0; index < x.length; index++) {
//	console.log('Entrando pela ', index, " vez")
//	var msg = new capnp.Message();
//	var msg_root = msg.initRoot(capnp.Data)
//	msg_root.copyBuffer(new Uint8Array([32]).buffer)
//	console.log(msg_root.toUint8Array())
	//console.log(x[index])
	//console.log(msg_root.toUint8Array())
	//data_list_root.set(index, msg_root)
//}

//console.log(data_list_root.toString())

//const msg = new capnp.Message();
//const req = msg.initRoot(capnp.DataList);
//req.set(0,capnp.Uint64.fromNumber(0).length)
//req.setTimestamp(capnp.Uint64.fromNumber(0));
//console.log(capnp.Uint64.fromNumber(0))