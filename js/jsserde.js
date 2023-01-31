import * as capnp from "capnp-ts";
import { RecursiveSerde } from "./capnp/recursive_serde.capnp.cjs";
import { KVIterable } from "./capnp/kv_iterable.capnp.cjs";
import { Iterable } from "./capnp/iterable.capnp.cjs";

class SimpleObject{
	constructor(first, second, third) {
		this.first = first
		this.second = second
		this.third = third
		this.fqn = "grid.api.syft.syft.SimpleObject"
	}
}
	
class JSSerde {

	constructor(url) {
		this.url = url;
	}

	async loadTypeBank() {
		var _this = this
		this.type_bank = await fetch(this.url)
				 .then(response => response.json())
				 .then(function(response) {return response['bank']})
		
				 this.type_bank['builtins.int'] = [ true,
					  function (number) { return capnp.Int64.fromNumber(number).buffer},
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
		let fqn = ""
		if ( typeof obj === 'boolean' ) {
			fqn = 'builtins.bool'
		}
		else if ( typeof obj === 'string' ) {
			fqn = "builtins.str"
		}
		else if ( typeof obj === 'number' ) {
			fqn = "builtins.int"
		}else {
			fqn = obj.fqn
		}
		const message = new capnp.Message();
		const rs = message.initRoot(RecursiveSerde);
		let objSerdeProps = this.type_bank[fqn]
		let notRecursive = objSerdeProps[0]
		rs.setFullyQualifiedName(fqn)
		if (!notRecursive) {
			const txt = rs.initFieldsName(4)
			const data = rs.initFieldsData(4)
			let count = 0
			for (let attr in obj){
				txt.set(count, attr)
				let serializedObj = this.serialize(obj[attr])
				let dataMsg = new capnp.Message(serializedObj, false);
				let dataRoot = dataMsg.getRoot(capnp.Data);
				data.set(count, dataRoot)
				count += 1
			}
			console.log(this.deserialize(message.toArrayBuffer()))
			return message.toArrayBuffer()
		} else{
			let serializedObj = this.type_bank[fqn][1](obj)
			let data = rs.initNonrecursiveBlob(serializedObj.byteLength)
			data.copyBuffer(serializedObj)
			return message.toArrayBuffer()
		}
	}

	deserialize(buffer) {
		const message = new capnp.Message(buffer, false);
		const rs = message.getRoot(RecursiveSerde);
		const fieldsName = rs.getFieldsName()
		const size = fieldsName.getLength()
		const fqn = rs.getFullyQualifiedName()
		let obj_properties = this.type_bank[fqn]
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

	roughSizeOfObject( object ) {

		var objectList = [];
	
		var recurse = function( value )
		{
			var bytes = 0;
	
			if ( typeof value === 'boolean' ) {
				bytes = 4;
			}
			else if ( typeof value === 'string' ) {
				bytes = value.length * 2;
			}
			else if ( typeof value === 'number' ) {
				bytes = 8;
			}
			else if
			(
				typeof value === 'object'
				&& objectList.indexOf( value ) === -1
			)
			{
				objectList[ objectList.length ] = value;
				for( let i in value) {
					bytes+= 8; // an assumed existence overhead
					bytes+= recurse( value[i] )
				}
			}
	
			return bytes;
		}
	
		return recurse( object );
	}
}

const js = new JSSerde('http://localhost:8081/api/v1/syft/serde')
await js.loadTypeBank();
const response = await fetch('http://localhost:8081/api/v1/syft/js')
const bytes_arr = await response.arrayBuffer()
let simpleObj = new SimpleObject(1,true,'hello world')
console.log(js.serialize(simpleObj))

//let new_msg = new capnp.Message().initRoot(RecursiveSerde)
//let textMsg = new capnp.Message()
//let textstc = textMsg.initRoot(capnp.Text)
//let textFieldsMsg = new capnp.Message()
//let textFields = textFieldsMsg.initRoot(capnp.TextList)

//textstc.set(0, "Hello World")
//textFields.set(0,textMsg)
//textFields.set(1,textMsg)
//textFields.set(2,textMsg)
//textFields.set(3,textMsg)
//console.log("Index 1: ", textFields.get(1))
//console.log(textMsg.dump())
//console.log(textFieldsMsg.dump())
//textFields.set(1,second_str)
//textFields.set(2,third_str)
//console.log("My Field: ", textFields.get(0))

//console.log(req.getLength())
//const new_msg = new capnp.Message(msg.toArrayBuffer(), false);
//console.log(new_msg.getRoot(capnp.Text))

//console.log(req.getLength())
//console.log(msg.dump())
//console.log(req.segment.toString())
//console.log(msg.toArrayBuffer())
//console.log(req.get(0))
//
//
//console.log(js.deserialize(bytes_arr))

//console.log(req.get(0))
//req.
//let new_msg = new capnp.Message()
//let new_req = new_msg.getRoot(capnp.Text)
//console.log(new_req.get(1))
//for (let i = 0; i < x.length; i++) {
//	req.set(i, x[i])
//}
//console.log(req.segment.getUint32(req.byteOffset + 4) >>> 3)
//req.set(0, "Hello World")

//const new_msg = new capnp.Message(msg.toArrayBuffer());
//const new_req = msg.getRoot(capnp.Text);
//console.log(msg.toArrayBuffer())