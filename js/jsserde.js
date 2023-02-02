import * as capnp from "capnp-ts";
import { RecursiveSerde } from "./capnp/recursive_serde.capnp.cjs";
import { KVIterable } from "./capnp/kv_iterable.capnp.cjs";
import { Iterable } from "./capnp/iterable.capnp.cjs";

class SimpleObject{
	constructor(first, second, third, fourth, fifth) {
		this.first = first
		this.second = second
		this.third = third
		this.fourth = fourth 
		this.fifth = new Map(Object.entries(fifth))
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
					  function (number) { return capnp.Int64.fromNumber(number).buffer.reverse().buffer },
					  function (buffer) { 
						var buffer_array = new Uint8Array(buffer) // Not sure why but first byte is always zero, so we need to remove it.
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
				this.type_bank['builtins.list'] = [true, function (list) {
					const message = new capnp.Message()
					const rs = message.initRoot(Iterable)
					const listStruct = rs.initValues(list.length)
					let count = 0
					for (let index = 0; index < list.length; index++){
						let serializedObj = _this.serialize(list[index])
						let dataMsg = new capnp.Message(serializedObj, false);
						let dataCapnpObj = capnp.Struct.initData(0, serializedObj.byteLength, dataMsg.getRoot(capnp.Data))
						dataCapnpObj.copyBuffer(serializedObj)
						listStruct.set(count, dataCapnpObj)
						count += 1								
					}
					return message.toArrayBuffer()
				}, function (buffer) {
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
				this.type_bank['builtins.dict'] = [true, function (dict) {
					const message = new capnp.Message()
					const rs = message.initRoot(KVIterable)
					const keys = rs.initKeys(dict.size)
					const values = rs.initValues(dict.size)
					let count = 0
					dict.forEach((value, key) => {
						let serializedKey = _this.serialize(key)
						let serializedValue = _this.serialize(value)
						let keyMsg = new capnp.Message(serializedKey, false);
						let valueMsg = new capnp.Message(serializedKey, false);
						let keyDataCapnpObj = capnp.Struct.initData(0, serializedKey.byteLength, keyMsg.getRoot(capnp.Data))
						let valueDataCapnpObj = capnp.Struct.initData(0, serializedValue.byteLength, valueMsg.getRoot(capnp.Data))
						keyDataCapnpObj.copyBuffer(serializedKey)
						valueDataCapnpObj.copyBuffer(serializedValue)
						keys.set(count, keyDataCapnpObj)
						values.set(count, valueDataCapnpObj)
						count += 1
					});
					return message.toArrayBuffer()
				}, function (buffer) {
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
		}
		else if (Array.isArray(obj)){
			fqn = 'builtins.list'
		}
		else if (obj instanceof Map){
			fqn = 'builtins.dict'
		}
		else {
			fqn = obj.fqn
		}
		const message = new capnp.Message();
		const rs = message.initRoot(RecursiveSerde);
		let objSerdeProps = this.type_bank[fqn]
		let notRecursive = objSerdeProps[0]
		rs.setFullyQualifiedName(fqn)
		if (!notRecursive) {
			delete obj.fqn
			const txt = rs.initFieldsName(Object.keys(obj).length)
			const data = rs.initFieldsData(Object.keys(obj).length)
			let count = 0
			for (let attr in obj){
				txt.set(count, attr)
				let serializedObj = this.serialize(obj[attr])
				let dataMsg = new capnp.Message(serializedObj, false);
				let dataCapnpObj = capnp.Struct.initData(0, serializedObj.byteLength, dataMsg.getRoot(capnp.Data))
				dataCapnpObj.copyBuffer(serializedObj)
				data.set(count, dataCapnpObj)
				count += 1
			}
		} else{
			let serializedObj = this.type_bank[fqn][1](obj)
			let data = rs.initNonrecursiveBlob(serializedObj.byteLength)
			data.copyBuffer(serializedObj)

		}

		return message.toArrayBuffer()
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
}
const js = new JSSerde('http://localhost:8081/api/v1/syft/serde')
await js.loadTypeBank();
const response = await fetch('http://localhost:8081/api/v1/syft/js')
const bytes_arr = await response.arrayBuffer()
//console.log(js.serialize(1))
let result = js.deserialize(js.serialize(new SimpleObject(589462615899, true, 'hello world', [1,2,3,4,5], {'Hello': 'World', 'field1': 159})))
console.log(result)
console.log(typeof result.get('fourth')[0])
//console.log(" Object Sent From PySyft to JavaScript : ", js.deserialize(bytes_arr))
//const response2 = await fetch('http://localhost:8081/api/v1/syft/js', {method: "POST", headers: {"content-type": "application/octect-stream"}, body: js.serialize(new SimpleObject(true, 'JavaScript SimpleObject'))})