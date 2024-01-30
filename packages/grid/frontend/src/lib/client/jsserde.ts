import * as capnp from "capnp-ts"
import { RecursiveSerde } from "./capnp/recursive_serde.capnp.js"
import { KVIterable } from "./capnp/kv_iterable.capnp.js"
import { Iterable } from "./capnp/iterable.capnp.js"
import { DataList } from "./capnp/datalist.capnp.js"
import { DataBox } from "./capnp/databox.capnp.js"
import { stringify as uuidStringify } from "uuid"
import { parse as uuidParse } from "uuid"
import { VerifyKey } from "$lib/client/objects/key.ts"
import { classMapping } from "./jsPyClassMap.js"

export class JSSerde {
  constructor() {
    this.type_bank = {}
    this.type_bank["builtins.int"] = [
      true,
      function (number) {
        return capnp.Int64.fromNumber(number).buffer.reverse().buffer
      },
      function (buffer) {
        const buffer_array = new Uint8Array(buffer) // Not sure why but first byte is always zero, so we need to remove it.
        buffer_array.reverse() // Little endian / big endian
        if (buffer.byteLength < 8) {
          const array64 = new Uint8Array(8)
          array64.set(new Uint8Array(buffer_array))
          buffer = array64.buffer
        } else {
          buffer = buffer_array.buffer
        }
        return capnp.Int64.fromArrayBuffer(buffer).toNumber(false)
      },
      null,
      {},
    ]
    this.type_bank["builtins.float"] = [
      true,
      function (number) {
        let hex_str = ""

        if (number < 0) {
          hex_str += "-"
          number = number * -1
        }
        hex_str += "0x"
        const exponent = 10

        const n_1 = number / 2 ** exponent
        hex_str += Math.trunc(n_1).toString(16)
        hex_str += "."

        let notZeros = true
        let currNumber = n_1
        while (notZeros) {
          currNumber = (currNumber % 1) * 16
          hex_str += Math.trunc(currNumber).toString(16)
          if (currNumber % 1 === 0) {
            notZeros = false
          }
        }
        hex_str += "p" + exponent
        return new TextEncoder().encode(hex_str)
      },
      function (buffer) {
        const hex_str = new TextDecoder().decode(buffer)
        let aggr = 0
        let [signal, int_n, hex_dec_n, exp] = hex_str
          .replaceAll(".", " ")
          .replaceAll("0x", " ")
          .replaceAll("p", " ")
          .split(" ")
        aggr += parseInt(int_n, 16)

        if (signal) {
          signal = -1
        } else {
          signal = 1
        }
        // bracket notation
        for (let i = 0; i < hex_dec_n.length; i++) {
          aggr += parseInt(hex_dec_n[i], 16) / 16.0 ** (i + 1)
        }
        return aggr * 2 ** parseInt(exp, 10) * signal
      },
      null,
      {},
    ]
    this.type_bank["builtins.str"] = [
      true,
      function (text) {
        return new TextEncoder().encode(text)
      },
      function (buffer) {
        return new TextDecoder().decode(buffer)
      },
      null,
      {},
    ]
    this.type_bank["builtins.type"] = [
      true,
      function (text) {
        return new TextEncoder().encode(text)
      },
      function (buffer) {
        return new TextDecoder().decode(buffer)
      },
      null,
      {},
    ]
    this.type_bank["typing._SpecialForm"] = [
      true,
      function (text) {
        return new TextEncoder().encode(text)
      },
      function (buffer) {
        return new TextDecoder().decode(buffer)
      },
      null,
      {},
    ]
    this.type_bank["builtins.bytes"] = [
      true,
      function (bytes) {
        return bytes.buffer
      },
      function (buffer) {
        return new Uint8Array(buffer)
      },
      null,
      {},
    ]
    this.type_bank["syft.types.uid.UID"] = [
      false,
      (uuid) => {
        const message = new capnp.Message()
        const rs = message.initRoot(RecursiveSerde)
        const fields = rs.initFieldsName(1)
        const data = rs.initFieldsData(1)
        rs.setFullyQualifiedName("syft.types.uid.UID")

        fields.set(0, "value")
        const serializedObj = this.serialize(uuidParse(uuid.value))

        const dataList = this.createDataList(1)

        const dataStruct = this.createData(serializedObj.byteLength)
        dataStruct.copyBuffer(serializedObj)
        dataList.set(0, dataStruct)
        data.set(0, dataList)
        return message.toArrayBuffer()
      },
      null,
      {},
    ]
    this.type_bank["builtins.bool"] = [
      true,
      function (boolean) {
        return boolean
          ? new Uint8Array([49]).buffer
          : new Uint8Array([48]).buffer
      },
      function (buffer) {
        return new Uint8Array(buffer)[0] == 49 ? true : false
      },
      null,
      {},
    ]
    ;(this.type_bank["builtins.list"] = [
      true,
      (list) => {
        const message = new capnp.Message()
        const rs = message.initRoot(Iterable)
        const listStruct = rs.initValues(list.length)
        let count = 0
        for (let index = 0; index < list.length; index++) {
          const serializedObj = this.serialize(list[index])
          const chunks = this.splitChunks(serializedObj)
          const chunkList = this.serializeChunks(chunks)
          listStruct.set(count, chunkList)
          count += 1
        }
        return message.toArrayBuffer()
      },
      (buffer) => {
        const iter = []
        const message = new capnp.Message(buffer, false)
        const rs = message.getRoot(Iterable)
        const values = rs.getValues()
        const size = values.getLength()
        for (let index = 0; index < size; index++) {
          const value = values.get(index)
          const obj = this.processObject(value)
          iter.push(obj)
        }
        return iter
      },
      null,
      {},
    ]),
      (this.type_bank["builtins.set"] = [
        true,
        this.type_bank["builtins.list"][1],
        this.type_bank["builtins.list"][2],
        null,
        {},
      ]),
      (this.type_bank["builtins.NoneType"] = [
        true,
        // eslint-disable-next-line
        function (NoneType) {
          return new Uint8Array([49]).buffer
        },
        // eslint-disable-next-line
        function (buffer) {
          return undefined
        },
        null,
        {},
      ])
    this.type_bank["nacl.signing.SigningKey"] = [
      true,
      (key) => {
        return key
      },
      (buffer) => {
        return new Uint8Array(buffer)
      },
      null,
      {},
    ]
    this.type_bank["nacl.signing.VerifyKey"] = [
      true,
      (key) => {
        return key.key.buffer
      },
      (buffer) => {
        return new VerifyKey(new Uint8Array(buffer))
      },
      null,
      {},
    ]
    this.type_bank["builtins.tuple"] = [
      true,
      this.type_bank["builtins.list"][1],
      this.type_bank["builtins.list"][2],
      null,
      {},
    ]
    this.type_bank["pydantic.main.ModelMetaclass"] = [
      true,
      function (text) {
        return new TextEncoder().encode(text)
      },
      function (buffer) {
        return new TextDecoder().decode(buffer)
      },
      null,
      {},
    ]
    this.type_bank["builtins.dict"] = [
      true,
      (dict) => {
        const message = new capnp.Message()
        const rs = message.initRoot(KVIterable)
        const keys = rs.initKeys(dict.size)
        const values = rs.initValues(dict.size)
        let count = 0
        dict.forEach((value, key) => {
          // Saving Key
          const serializedKey = this.serialize(key)
          const keyDataStruct = this.createData(serializedKey.byteLength)
          keyDataStruct.copyBuffer(serializedKey)
          keys.set(count, keyDataStruct)

          // Saving Values
          const serializedValue = this.serialize(value)
          const chunks = this.splitChunks(serializedValue)
          const chunkList = this.serializeChunks(chunks)
          values.set(count, chunkList)

          count += 1
        })
        return message.toArrayBuffer()
      },
      (buffer) => {
        // Initialize an empty object to store the key-value pairs.
        const kv_iter = {}

        // Create a capnp.Message object from the input buffer and get the root object of type KVIterable.
        const message = new capnp.Message(buffer, false)
        const rs = message.getRoot(KVIterable)

        // Get the values, keys, and size of the KVIterable object.
        const values = rs.getValues()
        const keys = rs.getKeys()
        const size = values.getLength()

        // Iterate over the KVIterable object and process each value.
        for (let index = 0; index < size; index++) {
          const value = values.get(index)
          const key = keys.get(index)

          // Process the value using the processObject function.
          const obj = this.processObject(value)

          // Deserialize the key using the deserialize method and convert it to an ArrayBuffer.
          const deserializedKey = this.deserialize(key.toArrayBuffer())
          let keyObj = ""
          if (
            Object.prototype.toString.call(deserializedKey) ===
            "[object Object]"
          ) {
            keyObj = JSON.stringify(deserializedKey)
          } else {
            keyObj = deserializedKey
          }
          // Store the processed object as a value corresponding to the deserialized key in the kv_iter object.
          kv_iter[keyObj] = obj
        }
        return kv_iter
      },
      null,
      {},
    ]
    this.type_bank["inspect.Signature"] = [
      true,
      this.type_bank["builtins.dict"][1],
      (buffer) => {
        const message = new capnp.Message(buffer, false)
        const rs = message.getRoot(RecursiveSerde)

        // If the data is a blob, deserialize the blob and return it
        const blob = rs.getNonrecursiveBlob()
        if (blob.getLength() === 1) {
          return this.type_bank["builtins.dict"][2](blob.get(0).toArrayBuffer())
        } else {
          const totalChunk = this.processChunks(blob)
          return this.type_bank["builtins.dict"][2](totalChunk.buffer)
        }
      },
      null,
      {},
    ]
    this.type_bank["typing._GenericAlias"] = [
      true,
      this.type_bank["builtins.dict"][1],
      (buffer) => {
        const message = new capnp.Message(buffer, false)
        const rs = message.getRoot(RecursiveSerde)

        // If the data is a blob, deserialize the blob and return it
        const blob = rs.getNonrecursiveBlob()
        if (blob.getLength() === 1) {
          return this.type_bank["builtins.dict"][2](blob.get(0).toArrayBuffer())
        } else {
          const totalChunk = this.processChunks(blob)
          return this.type_bank["builtins.dict"][2](totalChunk.buffer)
        }
      },
    ]
    this.type_bank["syft.types.dicttuple.DictTuple"] = [
      true,
      this.type_bank["builtins.dict"][1],
      this.type_bank["builtins.dict"][2],
      null,
      {},
    ]
  }

  createData(length) {
    const newDataMsg = new capnp.Message()
    const dataRoot = newDataMsg.initRoot(DataBox)
    dataRoot.initValue(length)
    return dataRoot.getValue()
  }

  createDataList(length) {
    const newDataList = new capnp.Message()
    const dataListRoot = newDataList.initRoot(DataList)
    dataListRoot.initValues(length)
    return dataListRoot.getValues()
  }

  /**
   * Splits a serialized data object into chunks of a maximum size.
   *
   * @param {DataObject} serializedObj - The serialized data object to split.
   * @returns {ArrayBuffer[]} An array of binary data chunks.
   */
  splitChunks(serializedObj) {
    const sizeLimit = 5.12 ** 8
    const chunks = []
    let pointer = 0

    if (serializedObj.byteLength <= sizeLimit) {
      // If the serialized object is smaller than the size limit, add it as a single chunk
      chunks.push(serializedObj)
    } else {
      // If the serialized object is larger than the size limit, split it into multiple chunks
      const numSlices = Math.ceil(serializedObj.byteLength / sizeLimit)
      for (let i = 0; i < numSlices - 1; i++) {
        // Push a slice of the serialized object to the chunks array
        chunks.push(serializedObj.slice(pointer, pointer + sizeLimit))
        pointer += sizeLimit
      }
      // Push the last slice to the chunks array
      chunks.push(serializedObj.slice(pointer))
    }

    return chunks
  }

  /**
   * Serializes an array of binary data chunks into a data list.
   *
   * @param {ArrayBuffer[]} chunks - An array of binary data chunks.
   * @returns {DataList} A serialized data list.
   */
  serializeChunks(chunks) {
    // Create a new data list with a length equal to the number of input chunks
    const dataList = this.createDataList(chunks.length)

    // Iterate over each chunk and add it to the data list
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i]

      // Create a new data structure with the same length as the current chunk
      const dataStruct = this.createData(chunk.byteLength)

      // Copy the contents of the current chunk into the data structure
      dataStruct.copyBuffer(chunk)

      // Add the data structure to the data list
      dataList.set(i, dataStruct)
    }

    return dataList
  }

  // Determine the fully qualified name (FQN) of the object.
  getFqn(obj) {
    if (typeof obj === "boolean") {
      return "builtins.bool"
    } else if (typeof obj === "undefined") {
      return "builtins.NoneType"
    } else if (typeof obj === "string") {
      return "builtins.str"
    } else if (typeof obj === "number") {
      return Number.isInteger(obj) ? "builtins.int" : "builtins.float"
    } else if (Array.isArray(obj)) {
      return "builtins.list"
    } else if (obj && obj.byteLength !== undefined) {
      return "builtins.bytes"
    } else if (obj instanceof Map) {
      return "builtins.dict"
    } else {
      return obj.fqn
    }
  }

  /**
   * Serializes a non-recursive object and stores it in the Cap'n Proto message.
   *
   * @param {Object} obj - The object to serialize.
   * @param {RecursiveSerde} rs - The Cap'n Proto object to store the serialized data in.
   * @param {function} serializer - The function to use to serialize the object.
   */
  serializeNonRecursive(obj, rs, serializer) {
    // Serialize the object using the specified serializer function
    const serializedObj = serializer(obj)
    // Split the serialized object into chunks and initialize the data field
    const chunks = this.splitChunks(serializedObj)
    const data = rs.initNonrecursiveBlob(chunks.length)

    // Copy each chunk into a new payload and set it in the data field
    for (let i = 0; i < chunks.length; i++) {
      const payload = this.createData(chunks[i].byteLength)
      payload.copyBuffer(chunks[i])
      data.set(i, payload)
    }
  }

  /**
   * Serializes a recursive object and its properties and stores them in the Cap'n Proto message.
   *
   * @param {Object} obj - The object to serialize.
   * @param {RecursiveSerde} rs - The Cap'n Proto object to store the serialized data in.
   */
  serializeRecursive(obj, rs) {
    // Remove fqn obj property to avoid serializing it as a valid field attribute.
    const { fqn, ...newObj } = obj

    // Initialize the fields for the object's text and data
    const txt = rs.initFieldsName(Object.keys(newObj).length)
    const data = rs.initFieldsData(Object.keys(newObj).length)

    // Loop over each property of the object
    let count = 0
    for (const attr in newObj) {
      // Serialize the property's value and store it in the Cap'n Proto message
      txt.set(count, attr)
      const serializedObj = this.serialize(newObj[attr])
      const chunks = this.splitChunks(serializedObj)
      const chunkList = this.serializeChunks(chunks)
      data.set(count, chunkList)
      count += 1
    }
  }

  /**
   * Serializes the given object into a Cap'n Proto message.
   *
   * @param {Object} obj - The object to serialize.
   * @returns {ArrayBuffer} - The serialized object as an ArrayBuffer.
   */
  serialize(obj) {
    // Get the fully qualified name of the object
    const fqn = this.getFqn(obj)
    // Create a new Cap'n Proto message
    const message = new capnp.Message()

    // Initialize the root object of the message
    const rs = message.initRoot(RecursiveSerde)
    rs.setFullyQualifiedName(fqn)

    // Get the serialization properties for the object
    const objSerdeProps = this.type_bank[fqn]

    // Check if the object is not recursive
    if (objSerdeProps) {
      if (fqn === "syft.types.uid.UID") {
        return objSerdeProps[1](obj)
      }

      // If the object is not recursive, serialize it non-recursively
      this.serializeNonRecursive(obj, rs, objSerdeProps[1])
    } else {
      // Otherwise, serialize it recursively
      this.serializeRecursive(obj, rs)
    }

    // Return the serialized message as an ArrayBuffer
    return message.toArrayBuffer()
  }

  /**
   * Processes an array of binary data chunks into a single binary data array.
   *
   * @param {DataList} chunks - An array of binary data chunks.
   * @returns {Uint8Array} A single binary data array.
   */
  processChunks(chunks) {
    let totalSize = 0

    // Calculate the total size of all the chunks
    for (let i = 0; i < chunks.getLength(); i++) {
      totalSize += chunks.get(i).getLength()
    }

    // Create a new array with the total size
    const tmp = new Uint8Array(totalSize)
    let position = 0

    // Fill the new array with the data from the chunks
    for (let i = 0; i < chunks.getLength(); i++) {
      const chunkData = this.deserialize(chunks.get(i).toArrayBuffer())
      const dataChunk = new Uint8Array(chunkData)
      tmp.set(dataChunk, position)
      position += dataChunk.byteLength
    }

    return tmp
  }

  /**
   * Processes a serialized data object into its original data form.
   *
   * @param {DataObject} obj - The serialized data object to process.
   * @returns {*} The original data object.
   */
  processObject(obj) {
    let result = null

    if (obj.getLength() === 1) {
      // If the object fits into a single chunk, deserialize the chunk and return it
      result = this.deserialize(obj.get(0).toArrayBuffer())
    } else {
      // If the object is split into multiple chunks, process the chunks and deserialize the result
      const totalChunk = this.processChunks(obj)
      result = this.deserialize(totalChunk.buffer)
    }

    return result
  }

  /**
   * Deserializes a binary data buffer into its original data form.
   *
   * @param {ArrayBuffer} buffer - The binary data buffer to deserialize.
   * @returns {*} The original data object.
   */
  deserialize(buffer) {
    const message = new capnp.Message(buffer, false)
    const rs = message.getRoot(RecursiveSerde)
    const fieldsName = rs.getFieldsName()
    const size = fieldsName.getLength()
    const fqn = rs.getFullyQualifiedName()
    const objSerdeProps = this.type_bank[fqn]

    // console.log({ fieldsName, fqn, objSerdeProps })

    if (size < 1) {
      // If the data is a blob, deserialize the blob and return it
      const blob = rs.getNonrecursiveBlob()
      if (blob.getLength() === 1) {
        return objSerdeProps[2](blob.get(0).toArrayBuffer())
      } else {
        const totalChunk = this.processChunks(blob)
        return objSerdeProps[2](totalChunk.buffer)
      }
    } else {
      // If the data is a structured object, deserialize its fields into a map
      const fieldsData = rs.getFieldsData()
      const kvIterable = {}

      // Check if the number of fields in the object matches the number of field names
      if (fieldsData.getLength() !== fieldsName.getLength()) {
        console.log("Error!!")
      } else {
        // Iterate over the fields in the object and deserialize their values
        for (let i = 0; i < size; i++) {
          const key = fieldsName.get(i) // Get the name of the current field
          const bytes = fieldsData.get(i) // Get the binary data buffer for the current field
          const obj = this.processObject(bytes) // Recursively deserialize the binary data buffer
          if (fqn === "syft.types.uid.UID") {
            const hexuid = uuidStringify(obj)
            kvIterable[key] = hexuid // Add the deserialized value to the key-value iterable
          } else {
            kvIterable[key] = obj // Add the deserialized value to the key-value iterable
          }
        }
      }

      if (classMapping[fqn]) {
        const objInstance = new classMapping[fqn]()
        Object.assign(objInstance, kvIterable)
        return objInstance
      } else {
        kvIterable["fqn"] = fqn
        return kvIterable
      }
    }
  }
}

export const serde = new JSSerde()
