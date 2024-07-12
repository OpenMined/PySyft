import { expect, it, expectTypeOf } from "vitest"
import { JSSerde } from "$lib/client/jsserde.ts"
import { v4 as uuidv4 } from "uuid"
import { UUID } from "$lib/client/objects/uid.ts"
import { SyftVerifyKey } from "$lib/client/objects/key.ts"
import { APICall } from "$lib/client/messages/syftMessage.ts"

const serde = new JSSerde()

it("Test Boolean primitive serialization/deserialization", () => {
  const serializedTrue = serde.serialize(true)
  const serializedFalse = serde.serialize(false)

  expectTypeOf(serializedTrue).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedFalse).toMatchTypeOf(Uint8Array)

  expect(serde.deserialize(serializedTrue)).toBe(true)
  expect(serde.deserialize(serializedFalse)).toBe(false)
})

it("Test Int primitive serialization/deserialization", () => {
  const positiveInt32 = 2147483647
  const negativeInt32 = -2147483648
  const zeroInt = 0

  const serializedPositiveInt32 = serde.serialize(positiveInt32)
  const serializedNegativeInt32 = serde.serialize(negativeInt32)
  const serializedzeroInt = serde.serialize(zeroInt)

  expectTypeOf(serializedPositiveInt32).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedNegativeInt32).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedzeroInt).toMatchTypeOf(Uint8Array)

  expect(serde.deserialize(serializedPositiveInt32)).toBe(positiveInt32)
  expect(serde.deserialize(serializedNegativeInt32)).toBe(negativeInt32)
  expect(serde.deserialize(serializedzeroInt)).toBe(zeroInt)
})

it("Test Float primitive serialization/deserialization", () => {
  const positiveBigFloat = 2147483647.2532
  const negativeBigFloat = -2147483648.1456
  const positiveMultipleDecimalFloat = 0.1632162264589963231812316
  const negativeMultipleDecimalFloat = -0.1632162264589963231812316
  const zeroFloat = 0.0

  const serializedpositiveBigFloat = serde.serialize(positiveBigFloat)
  const serializednegativeBigFloat = serde.serialize(negativeBigFloat)
  const serializedPositiveMultipleDecimalFloat = serde.serialize(
    positiveMultipleDecimalFloat
  )
  const serializedNegativeMultipleDecimalFloat = serde.serialize(
    negativeMultipleDecimalFloat
  )
  const serializedzeroFloat = serde.serialize(zeroFloat)

  expectTypeOf(serializedpositiveBigFloat).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializednegativeBigFloat).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedzeroFloat).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedPositiveMultipleDecimalFloat).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedNegativeMultipleDecimalFloat).toMatchTypeOf(Uint8Array)

  expect(serde.deserialize(serializedpositiveBigFloat)).toBe(positiveBigFloat)
  expect(serde.deserialize(serializednegativeBigFloat)).toBe(negativeBigFloat)
  expect(serde.deserialize(serializedzeroFloat)).toBe(zeroFloat)
  expect(serde.deserialize(serializedPositiveMultipleDecimalFloat)).toBe(
    positiveMultipleDecimalFloat
  )
  expect(serde.deserialize(serializedNegativeMultipleDecimalFloat)).toBe(
    negativeMultipleDecimalFloat
  )
})

it("Test String primitive serialization/deserialization", () => {
  const emptyString = ""
  const complexStringStructure = "TeStInG123456*/{}  "

  const serializedEmptyString = serde.serialize(emptyString)
  const serializedcomplexStringStructure = serde.serialize(
    complexStringStructure
  )

  expectTypeOf(serializedEmptyString).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedcomplexStringStructure).toMatchTypeOf(Uint8Array)

  expect(serde.deserialize(serializedEmptyString)).toBe(emptyString)
  expect(serde.deserialize(serializedcomplexStringStructure)).toBe(
    complexStringStructure
  )
})

it("Test List/Array primitive serialization/deserialization", () => {
  const emptyArray = []
  const IntArray = [1, 2, 3, 4, 5, 6]
  const FloatArray = [1.5, 2.3, 3.4, 4.7, 5.5, 6.2]
  const StringArray = ["testing", "test", "tst"]
  const ArrayOfArrays = [[], ["Test"], [1, 2, 3, 4, 5, 6]]

  const serializedEmptyArray = serde.serialize(emptyArray)
  const serializedIntArray = serde.serialize(IntArray)
  const serializedFloatArray = serde.serialize(FloatArray)
  const serializedStringArray = serde.serialize(StringArray)
  const serializedArrayOfArrays = serde.serialize(ArrayOfArrays)

  expectTypeOf(serializedEmptyArray).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedIntArray).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedFloatArray).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedStringArray).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedArrayOfArrays).toMatchTypeOf(Uint8Array)

  expect(serde.deserialize(serializedEmptyArray)).toStrictEqual(emptyArray)
  expect(serde.deserialize(serializedIntArray)).toStrictEqual(IntArray)
  expect(serde.deserialize(serializedFloatArray)).toStrictEqual(FloatArray)
  expect(serde.deserialize(serializedStringArray)).toStrictEqual(StringArray)
  expect(serde.deserialize(serializedArrayOfArrays)).toStrictEqual(
    ArrayOfArrays
  )
})

it("Test JS Map/Python Dictionary primitive serialization/deserialization", () => {
  const emptyMap = new Map()
  const composedMap = new Map(
    Object.entries({
      string: "Test",
      bool: true,
      float: 3.5,
      int: 42,
      array: [1, 2, 3, 4],
    })
  )

  const serializedEmptyMap = serde.serialize(emptyMap)
  const serializedComposedMap = serde.serialize(composedMap)

  expectTypeOf(serializedEmptyMap).toMatchTypeOf(Uint8Array)
  expectTypeOf(serializedComposedMap).toMatchTypeOf(Uint8Array)

  expect(serde.deserialize(serializedEmptyMap)).toStrictEqual(
    Object.fromEntries(emptyMap.entries())
  )
  expect(serde.deserialize(serializedComposedMap)).toStrictEqual(
    Object.fromEntries(composedMap.entries())
  )
})

it("Test UUID Object serialization/deserialization", () => {
  const uuid = new UUID(uuidv4())

  const serializedUuid = serde.serialize(uuid)

  expectTypeOf(serializedUuid).toMatchTypeOf(Uint8Array)

  expect(serde.deserialize(serializedUuid)).toStrictEqual(uuid)
})

it("Test  SyftVerifyKey Object serialization/deserialization", () => {
  const uuid = new UUID(uuidv4())
  const verifyKey = new SyftVerifyKey(new Uint8Array(32))

  const serializedVerifyKey = serde.serialize(verifyKey)

  expectTypeOf(serializedVerifyKey).toMatchTypeOf(Uint8Array)

  expect(serde.deserialize(serializedVerifyKey)).toStrictEqual(verifyKey)
})

it("Test Get Specific User ID APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  const apiCall = new APICall(uuid, "user.view", [], {
    uid: new UUID(uuidv4()),
  })

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})

it("Test  Update Metadata APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  let newMetadata = {
    name: "New Server",
    organization: "New Organization",
    description: "New Description",
    fqn: "syft.service.metadata.server_metadata.ServerMetadataUpdate",
  }

  const apiCall = new APICall(uuid, "metadata.update", [], newMetadata)

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})

it("Test  Update User APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  const userUpdate = {
    userId: new UUID(uuidv4()),
    user_update: {
      email: "test@email.com",
      password: "pwd123",
      name: "test",
      institution: "Test Test",
      website: "http://test.com",
      fqn: "syft.service.user.user.UserUpdate",
    },
  }

  const apiCall = new APICall(uuid, "user.update", [], userUpdate)

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})

it("Test  UserCreate APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  const userCreate = {
    email: "test@email.com",
    password: "pwd123",
    institution: "Test Test",
    website: "http://test.com",
    password_verify: "pwd123",
    fqn: "syft.service.user.user.UserCreate",
  }

  const apiCall = new APICall(uuid, "user.create", [], userCreate)

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})

it("Test Dataset Get All APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  const apiCall = new APICall(uuid, "dataset.get_all", [], {})

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})

it("Test Dataset Get Specific Dataset APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  const apiCall = new APICall(uuid, "dataset.get_by_id", [], {
    uid: new UUID(uuidv4()),
  })

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})

it("Test Delete Specific Dataset APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  const apiCall = new APICall(uuid, "dataset.delete_by_id", [], {
    uid: new UUID(uuidv4()),
  })

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})

it("Test Get All Code APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  const apiCall = new APICall(uuid, "code.get_all", [], {})

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})

it("Test Get Code Specific ID APICall serialization/deserialization", () => {
  const uuid = uuidv4()

  const apiCall = new APICall(uuid, "code.get_by_id", [], {
    uid: new UUID(uuidv4()),
  })

  const serializedApiCall = serde.serialize(apiCall)

  expectTypeOf(serializedApiCall).toMatchTypeOf(Uint8Array)

  let deserializedapiCall = serde.deserialize(serializedApiCall)
  expect(deserializedapiCall.kwargs).toStrictEqual(
    Object.fromEntries(apiCall.kwargs)
  )
})
