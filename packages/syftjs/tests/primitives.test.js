import { expect, it, expectTypeOf } from 'vitest';
import {serialize} from './lib/index.js'
import {deserialize} from './lib/index.js'

it('Testing string serialization', () => {
  const emptyString = '';
  const complexStringStructure = 'TeStInG123456*/{}  ';

  const serializedEmptyString = serialize(emptyString);
  const serializedcomplexStringStructure = serialize(complexStringStructure);

  
  expectTypeOf(serializedEmptyString).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedcomplexStringStructure).toMatchTypeOf(Uint8Array);

  expect(deserialize(serializedEmptyString)).toBe(emptyString);
  expect(deserialize(serializedcomplexStringStructure)).toBe(complexStringStructure);
});

it('Testing boolean serialization and deserialization', () => {
  const serializedTrue = serialize(true);
  const serializedFalse = serialize(false);

  expectTypeOf(serializedTrue).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedFalse).toMatchTypeOf(Uint8Array);

  expect(deserialize(serializedTrue)).toBe(true);
  expect(deserialize(serializedFalse)).toBe(false);
});

it('Testing bytes serialization and deserialization', () => {
  const serializedBytesSample = serialize(new Uint8Array([16,12,13,14,15]));
  const serializedEmptySample = serialize(new Uint8Array([]));

  expectTypeOf(serializedBytesSample).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedEmptySample).toMatchTypeOf(Uint8Array);

  expect(deserialize(serializedBytesSample)).toStrictEqual(new Uint8Array([16,12,13,14,15]));
  expect(deserialize(serializedEmptySample)).toStrictEqual(new Uint8Array([]));
});

it('Test Int primitive serialization/deserialization', () => {
  const positiveInt32 = 2147483647;
  const negativeInt32 = -2147483648;
  const zeroInt = 0;

  const serializedPositiveInt32 = serialize(positiveInt32);
  const serializedNegativeInt32 = serialize(negativeInt32);
  const serializedzeroInt = serialize(zeroInt);

  expectTypeOf(serializedPositiveInt32).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedNegativeInt32).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedzeroInt).toMatchTypeOf(Uint8Array);

  expect(deserialize(serializedPositiveInt32)).toBe(positiveInt32);
  expect(deserialize(serializedNegativeInt32)).toBe(negativeInt32);
  expect(deserialize(serializedzeroInt)).toBe(zeroInt);
});

it('Test Float primitive serialization/deserialization', () => {
  const positiveBigFloat = 2147483647.2532;
  const negativeBigFloat = -2147483648.1456;
  const positiveMultipleDecimalFloat = 0.1632162264589963231812316;
  const negativeMultipleDecimalFloat = -0.1632162264589963231812316;
  const zeroFloat = 0.0;

  const serializedpositiveBigFloat = serialize(positiveBigFloat);
  const serializednegativeBigFloat = serialize(negativeBigFloat);
  const serializedPositiveMultipleDecimalFloat = serialize(positiveMultipleDecimalFloat);
  const serializedNegativeMultipleDecimalFloat = serialize(negativeMultipleDecimalFloat);
  const serializedzeroFloat = serialize(zeroFloat);

  expectTypeOf(serializedpositiveBigFloat).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializednegativeBigFloat).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedzeroFloat).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedPositiveMultipleDecimalFloat).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedNegativeMultipleDecimalFloat).toMatchTypeOf(Uint8Array);

  expect(deserialize(serializedpositiveBigFloat)).toBe(positiveBigFloat);
  expect(deserialize(serializednegativeBigFloat)).toBe(negativeBigFloat);
  expect(deserialize(serializedzeroFloat)).toBe(zeroFloat);
  expect(deserialize(serializedPositiveMultipleDecimalFloat)).toBe(
    positiveMultipleDecimalFloat
  );
  expect(deserialize(serializedNegativeMultipleDecimalFloat)).toBe(
    negativeMultipleDecimalFloat
  );
});

it('Test List/Array primitive serialization/deserialization', () => {
  const emptyArray = [];
  const IntArray = [1, 2, 3, 4, 5, 6];
  const FloatArray = [1.5, 2.3, 3.4, 4.7, 5.5, 6.2];
  const StringArray = ['testing', 'test', 'tst'];
  const ArrayOfArrays = [[], ['Test'], [1, 2, 3, 4, 5, 6]];

  const serializedEmptyArray = serialize(emptyArray);
  const serializedIntArray = serialize(IntArray);
  const serializedFloatArray = serialize(FloatArray);
  const serializedStringArray = serialize(StringArray);
  const serializedArrayOfArrays = serialize(ArrayOfArrays);

  expectTypeOf(serializedEmptyArray).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedIntArray).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedFloatArray).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedStringArray).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedArrayOfArrays).toMatchTypeOf(Uint8Array);

  expect(deserialize(serializedEmptyArray)).toStrictEqual(emptyArray);
  expect(deserialize(serializedIntArray)).toStrictEqual(IntArray);
  expect(deserialize(serializedFloatArray)).toStrictEqual(FloatArray);
  expect(deserialize(serializedStringArray)).toStrictEqual(StringArray);
  expect(deserialize(serializedArrayOfArrays)).toStrictEqual(ArrayOfArrays);
});

it('Test JS Map/Python Dictionary primitive serialization/deserialization', () => {
  const emptyMap = new Map();
  const composedMap = new Map(
    Object.entries({ string: 'Test', bool: true, float: 3.5, int: 42, array: [1, 2, 3, 4] })
  );

  const serializedEmptyMap = serialize(emptyMap);
  const serializedComposedMap = serialize(composedMap);

  expectTypeOf(serializedEmptyMap).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedComposedMap).toMatchTypeOf(Uint8Array);

  expect(deserialize(serializedEmptyMap)).toStrictEqual(
    Object.fromEntries(emptyMap.entries())
  );
  expect(deserialize(serializedComposedMap)).toStrictEqual(
    Object.fromEntries(composedMap.entries())
  );
});
