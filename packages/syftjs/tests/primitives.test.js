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

  //expect(deserialize(serializedEmptyString)).toBe(emptyString);
  //expect(deserialize(serializedcomplexStringStructure)).toBe(complexStringStructure);
});

