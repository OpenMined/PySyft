import { expect, it, expectTypeOf } from 'vitest';
import { serialize } from './lib/index.js';
import { deserialize } from './lib/index.js';
import { UID } from './lib/index.js';
import { SyftVerifyKey } from './lib/index.js';

it('Testing bytes serialization and deserialization', () => {
  const randomUUID = new UID();
  const presetUUID = new UID(new Uint8Array([16, 12, 13, 14, 15]));
  const serializedrandomUID = serialize(randomUUID);
  const serializedpresetUID = serialize(presetUUID);

  expectTypeOf(serializedrandomUID).toMatchTypeOf(Uint8Array);
  expectTypeOf(serializedpresetUID).toMatchTypeOf(Uint8Array);

  expect(deserialize(serializedrandomUID)).toEqual(randomUUID);
  expect(deserialize(serializedpresetUID)).toEqual(presetUUID);
});

it(
  ('Testing Verify Key serialization and deserialization',
  () => {
    const verifyKey = new SyftVerifyKey(new Uint8Array(32));

    const serializedVerifyKey = serde.serialize(verifyKey);

    expectTypeOf(serializedVerifyKey).toMatchTypeOf(Uint8Array);

    expect(serde.deserialize(serializedVerifyKey)).toEqual(verifyKey);
  }),
);
