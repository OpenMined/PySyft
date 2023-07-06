import { parse as uuidParse } from 'uuid';
import { v4 as uuidv4 } from 'uuid';

import { ObjectInterface } from './object';

export class UID implements ObjectInterface {
  value?: Uint8Array;
  fqn = 'syft.types.uid.UID';

  constructor(value: Uint8Array | undefined = undefined) {
    if (value) {
      this.value = value;
    } else {
      const uuid = uuidv4();
      this.value = uuidParse(uuid);
    }
  }
}
