import { parse as uuidParse } from 'uuid';
import { v4 as uuidv4 } from 'uuid';
import { stringify } from 'uuid';
import { ObjectInterface } from './object';

export class UID extends ObjectInterface {
  value: Uint8Array;
  fqn: string;
  public static readonly classFqn: string = 'syft.types.uid.UID';

  constructor(value: Uint8Array | undefined = undefined) {
    super();

    if (value) {
      this.value = value;
    } else {
      const uuid = uuidv4();
      this.value = uuidParse(uuid);
    }
    this.fqn = UID.classFqn;
  }

  get hash() {
    return stringify(this.value);
  }
}
