import { parse as uuidParse } from 'uuid';
import { v4 as uuidv4 } from 'uuid';
import { stringify } from 'uuid';
import { ObjectInterface } from './object';

/**
 * UID class extends the ObjectInterface.
 * A class to handle universally unique identifiers (UUID) within the application.
 */
export class UID extends ObjectInterface {
  /**
   * The value of the UID object which will be a Uint8Array representation of UUID.
   */
  value: Uint8Array;

  /**
   * The fully qualified name of the UID class.
   */
  fqn: string;

  /**
   * The static, readonly property of class fully qualified name.
   * It represents the class path in PySyft library. This can be accessed without instantiating the class.
   */
  public static readonly classFqn: string = 'syft.types.uid.UID';

  /**
   * Constructs a new instance of UID.
   *
   * @param value A Uint8Array value to be assigned to the UID object. If not provided, a new UUIDv4 value will be generated.
   */
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

  /**
   * Get the stringified value of the UID object.
   *
   * @returns The stringified value of the UID.
   */
  get hash() {
    return stringify(this.value);
  }
}
