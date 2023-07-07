import { ObjectInterface } from './object';

/**
 * SyftVerifyKey class extends the ObjectInterface.
 * A class to represents signing VerifyKey in PySyft context.
 */
export class SyftVerifyKey extends ObjectInterface {
    /**
   * The VerifyKey object.
   */
    verify_key: VerifyKey;
    /**
    * The fully qualified name of the SyftVerifyKey class.
    */
    fqn: string;
    
    /**
    * The static, readonly property of class fully qualified name. 
    * It represents the class path in PySyft library. This can be accessed without instantiating the class.
    */
    public static readonly classFqn: string = 'syft.node.credentials.SyftVerifyKey';

    /**
    * Constructs a new instance of SyftVerifyKey.
    *
    * @param verify_key A Uint8Array value used to create a new VerifyKey object.
    */
    constructor(verify_key: Uint8Array) {
        super();
        this.verify_key = new VerifyKey(verify_key);
        this.fqn = SyftVerifyKey.classFqn;
    }
}

class VerifyKey extends ObjectInterface {
    key: Uint8Array;
    fqn: string;
    public static readonly classFqn: string = 'nacl.signing.VerifyKey';

    constructor(key: Uint8Array) {
        super();
        this.key = key;
        this.fqn = VerifyKey.classFqn;
    }
}