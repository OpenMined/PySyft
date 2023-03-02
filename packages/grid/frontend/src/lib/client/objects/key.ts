class VerifyKey {
  key: Uint8Array;
  fqn: string;

  constructor(key) {
    this.key = key;
    this.fqn = 'nacl.signing.VerifyKey';
  }
}

export class SyftVerifyKey {
  verify_key: VerifyKey;
  fqn: string;

  constructor(verify_key) {
    this.verify_key = new VerifyKey(verify_key);
    this.fqn = 'syft.core.node.new.credentials.SyftVerifyKey';
  }
}
