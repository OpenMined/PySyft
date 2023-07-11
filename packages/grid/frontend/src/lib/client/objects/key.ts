export class VerifyKey {
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

  constructor(verify_key: Uint8Array) {
    this.verify_key = new VerifyKey(verify_key);
    this.fqn = 'syft.node.credentials.SyftVerifyKey';
  }
}

export class SigningKey {
  seed: Uint8Array;
  fqn: string;

  constructor(seed: Uint8Array) {
    this.seed = seed;
    this.fqn = 'nacl.signing.SigningKey';
  }
}

export class SyftSigningKey {
  signing_key: SigningKey;
  fqn: string;

  constructor(signing_key: Uint8Array) {
    this.signing_key = new SigningKey(signing_key);
    this.fqn = 'syft.node.credentials.SyftSigningKey';
  }
}
