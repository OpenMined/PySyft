<script context="module">
  import { JSSerde } from '../jsserde';
  import { UUID } from '../objects/uid';
  import { APICall } from '../messages/syftMessage.ts';
  import sodium from 'libsodium-wrappers';
  import { UserCode } from '../objects/userCode';
  import { API_BASE_URL } from '$lib/constants';

  export class JSClient {
    /**
     * Constructs a new instance of the class.
     * @returns {Promise} A promise that resolves to an instance of the class.
     */
    constructor() {
      const url = API_BASE_URL;

      this.serde = new JSSerde();
      this.url = url;
      this.msg_url = `${url}/api_call`;
      this.key = window.localStorage.getItem('key');

      if (this.key) {
        this.key = Uint8Array.from(this.key.split(','));
        this.key = sodium.crypto_sign_seed_keypair(this.key);
      }

      this.userId = window.localStorage.getItem('id');
      this.serverId = window.localStorage.getItem('serverId');

      return this;
    }

    /**
     * Log in with the provided email and password.
     *
     * @param {string} email - The user's email address.
     * @param {string} password - The user's password.
     * @throws {Error} - If the login request returns an error response.
     */
    async login(email, password) {
      // Send a POST request to the login API endpoint with the email and password.
      const response = await fetch(`${this.url}/login`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      // Get the response body as an ArrayBuffer and deserialize it using the provided Serde.
      const body = await response.arrayBuffer();
      const responseData = this.serde.deserialize(body);

      try {
        // Extract the private key seed from the response data and generate a keypair using sodium.
        const {
          signing_key: { signing_key: private_key_seed }
        } = responseData;

        // Create the keypair using private key seed
        const keypair = sodium.crypto_sign_seed_keypair(private_key_seed);

        // Set the keypair as the key for this instance.
        this.key = keypair;
        // Set current userId
        this.userId = new UUID(responseData.id.value);

        // Create Session obj to be stored at sessionStorage
        const arr = Array.from // if available
          ? Array.from(private_key_seed) // use Array#from
          : [].map.call(private_key_seed, (v) => v); // otherwise map()

        const session = {
          key: arr,
          id: responseData.id.value
        };
        window.sessionStorage.setItem('session', JSON.stringify(session));
      } catch (error) {
        // If an error occurs while extracting the private key seed or generating the keypair, throw an error with the response data's error message.
        throw new Error(responseData.Error);
      }
    }

    recoverSession(session) {
      const sessionObj = JSON.parse(session);
      this.key = sodium.crypto_sign_seed_keypair(new Uint8Array(sessionObj.key));
      this.userId = new UUID(sessionObj.id);
    }

    get user() {
      return (async () => {
        return await this.send([], { uid: this.userId }, 'user.view');
      })();
    }

    /**
     * Returns a promise that resolves to an array of datasets
     * @returns {Promise<Array<Object>>} A promise that resolves to an array of datasets
     */
    get datasets() {
      return (async () => {
        return await this.send([], {}, 'dataset.get_all');
      })();
    }

    /**
     * Returns a promise that resolves to an specific Dataset Obj
     * @returns {Promise<Array<Object>>} A promise that resolves to a dataset
     */
    getDataset(datasetId) {
      return (async () => {
        return await this.send([], { uid: new UUID(datasetId) }, 'dataset.get_by_id');
      })();
    }

    /**
     * Returns a promise that resolves to an specific Dataset Obj
     * @returns {Promise<Array<Object>>} A promise that resolves to a dataset
     */
    deleteDataset(datasetId) {
      return (async () => {
        return await this.send([], { uid: new UUID(datasetId) }, 'dataset.delete_by_id');
      })();
    }

    /**
     * Returns a Promise that resolves to an array of all code requests.
     * @returns {Promise} A Promise that resolves to the result of calling the `send()` method with the parameters `[]`, `{ }`, and `'code.get_all'`.
     */
    getCodeRequests() {
      return (async () => {
        return await this.send([], {}, 'code.get_all');
      })();
    }

    /**
     * Returns a Promise that resolves to a `UserCode` object for the code request with the given `codeId`.
     * @param {string} codeId The unique identifier for the code request.
     * @returns {Promise} A Promise that resolves to a `UserCode` object constructed from the result of calling the `send()` method with the parameters `[]`, `{ uid: new UUID(codeId) }`, and `'code.get_by_id'`.
     */
    getCodeRequest(codeId) {
      return (async () => {
        return new UserCode(await this.send([], { uid: new UUID(codeId) }, 'code.get_by_id'));
      })();
    }

    /** Updates the current metadata with new fields using an API call and returns a Promise that resolves to the result of the call.
     * @param {Object} updatedMetadata - An object of metadata fields to pass to the API call.
     * @returns {Promise<object>} A Promise that resolves to an object containing the new metadata information.
     * */
    updateMetadata(newMetadata) {
      // Create a new object called 'newMetadata' with updated fields and a new property called 'fqn' with a value.
      const updateMetadata = {
        ...newMetadata,
        fqn: 'syft.service.metadata.server_metadata.ServerMetadataUpdate'
      };

      // Create a new object called 'reqFields' with one property: 'metadata',  which is set to 'updateMetadata'.
      const reqFields = { metadata: updateMetadata };

      // Return a new Promise that calls the 'send' method on 'this' with arguments to update metadata and resolves to the result of the call.
      return new Promise((resolve, reject) => {
        this.send([], reqFields, 'metadata.update')
          .then((result) => resolve(result))
          .catch((error) => reject(error));
      });
    }

    /**
     * Registers a new user with the server.
     * @param {Object} newUser - An object representing the new user to be registered.
     * @returns {Promise} A Promise that resolves to the result of the registration call.
     * @throws {Error} If the registration fails for any reason.
     */
    register(newUser) {
      return (async () => {
        // Create a register payload object by copying the newUser object and adding a fully-qualified name (fqn) property.
        const registerPayload = { ...newUser, fqn: 'syft.service.user.user.UserCreate' };

        // Make a POST request to the server with the register payload.
        const response = await fetch(`${this.url}/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: this.serde.serialize(registerPayload)
        });

        // If the response is not OK, throw an error with the HTTP status.
        if (!response.ok) {
          throw new Error(`HTTP error ${response.status}`);
        }

        // Deserialize the response and check its type.
        const responseBuffer = await response.arrayBuffer();
        const responseMsg = this.serde.deserialize(responseBuffer);

        if (Array.isArray(responseMsg)) {
          // If the response is an array, return it.
          return responseMsg;
        } else {
          // If the response is not an array, throw an error with the message.
          throw new Error(responseMsg.message);
        }
      })();
    }

    /** Updates the current user with new fields using an API call and returns a Promise that resolves to the result of the call.
     * @param {Object} updatedFields - An object of user fields to pass to the API call.
     * @returns {Promise<object>} A Promise that resolves to an object containing the new user information.
     * */
    updateCurrentUser(updatedFields) {
      // Create a new object called 'userUpdate' with updated fields and a new property called 'fqn' with a value.
      const userUpdate = { ...updatedFields, fqn: 'syft.service.user.user.UserUpdate' };

      // Create a new object called 'reqFields' with two properties: 'uid', which is set to the value of 'userId', and 'user_update', which is set to 'userUpdate'.
      const reqFields = { uid: this.userId, user_update: userUpdate };

      // Return a new Promise that calls the 'send' method on 'this' with arguments to update user and resolves to the result of the call.
      return new Promise((resolve, reject) => {
        this.send([], reqFields, 'user.update')
          .then((result) => resolve(result))
          .catch((error) => reject(error));
      });
    }

    /**
     * Returns metadata from the server.
     *
     * @returns {Promise<object>} A Promise that resolves to an object containing metadata information.
     */
    get metadata() {
      return (async () => {
        const response = await fetch(`${this.url}/metadata_capnp`);
        const metadataBuffer = await response.arrayBuffer();
        const metadata = this.serde.deserialize(metadataBuffer);

        // Store the metadata in session storage.
        window.sessionStorage.setItem('metadata', JSON.stringify(metadata));

        // Return the metadata map.
        return metadata;
      })();
    }

    /**
     * Sends an API call to the server.
     *
     * @param {Array} args - An array of arguments to pass to the API call.
     * @param {Object} kwargs - An object of keyword arguments to pass to the API call.
     * @param {string} path - The API endpoint to call.
     * @returns {Promise<object>} A Promise that resolves to an object containing the API call response.
     * @throws {Error} An error is thrown if the message signature and public key don't match.
     */
    async send(args, kwargs, path) {
      const signedCall = new APICall(this.serverId, path, args, kwargs).sign(this.key, this.serde);

      try {
        // Make a POST request to the server with the signed call.
        const response = await fetch(this.msg_url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: this.serde.serialize(signedCall)
        });

        // Deserialize the response and check its signature.
        const responseBuffer = await response.arrayBuffer();
        const signedMsg = this.serde.deserialize(responseBuffer);

        const isValid = sodium.crypto_sign_verify_detached(
          signedMsg.signature,
          signedMsg.serialized_message,
          signedMsg.credentials.verify_key
        );

        if (!isValid) {
          throw new Error("Message signature and public key don't match!");
        }

        // Return the message contained in the response.
        return this.serde.deserialize(signedMsg.serialized_message).data;
      } catch (error) {
        console.error('Error occurred in send()', error);
        throw error;
      }
    }
  }
</script>
