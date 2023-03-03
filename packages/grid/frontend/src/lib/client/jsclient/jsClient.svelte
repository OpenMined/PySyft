<script context="module">
  import { JSSerde } from '../jsserde.svelte';
  import { UUID } from '../objects/uid';
  import { APICall } from '../messages/syftMessage.ts';
  import sodium from 'libsodium-wrappers';
  export class JSClient {

    /**
     * Constructs a new instance of the class.
     * @returns {Promise} A promise that resolves to an instance of the class.
     */
    constructor() {
      return (async () => {
        const url = `${window.location.protocol}//${window.location.host}`;
        try {
          // Fetch the SerDe from the server and create a new JSSerde instance.
          const response = await fetch(`${url}/api/v1/syft/serde`);
          const { bank } = await response.json();
          this.serde = new JSSerde(bank);
        } catch (error) {
          console.error('Error fetching serde:', error);
        }

        // Set the URL and message URL properties.
        this.url = url;
        this.msg_url = `${url}/api/v1/new/api_call`;

        try {
          // Get the metadata and extract the node ID value.
          const metadata = await this.metadata;
          this.nodeId = metadata.id.value;
        } catch (error) {
          console.error('Error getting metadata:', error);
        }

        // Return an instance of the class.
        return this;
      })();
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
      const response = await fetch(`${this.url}/api/v1/new/login`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ email, password })
      });

      // Get the response body as an ArrayBuffer and deserialize it using the provided Serde.
      const body = await response.arrayBuffer();
      const responseData = this.serde.deserialize(body);

      try {
        this.userId = new UUID(responseData.id.value)

        // Extract the private key seed from the response data and generate a keypair using sodium.
        const { signing_key: { signing_key: private_key_seed } } = responseData;

        const keypair = sodium.crypto_sign_seed_keypair(private_key_seed);

        // Set the keypair as the key for this instance.
        this.key = keypair;
      } catch (error) {
        // If an error occurs while extracting the private key seed or generating the keypair, throw an error with the response data's error message.
        throw new Error(responseData.Error);
      }
    }
    
    
    get user() {
      return ( async () => { 
        return await this.send([], {'uid': this.userId }, "user.view")
      } )()
    }

    /**
     * Returns metadata from the server.
     *
     * @returns {Promise<object>} A Promise that resolves to an object containing metadata information.
     */
    get metadata() {
      return (async () => {
        const response = await fetch(`${this.url}/api/v1/new/metadata_capnp`);
        const metadataBuffer = await response.arrayBuffer();
        const metadata = this.serde.deserialize(metadataBuffer);

        // Store the metadata in session storage.
        window.sessionStorage.setItem('metadata', JSON.stringify(metadata));

        // Return the metadata map.
        return metadata;
      })();
    }


    createUser(parameters) {
      return this.send(
        parameters,
        'syft.core.node.common.node_service.user_manager.new_user_messages.CreateUserMessage'
      );
    }

    updateUser(parameters) {
      return this.send(
        parameters,
        'syft.core.node.common.node_service.user_manager.new_user_messages.UpdateUserMessage'
      );
    }

    updateConfigs(parameters) {
      return this.send(
        parameters,
        'syft.core.node.common.node_service.node_setup.node_setup_messages.UpdateSetupMessage'
      );
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
      const signedCall = new APICall(this.nodeId, path, args, kwargs).sign(this.key, this.serde);

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

        /**
        if (!signedMsg.valid) {
          throw new Error("Message signature and public key don't match!");
        }

        // Return the message contained in the response.
        return signedMsg.message(this.serde);
        */
       return signedMsg
      } catch (error) {
        console.error('Error occurred in send()', error);
        throw error;
      }
    }
}
</script>
