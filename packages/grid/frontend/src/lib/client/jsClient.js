import { JSSerde } from './jsserde.js';
import { SyftMessageWithoutReply } from './objects/syftMessage.ts';
export class JSClient {
  constructor() {
    return (async () => {
      const url = window.location.protocol + '//' + window.location.host;
      await fetch(url + '/api/v1/syft/serde')
        .then((response) => response.json())
        .then((response) => {
          this.serde = new JSSerde(response['bank']);
        });
      this.url = url;
      this.msg_url = url + '/api/v1/syft/js';
      this.node_id = await this.metadata.then((metadata) => {
        return metadata.get('id').get('value');
      });
      return this;
    })();
  }

  login(email, password) {
    return fetch(this.url + '/api/v1/login', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ email: email, password: password })
    }).then((response) => {
      if (response.status === 401) {
        throw new Error('Incorrect email or password!');
      } else {
        return response.json().then((body) => {
          this.access_token = 'Bearer ' + body['access_token'];
          this.key = body['key'];
          return body;
        });
      }
    });
  }

  get user() {
    if (!this.access_token) {
      throw new Error('User not authenticated!');
    } else {
      return fetch(this.url + '/api/v1/users/me', {
        method: 'GET',
        headers: { Authorization: this.access_token }
      })
        .then((response) => response.json())
        .then((body) => {
          return body;
        });
    }
  }

  get metadata() {
    return fetch(this.url + '/api/v1/new/metadata_capnp')
      .then((response) => response.arrayBuffer())
      .then((response) => {
        let metadata = this.serde.deserialize(response);

        let nodeAddrObj = {};
        metadata.get('id').forEach((value, key) => {
          nodeAddrObj[key] = value;
        });

        let metadataObj = {};
        metadata.forEach((value, key) => {
          metadataObj[key] = value;
        });

        metadataObj.id = nodeAddrObj;
        window.sessionStorage.setItem('metadata', JSON.stringify(metadataObj));
        return metadata;
      });
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

  send(parameters, fqn) {
    // Update User Profile
    let msg = new SyftMessageWithoutReply(this.node_id, parameters, fqn);

    return fetch(this.msg_url, {
      method: 'POST',
      headers: { 'content-type': 'application/octect-stream', Authorization: this.access_token },
      body: this.serde.serialize(msg)
    })
      .then((response) => response.arrayBuffer())
      .then((response) => {
        return this.serde.deserialize(response);
      });
  }
}
