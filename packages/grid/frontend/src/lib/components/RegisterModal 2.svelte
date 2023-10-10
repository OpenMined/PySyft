<script>
  import { Input, Label, Modal, Checkbox, Button, Helper } from 'flowbite-svelte';
  import { SyftMessageWithoutReply } from '$lib/jsserde/objects/syftMessage.ts';
  export let formModal;
  export let client;
  export let nodeId;

  let guestCredentials;
  let name;
  let email;
  $: password = undefined;
  let passwordConfirmation;

  async function createUser() {
    if (!guestCredentials) {
      await fetch('http://localhost:8081/api/v1/guest', {
        method: 'POST',
        headers: { 'content-type': 'application/json' }
      })
        .then((response) => response.json())
        .then(function (response) {
          guestCredentials = response['access_token'];
        });
    }

    // this doesnt exist anymore
    //'syft.core.node.common.node_service.user_manager.new_user_messages.CreateUserMessage'

    let msg = new SyftMessageWithoutReply(
      nodeId,
      { email: email, password: password, name: name, role: 'Data Scientist', institution: 'DPUK' },
      'syft.service.user.user.CreateUser'
    );

    let client_bytes = client.serde.serialize(msg);

    let token = 'Bearer ' + guestCredentials;
    await fetch('http://localhost:8081/api/v1/syft/js', {
      method: 'POST',
      headers: { 'content-type': 'application/octect-stream', Authorization: token },
      body: client_bytes
    })
      .then((response) => response.arrayBuffer())
      .then((byte_msg) => client.serde.deserialize(byte_msg));
  }
</script>

<Modal bind:open={formModal} size="xs" autoclose={true} class="w-full">
  <form class="flex flex-col space-y-6" action="#">
    <h3 class="text-xl font-medium text-gray-900 dark:text-white p-0" style="text-align:center">
      Start Today!
    </h3>
    <Label class="space-y-2">
      <Label class="block mb-2">Name</Label>
      <Input bind:value={name} type="text" name="name" placeholder="Jana Doe" required />
    </Label>
    <Label class="space-y-2">
      <span>Email</span>
      <Input
        bind:value={email}
        type="email"
        name="email"
        placeholder="info@openmined.org"
        required
      />
    </Label>
    <Label class="space-y-2">
      <span>Password</span>
      <Input bind:value={password} type="password" name="password" placeholder="•••••" required />
      <Helper class="mt-2">Your password must have more than 8 characters</Helper>
    </Label>

    <Label class="space-y-2">
      <span>Password Confirmation</span>
      <Input
        bind:value={passwordConfirmation}
        type="password"
        name="password"
        placeholder="•••••"
        required
      />
    </Label>
    <div class="flex items-start">
      <Checkbox>Remember me</Checkbox>
      <a href="/" class="ml-auto text-sm text-blue-700 hover:underline dark:text-blue-500">
        Lost password?
      </a>
    </div>
    <Button type="submit" on:click={() => createUser()} class="w-full1">Register</Button>
  </form>
</Modal>
