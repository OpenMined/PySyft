<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import Modal from '$lib/components/NewModal.svelte';
  import DomainMetadataPanel from '$lib/components/authentication/DomainMetadataPanel.svelte';
  import Input from '$lib/components/Input.svelte';
  import DomainOnlineIndicator from '$lib/components/DomainOnlineIndicator.svelte';
  import type { DomainMetadata } from '../../../types/domain/metadata';

  //   async function login(client) {
  //     let password = document.getElementById('password').value;
  //     let email = document.getElementById('email').value;

  //     await client
  //       .login(email, password)
  //       .then((body) => {
  //         if (browser) {
  //           window.sessionStorage.setItem('session_token', 'Bearer ' + body['access_token']);
  //           window.sessionStorage.setItem('key', body['key']);
  //         }
  //         goto('/home');
  //       })
  //       .catch((error) => {
  //         errorText = error.message;
  //         inputColor = 'red';
  //         displayError = 'block';
  //       });
  //   }

  //   const copyToClipBoard = () => {
  //     // Get the text field
  //     var copyText = document.getElementById('gridUID');

  //     // Copy the text inside the text field
  //     navigator.clipboard.writeText(copyText?.textContent);

  //     // Alert the copied text
  //     alert('Domain UID copied!');
  //   };
  let metadata: DomainMetadata = {
    title: 'OpenMined',
    organization: 'OpenMined',
    description:
      'OpenMined is a community of researchers and engineers focused on building a privacy-preserving, decentralized future.',
    id: {
      value: 'openmined'
    },
    deployed_on: '2021-08-01T00:00:00.000Z'
  };
</script>

<div class="flex flex-col xl:flex-row w-full h-full xl:justify-around items-center gap-12">
  <DomainMetadataPanel {metadata} />
  <form class="contents">
    <Modal>
      <div
        class="flex flex-shrink-0 justify-between p-4 pb-0 flex-nowrap w-full h-min"
        slot="header"
      >
        <span class="block text-center w-full">
          <p class="text-2xl font-bold text-gray-800">Welcome</p>
        </span>
      </div>
      <div class="contents" slot="body">
        <DomainOnlineIndicator />
        <Input label="Email" type="email" id="email" placeholder="info@openmined.org" required />
        <Input label="Password" type="password" id="password" placeholder="******" required />
        <p class="text-center">
          Don't have an account yet? Apply for an account <a href="/signup">here</a>.
        </p>
      </div>
      <Button variant="secondary" slot="button-group">Login</Button>
    </Modal>
  </form>
</div>
