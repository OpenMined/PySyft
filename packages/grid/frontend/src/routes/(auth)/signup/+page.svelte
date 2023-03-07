<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import TagCloud from '$lib/components/TagCloud.svelte';
  import DomainMetadataPanel from '$lib/components/authentication/DomainMetadataPanel.svelte';
  import DomainOnlineIndicator from '$lib/components/DomainOnlineIndicator.svelte';
  import Modal from '$lib/components/NewModal.svelte';
  import Input from '$lib/components/Input.svelte';
  import type { DomainMetadata } from '../../../types/domain/metadata';

  // async function createUser(client) {
  //     let email = document.getElementById('email') ? document.getElementById('email').value : null;
  //     let password = document.getElementById('password')
  //       ? document.getElementById('password').value
  //       : null;
  //     let passwordConfirmation = document.getElementById('password')
  //       ? document.getElementById('confirm').value
  //       : null;
  //     let name = document.getElementById('fullname')
  //       ? document.getElementById('fullname').value
  //       : null;
  //     let institution = document.getElementById('company')
  //       ? document.getElementById('company').value
  //       : null;
  //     let website = document.getElementById('website')
  //       ? document.getElementById('website').value
  //       : null;

  //     if (password !== passwordConfirmation) {
  //       throw Error('Password and password confirmation mismatch');
  //     }

  //     let newUser = {
  //       email: email,
  //       password: password,
  //       password_verify: passwordConfirmation,
  //       name: name,
  //       institution: institution,
  //       website: website
  //     };

  //     // Filter attributes that doesn't exist
  //     Object.keys(newUser).forEach((k) => newUser[k] == null && delete newUser[k]);
  //     await client.register(newUser); // This will return a success message and the new user info
  //   }

  let metadata: DomainMetadata = {
    title: 'OpenMined',
    organization: 'OpenMined',
    description:
      'OpenMined is a community of researchers and engineers focused on building a privacy-preserving, decentralized future.',
    id: {
      value: 'openmined'
    },
    deployed_on: '2021-08-01T00:00:00.000Z',
    tags: []
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
          <p class="text-2xl font-bold text-gray-800">Apply for an account</p>
        </span>
      </div>
      <div class="contents" slot="body">
        <DomainOnlineIndicator />
        <div class="w-full gap-6 flex">
          <Input label="Full name" id="fullName" placeholder="Jane Doe" required />
          <Input label="Company/Institution" id="organization" placeholder="OpenMined University" />
        </div>
        <Input label="Email" id="email" placeholder="info@openmined.org" required />
        <div class="w-full gap-6 flex">
          <Input label="Password" id="password" placeholder="******" required />
          <Input label="Confirm Password" id="confirm_password" placeholder="******" required />
        </div>
        <Input label="Website/Profile" id="website" placeholder="https://openmined.org" />
        <p class="text-center">
          Already have an account? Sign in <a href="/login">here</a>.
        </p>
      </div>
      <Button variant="secondary" slot="button-group">Sign up</Button>
    </Modal>
  </form>
</div>
