<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import Modal from '$lib/components/NewModal.svelte';
  import DomainMetadataPanel from '$lib/components/authentication/DomainMetadataPanel.svelte';
  import Input from '$lib/components/Input.svelte';
  import DomainOnlineIndicator from '$lib/components/DomainOnlineIndicator.svelte';
  import { getClient } from '$lib/store';
  import { goto } from '$app/navigation';
  import type { DomainOnlineStatus } from '../../../types/domain/onlineIndicator';
  let status: DomainOnlineStatus = 'online';

  async function login({ email, password }, client) {
    await client
      .login(email.value, password.value)
      .then(() => {
        goto('/home');
      })
      .catch((error) => {
        console.log(error);
      });
  }
</script>

<div class="flex flex-col xl:flex-row w-full h-full xl:justify-around items-center gap-12">
  {#await getClient() then client}
    {#await client.metadata then metadata}
      <DomainMetadataPanel {metadata} />
      <form class="contents" on:submit|preventDefault={(e) => login(e.target, client)}>
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
            <div class="flex justify-center items-center gap-2">
              <DomainOnlineIndicator />
              <p class="text-600">
                {#if status === 'pending'}
                  Checking connection
                {:else}
                  Domain {status}
                {/if}
              </p>
            </div>
            <Input
              label="Email"
              type="email"
              id="email"
              placeholder="info@openmined.org"
              required
            />
            <Input label="Password" type="password" id="password" placeholder="******" required />
            <p class="text-center">
              Don't have an account yet? Apply for an account <a
                href="/signup"
                class="text-primary-600 underline hover:opacity-50">here</a
              >.
            </p>
          </div>
          <Button variant="secondary" slot="button-group">Login</Button>
        </Modal>
      </form>
    {/await}
  {/await}
</div>
