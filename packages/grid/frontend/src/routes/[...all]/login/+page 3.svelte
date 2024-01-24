<script lang="ts">
  import { goto } from '$app/navigation';
  import { metadata, user } from '$lib/store';
  import { login } from '$lib/api/auth';
  import Button from '$lib/components/Button.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import DomainMetadataPanel from '$lib/components/authentication/DomainMetadataPanel.svelte';
  import Input from '$lib/components/Input.svelte';
  import DomainOnlineIndicator from '$lib/components/DomainOnlineIndicator.svelte';
  import type { DomainOnlineStatus } from '../../../types/domain/onlineIndicator';
  import { getUserIdFromStorage } from '$lib/api/keys';
  import { getSelf } from '$lib/api/users';

  let status: DomainOnlineStatus = 'online';
  let email = '';
  let password = '';
  $: loginError = '';
  async function handleSubmit() {
    try {
      await login({ email, password });
      if (getUserIdFromStorage()) {
        const updatedUser = await getSelf();
        user.set(updatedUser);
      }
      goto('/datasets');
    } catch (error) {
      loginError = error.message;
    }
  }
</script>

<div class="flex flex-col xl:flex-row w-full h-full xl:justify-around items-center gap-12">
  <DomainMetadataPanel metadata={$metadata} />
  <form class="contents" on:submit|preventDefault={handleSubmit}>
    <section class="w-full max-w-[681px]">
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
            bind:error={loginError}
            bind:value={email}
            required
            data-testid="email"
          />
          <Input
            label="Password"
            type="password"
            id="password"
            placeholder="******"
            bind:value={password}
            bind:error={loginError}
            required
            data-testid="password"
          />
          <p class="text-center text-rose-500" hidden={!loginError}>{loginError}</p>
          <p class="text-center">
            Don't have an account yet? Apply for an account <a
              href="/signup"
              class="text-primary-600 underline hover:opacity-50"
            >
              here
            </a>
            .
          </p>
        </div>
        <Button type="submit" variant="secondary" slot="button-group">Login</Button>
      </Modal>
    </section>
  </form>
</div>
