<script lang="ts">
  import { user } from '$lib/store';
  import { getInitials } from '$lib/utils';
  import Avatar from '$lib/components/Avatar.svelte';
  import Button from '$lib/components/Button.svelte';
  import ButtonGhost from '$lib/components/ButtonGhost.svelte';
  import Dialog from '$lib/components/Dialog.svelte';
  import Input from '$lib/components/Input.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import Tabs from '$lib/components/Tabs.svelte';
  import TextArea from '$lib/components/TextArea.svelte';
  import XIcon from '$lib/components/icons/XIcon.svelte';
  import { getSelf } from '$lib/api/users';

  let tabs = [{ label: 'Account', id: 'tab1' }];
  let currentTab = tabs[0].id;
  let openModal: 'name' | 'website' | 'email' | 'password' | null = null;

  let { name, website, email } = $user ?? {};

  let password = '';
  let confirmPassword = '';

  const onClose = () => (openModal = null);

  const handleUpdate = async () => {
    try {
      await getSelf();
      onClose();
    } catch (error) {
      console.log(error);
    }
  };

  $: userInitials = getInitials($user?.name);

  $: profileInformation = [
    { label: 'Name', id: 'name', value: $user?.name },
    { label: 'Website', id: 'website', value: $user?.website }
  ];

  $: authenticationInformation = [
    { label: 'Email', id: 'email', value: $user?.email },
    { label: 'Password', id: 'password' }
  ];
</script>

<div>
  <Tabs {tabs} bind:active={currentTab} />
  <section class="pt-9 pl-[60px] pr-10 pb-20 flex flex-col tablet:flex-row gap-11">
    {#if !$user}
      Loading...
    {:else}
      <span class="block w-32 h-32 p-4">
        <Avatar initials={userInitials} />
      </span>
      <div class="w-full divide-y divide-gray-200">
        <div class="w-full flex flex-col py-4 gap-3">
          <h3 class="all-caps">Profile information</h3>
          {#each profileInformation as { label, id, value }}
            <div class="flex flex-col gap-2 pt-2 pb-4">
              <h4 class="text-xl capitalize">{label}</h4>
              {#if value}
                <p data-testid={id}>{value}</p>
              {/if}
              <span>
                <button class="link capitalize" on:click={() => (openModal = id)}>
                  Change {label}
                </button>
              </span>
            </div>
          {/each}
        </div>
        <div class="w-full flex flex-col py-4 gap-3">
          <h3 class="all-caps">Authentication information</h3>
          {#each authenticationInformation as { label, id, value }}
            <div class="flex flex-col gap-2 pt-2 pb-4">
              <h4 class="text-xl capitalize">{label}</h4>
              {#if value}
                <p data-testid={id}>{value}</p>
              {/if}
              <span>
                <button class="link capitalize" on:click={() => (openModal = id)}>
                  Change {label}
                </button>
              </span>
            </div>
          {/each}
        </div>
        <div class="w-full flex flex-col py-4 gap-3">
          <h3 class="all-caps">Caution zone</h3>
          <div class="flex flex-col gap-2 pt-2 pb-4">
            <h4 class="text-xl capitalize">Account</h4>
            <span>
              <button class="link capitalize" on:click={() => (openModal = 'delete')}>
                Delete account
              </button>
            </span>
          </div>
        </div>
      </div>
    {/if}
  </section>
</div>

{#if ['email', 'website', 'password', 'name'].includes(openModal)}
  <Dialog bind:open={openModal}>
    <Modal>
      <div slot="header" class="w-full text-right">
        <button on:click={onClose}>
          <XIcon class="w-6 h-6" />
        </button>
      </div>
      <div slot="body" class="flex flex-col gap-4">
        {#if openModal === 'email'}
          <Input label="Email" required bind:value={email} id="email" />
        {:else if openModal === 'website'}
          <Input label="Website" bind:value={website} id="website" />
        {:else if openModal === 'name'}
          <Input label="Name" bind:value={name} id="name" />
        {:else if openModal === 'password'}
          <Input label="Password" bind:value={password} id="password" />
          <Input label="Confirm Password" bind:value={confirmPassword} id="confirmPassword" />
        {/if}
      </div>
      <div class="flex w-full justify-end" slot="button-group">
        <div class="w-full justify-end flex px-4 gap-4">
          <ButtonGhost on:click={onClose}>Cancel</ButtonGhost>
          <Button type="submit" variant="secondary" on:click={handleUpdate}>Save</Button>
        </div>
      </div>
    </Modal>
  </Dialog>
{:else if openModal === 'delete'}
  <Dialog>
    <Modal>OK</Modal>
  </Dialog>
{/if}
