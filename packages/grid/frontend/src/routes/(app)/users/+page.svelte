<script lang="ts">
  import { onMount } from 'svelte';
  import debounce from 'just-debounce-it';
  import { getAllUsers, getSelf, searchUsersByName } from '$lib/api/users';
  import Badge from '$lib/components/Badge.svelte';
  import Search from '$lib/components/Search.svelte';
  import UserListItem from '$lib/components/Users/UserListItem.svelte';
  import PlusIcon from '$lib/components/icons/PlusIcon.svelte';
  import UserNewModal from '$lib/components/Users/UserNewModal.svelte';
  import UserCreateModal from '$lib/components/Users/UserCreateModal.svelte';
  import type { UserListView } from '../../../types/domain/users';

  let searchTerm = '';
  let userList: UserListView[] = [];

  let openModal: string | null = null;

  onMount(async () => {
    userList = await getAllUsers();
  });

  const closeModal = () => {
    openModal = null;
  };

  const onCreateGeneralUser = () => {
    openModal = 'step1';
  };

  const search = debounce(async () => {
    if (searchTerm === '') userList = await getAllUsers();
    else userList = await searchUsersByName(searchTerm);
  }, 300);

  const handleUpdate = async () => {
    userList = await getAllUsers();
  };
</script>

<div class="pt-8 desktop:pt-2 pl-16 pr-[140px] flex flex-col gap-[46px]">
  <div class="flex justify-center w-full">
    <div class="w-[438px] h-[263px]">
      <img
        src="images/illustrations/user-main.png"
        alt="User main alt"
        class="w-full h-fill object-contain"
      />
    </div>
  </div>
  <div class="flex items-center justify-between w-full gap-8">
    <div class="w-full max-w-[378px]">
      <Search type="text" placeholder="Search by name" bind:value={searchTerm} on:input={search} />
    </div>
    <div class="flex-shrink-0">
      <Badge variant="gray">Total: {userList?.length || 0}</Badge>
    </div>
  </div>
  <div class="divide-y divide-gray-100">
    {#each userList as user}
      <a class="block hover:bg-primary-100 cursor-pointer" href={`/users/${user.id.value}`}>
        <UserListItem {user} />
      </a>
    {/each}
  </div>
</div>
<div class="fixed bottom-10 right-12">
  <button
    class="bg-black text-white rounded-full w-14 h-14 flex items-center justify-center"
    on:click={() => (openModal = 'newUser')}
  >
    <PlusIcon class="w-6 h-6" />
  </button>
</div>

<UserNewModal open={openModal === 'newUser'} onClose={closeModal} {onCreateGeneralUser} />
<UserCreateModal open={openModal === 'step1'} onClose={closeModal} on:userUpdate={handleUpdate} />
