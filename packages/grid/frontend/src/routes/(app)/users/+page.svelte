<script lang="ts">
  import { onMount } from 'svelte';
  import debounce from 'just-debounce-it';
  import { getAllUsers, getSelf, searchUsersByName } from '$lib/api/users';
  import Badge from '$lib/components/Badge.svelte';
  import Filter from '$lib/components/Filter.svelte';
  import Search from '$lib/components/Search.svelte';
  import Pagination from '$lib/components/Pagination.svelte';
  import UserListItem from '$lib/components/Users/UserListItem.svelte';
  import PlusIcon from '$lib/components/icons/PlusIcon.svelte';
  import UserNewModal from '$lib/components/Users/UserNewModal.svelte';
  import UserCreateModal from '$lib/components/Users/UserCreateModal.svelte';
  import type { UserListView } from '../../../types/domain/users';

  let searchTerm = '';
  let userList: UserListView[] = [];
  let total: number = 0;
  let paginators: number[] = [1, 2, 3, 4, 5];
  let page_size: number = 2, page_index: number = 0, page_row: number = 3;

  let openModal: string | null = null;

  onMount(async () => {
    const results = await getAllUsers(page_size, page_index);
    userList = results.users;
    total = results.total;
  });

  const closeModal = () => {
    openModal = null;
  };

  const onCreateGeneralUser = () => {
    openModal = 'step1';
  };

  const search = debounce(async () => {
    if (searchTerm === '') {
      const results = await getAllUsers(page_size);
      userList = results.users;
      total = results.total;
    } else {
      const results = await searchUsersByName(searchTerm, page_size);
      // const results = await searchUsersByName(searchTerm, page_size, 0);
      // userList = results.users;
      // total = results.total;
      // console.log(results)
    }
  }, 300);

  const handleUpdate = async () => {
    const results = await getAllUsers(page_size, page_index);
    userList = results.users;
    total = results.total;
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
      <div class="flex gap-2.5">
        <Badge variant="gray">Total: {total || 0}</Badge>
        <Filter variant="gray" filters={paginators} bind:filter={page_size} bind:index={page_index} on:setFilter={handleUpdate}>Filter: </Filter>
      </div>
    </div>
  </div>
  <div class="divide-y divide-gray-100">
    {#each userList as user}
      <a class="block hover:bg-primary-100 cursor-pointer" href={`/users/${user.id.value}`}>
        <UserListItem {user} />
      </a>
    {/each}
  </div>
  <div class="flex justify-center items-center mb-8 divide-y divide-gray-100">
    <Pagination
      variant="gray"
      total={total}
      page_size={page_size}
      page_row={page_row}
      bind:page_index={page_index}
      on:setPagination={handleUpdate}
    ></Pagination>
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
