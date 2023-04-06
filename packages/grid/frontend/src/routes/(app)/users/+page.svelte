<script>
  import UserListItem from './userListItem.svelte';
  import UserDetail from './userDetail.svelte';
  import { getClient } from '$lib/store';
  import { onMount } from 'svelte';

  let client = '';
  $: selectedUser = {};

  onMount(async () => {
    await getClient()
      .then((response) => {
        client = response;
      })
      .catch((error) => {
        console.log(error);
      });
  });

  let pages = {
    isList: true,
    isDetail: false
  };

  const setPage = async (current) => {
    if (pages[current]) return;
    else {
      for (let page in pages) {
        pages[page] = page === current ? true : false;
      }
    }
  };
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col">
  <div class="page-container">
    {#if pages.isList}
      <UserListItem

        bind:selectedUser
        on:setPage={(event) => {
          setPage(event.detail);
        }}
      />
    {/if}
    {#if pages.isDetail}
      <UserDetail
        bind:selectedUser
        on:setPage={(event) => {
          setPage(event.detail);
        }}
      />
    {/if}
  </div>
</main>

<style>
  .page-container {
    width: 85%;
    padding-top: 12px;
    padding-left: 100px;
    padding-right: 100px;
    position: absolute;
    height: 93%;
    top: 7%;
    left: 15%;
  }
</style>
