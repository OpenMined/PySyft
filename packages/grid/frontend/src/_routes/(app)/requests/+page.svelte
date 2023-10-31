<script lang="ts">
  import { onMount } from 'svelte';
  import PrismCode from '$lib/components/PrismCode.svelte';
  import { getAllRequests, filterRequests } from '$lib/api/requests';
  import Tabs from '$lib/components/Tabs.svelte';
  import debounce from 'just-debounce-it';
  import Search from '$lib/components/Search.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import RequestNoneFound from '$lib/components/Requests/RequestNoneFound.svelte';
  import RequestListItem from '$lib/components/Requests/RequestListItem.svelte';

  let requests = [];
  let searchTerm = '';

  let tabs = [{ label: 'Queue', id: 'tab1' }];
  let currentTab = tabs[0].id;

  onMount(async () => {
    requests = await getAllRequests();
  });

  const search = debounce(async () => {
    if (searchTerm === '') requests = await getAllRequests();
    else requests = await filterRequests(searchTerm);
  }, 300);
</script>

<section class="p-10">
  <h1 class="text-3xl font-medium font-rubik pb-6">Requests</h1>
  <Tabs {tabs} bind:active={currentTab} />
  <div class="space-y-6 px-20 pt-8">
    <div class="flex items-center justify-between">
      <div class="w-full max-w-[378px]">
        <Search
          type="text"
          placeholder="Search by name"
          bind:value={searchTerm}
          on:input={search}
        />
      </div>
      <div class="flex-shrink-0">
        <Badge variant="gray">Total: {requests?.length || 0}</Badge>
      </div>
    </div>
    {#if !requests}
      <div>Loading...</div>
    {:else if requests.length === 0}
      <RequestNoneFound />
    {:else}
      <div class="flex flex-col divide-y divide-gray-100">
        {#each requests as request}
          <RequestListItem
            user={request.user}
            request={request.request}
            message={request.message}
          />
        {/each}
      </div>
    {/if}
  </div>
</section>
