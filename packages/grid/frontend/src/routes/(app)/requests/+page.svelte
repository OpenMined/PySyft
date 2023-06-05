<script lang="ts">
  import { onMount } from 'svelte';
  import PrismCode from '$lib/components/PrismCode.svelte';
  import { getAllCodeRequests, getCodeRequest, getAllRequests } from '$lib/api/requests';
  import Tabs from '$lib/components/Tabs.svelte';
  import debounce from 'just-debounce-it';
  import Search from '$lib/components/Search.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import RequestNoneFound from '$lib/components/Requests/RequestNoneFound.svelte';
  import RequestListItem from '$lib/components/Requests/RequestListItem.svelte';

  let requests;
  let searchTerm = '';
  let codeList = [];

  let tabs = [{ label: 'Queue', id: 'tab1' }];
  let currentTab = tabs[0].id;

  onMount(async () => {
    requests = await getAllRequests();
  });

  const search = debounce(async () => {
    if (searchTerm === '') codeList = await getAllCodeRequests();
    else codeList = await getCodeRequest(searchTerm);
  }, 300);

  const mock_requests = [
    {
      user: {
        name: 'John Doe',
        role: 'Data Scientist',
        organization: 'Linkedin'
      },
      request: {
        date: 'Jan 2, 2023',
        reason: 'I would like to run this code',
        replies: [
          { text: 'I want to run my experiment', uuid: '123456' },
          { text: ' I cant do it this way', uuid: '123457' },
          { text: 'Why not?', uuid: '123456' }
        ]
      }
    },
    {
      user: {
        name: 'Ana',
        role: 'Data Scientist',
        organization: 'UFMG'
      },
      request: {
        date: 'Jan 1, 2023',
        reason: 'I want this for my PhD research!',
        replies: [
          { text: 'I want to run my experiment', uuid: '123456' },
          { text: ' I cant do it this way', uuid: '123457' },
          { text: 'Why not?', uuid: '123456' }
        ]
      }
    },
    {
      user: {
        name: 'Jana Doe',
        role: 'Data Scientist',
        organization: 'UCSF'
      },
      request: {
        date: 'Jan 5, 2023',
        reason: 'I would like to run this code',
        replies: [
          { text: 'I want to run my experiment', uuid: '123456' },
          { text: ' I cant do it this way', uuid: '123457' },
          { text: 'Why not?', uuid: '123456' }
        ]
      }
    }
  ];
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
          <RequestListItem user={request.user} request={request.request} message={request.message} />
          <!-- prettier-ignore -->
          <!--<PrismCode code={request.raw_code} />-->
        {/each}
      </div>
    {/if}
  </div>
</section>
