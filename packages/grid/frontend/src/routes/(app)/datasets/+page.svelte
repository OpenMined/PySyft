<script>
  import Badge from '$lib/components/Badge.svelte';
  import Button from '$lib/components/Button.svelte';
  import Link from '$lib/components/Link.svelte';
  import NewDatasetModal from './newDatasetModal.svelte';
  import DatasetListItem from './datasetListItem.svelte';
  import DatasetDetail from './datasetDetail.svelte';
  import Search from 'svelte-search';
  import Fa from 'svelte-fa';
  import { getClient } from '$lib/store';
  import { faChevronDown } from '@fortawesome/free-solid-svg-icons';
  import { onMount } from 'svelte';

  let datasets = [];

  onMount(async () => {
    await getClient()
      .then((client) => {
        client.datasets.then((response) => {
          datasets = response;
        });
      })
      .catch((error) => {
        console.log(error);
      });
  });

  let searchValue = '';
  let showModal = false;

  let visible = true;
  let openDataset = {};
  function showOpenDataset(event) {
    openDataset = {
      name: event.detail.openName,
      datasetId: event.detail.openDatasetId,
      author: event.detail.openAuthor,
      assets: event.detail.openAssets,
      description: event.detail.openDescription,
      lastUpdated: event.detail.openLastUpdated,
      requests: event.detail.openRequests,
      fileSize: event.detail.openFileSize
    };
    visible = false;
  }

  function showHome() {
    visible = true;
  }

  const menuItems = [
    'Newest',
    'Oldest',
    'Most Activity',
    'Least Activity',
    'Largest File Size',
    'Smallest File Size'
  ];

  let sortMenuOpen = false;
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col">
  <div class="page-container overflow-auto">
    {#if visible}
      <NewDatasetModal bind:showModal />

      <!-- Header -->
      <div class="flex justify-between">
        <h2
          class="flex justify-left text-gray-800 font-rubik text-2xl leading-normal font-medium pb-4"
        >
          Datasets
        </h2>
        <Button variant="black" action={() => (showModal = true)}>+ New Dataset</Button>
      </div>

      <!-- Body content -->
      <section class="md:flex justify-between md:gap-x-[62px] lg:gap-x-[124px] mt-14">
        <Search
          bind:searchValue
          placeholder="Search by name"
          debounce={800}
          autofocus
          hideLabel
          on:submit={(e) => e.preventDefault()}
        />

        <div class="mr-6">
          <div class="flex items-center">
            <div class="dropdown pr-2">
              <Button variant="white" action={() => (sortMenuOpen = !sortMenuOpen)}
                >Sort By<Fa class="pl-2" icon={faChevronDown} size="xs" /></Button
              >

              <div class:show={sortMenuOpen} class="dropdown-content">
                {#each menuItems as item}
                  <Link link={item} />
                {/each}
              </div>
            </div>

            <div class="pl-2">
              <Badge variant="gray">Total: {datasets.length}</Badge>
            </div>
          </div>
        </div>
      </section>
      <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14">
        <article class="w-full">
          {#each datasets as d}
            <DatasetListItem
              on:hide={showOpenDataset}
              name={d.name}
              author={d.author}
              datasetId={d.id.value}
              description={d.description}
              lastUpdated={d.updated_at}
              assets={d.asset_list.length}
              requests={d.requests}
              fileSize={d.fileSize}
            />
          {/each}
        </article>
      </section>
    {:else}
      <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14">
        <article class="w-full">
          <DatasetDetail on:closeOpenCard={showHome} dataset={openDataset} />
        </article>
      </section>
    {/if}
  </div>
</main>

<style lang="postcss">
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

  :global([data-svelte-search]) {
    @apply w-5/12;
  }

  :global([data-svelte-search] input) {
    @apply w-full rounded-3xl;
  }

  .dropdown {
    @apply relative inline-block;
  }

  .dropdown-content {
    @apply hidden absolute bg-white-50 rounded min-w-max border;
  }

  .show {
    @apply block;
  }
</style>
