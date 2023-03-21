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
  import { data } from './mockDatasetData.ts';
  import { onMount } from 'svelte';

  let datasets = [];

  onMount(async () => {
    await getClient()
      .then((client) => {
        datasets = client.datasets();

        console.log(`datasets: ${JSON.stringify(datasets, null, 1)}`);
      })
      .catch((error) => {
        console.log(error);
      });
  });

  let searchValue = '';
  let showModal = false;

  let visible = true;
  let openDatasetName = '';
  let openDatasetAuthor = '';
  let openDatasetLastUpdated = '';
  let openDatasetAssets = '';
  let openDatasetRequests = '';
  let openDatasetFileSize = '';

  function showOpenDataset(event) {
    openDatasetName = event.detail.openName;
    openDatasetAuthor = event.detail.openAuthor;
    openDatasetLastUpdated = event.detail.openLastUpdated;
    openDatasetAssets = event.detail.openAssets;
    openDatasetRequests = event.detail.openRequests;
    openDatasetFileSize = event.detail.openFileSize;
    visible = false;
  }

  function showHome() {
    visible = true;
  }

  let dataShow;
  let originalData = Object.values(data);

  dataShow = originalData;

  let sortMenuOpen = false;
  let inputValue = '';

  const menuItems = [
    'Newest',
    'Oldest',
    'Most Activity',
    'Least Activity',
    'Largest File Size',
    'Smallest File Size'
  ];
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col">
  <div>
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

        <Badge variant="gray">Total: {dataShow.length}</Badge>
      </div>
    </section>
    <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14">
      {#if visible}
        <article class="w-full">
          {#each dataShow as d}
            <DatasetListItem
              on:hide={showOpenDataset}
              name={d.name}
              author={d.author}
              lastUpdated={d.lastUpdated}
              assets={d.assets}
              requests={d.requests}
              fileSize={d.fileSize}
            />
          {/each}
        </article>
      {:else}
        <DatasetDetail
          on:closeOpenCard={showHome}
          name={openDatasetName}
          author={openDatasetAuthor}
          lastUpdated={openDatasetLastUpdated}
          assets={openDatasetAssets}
          requests={openDatasetRequests}
          fileSize={openDatasetFileSize}
        />
      {/if}
    </section>
  </div>
</main>

<style lang="postcss">
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
    @apply hidden absolute bg-white-50 rounded min-w-max;
  }

  .show {
    @apply block;
  }
</style>
