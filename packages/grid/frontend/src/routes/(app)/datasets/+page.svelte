<script>
  import GetStartedModal from './getStartedModal.svelte';
  import DatasetListItem from './datasetListItem.svelte';
  import DatasetDetail from './datasetDetail.svelte';
  import { data } from './mockDatasetData.ts';

  let modalFlag = true;
  let visible = true;
  let openDatasetName = '';
  let openDatasetAuthor = '';
  let openDatasetLastUpdated = '';
  let openDatasetFileSize = '';

  function showOpenDataset(event) {
    openDatasetName = event.detail.openName;
    openDatasetAuthor = event.detail.openAuthor;
    openDatasetLastUpdated = event.detail.openLastUpdated;
    openDatasetFileSize = event.detail.openFileSize;
    visible = false;
  }

  function showHome() {
    visible = true;
  }

  let dataShow;
  let originalData = Object.values(data);

  dataShow = originalData;
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col">
  <GetStartedModal showModal={modalFlag} />

  <!-- Header -->
  <h1>Datasets</h1>

  <!-- Body content -->
  <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14 h-full">
    {#if visible}
      <article class="w-full">
        {#each dataShow as d}
          <DatasetListItem
            on:hide={showOpenDataset}
            name={d.name}
            author={d.author}
            lastUpdated={d.lastUpdated}
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
        fileSize={openDatasetFileSize}
      />
    {/if}
  </section>
</main>
