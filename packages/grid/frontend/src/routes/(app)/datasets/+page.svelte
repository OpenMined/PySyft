<script>
  import { onMount } from 'svelte';
  import Button from '$lib/components/Button.svelte';
  import DatasetListItem from '$lib/components/Datasets/DatasetListItem.svelte';
  import NoDatasetFound from '$lib/components/Datasets/DatasetNoneFound.svelte';
  import PlusIcon from '$lib/components/icons/PlusIcon.svelte';
  import DatasetModalNew from '$lib/components/Datasets/DatasetModalNew.svelte';
  import { getAllDatasets } from '$lib/api/datasets';

  let datasets = null;
  let openModalNew = false;

  function handleClick() {
    openModalNew = !openModalNew;
  }

  onMount(async () => {
    datasets = await getAllDatasets();
  });
</script>

<div class="grid grid-cols-6">
  <div class="w-full flex flex-col col-span-4 col-start-2">
    <!-- Heading -->
    <section class="heading pt-8 pb-4">
      <div class="flex justify-between">
        <h1 class="text-3xl leading-[1.2] whitespace-pre">Datasets</h1>
        <Button on:click={handleClick}>
          <div class="w-full flex items-center gap-1.5 flex-shrink-0">
            <PlusIcon class="w-4 h-4" />
            <span>New Dataset</span>
          </div>
        </Button>
      </div>
    </section>
    <!-- List Actions -->
    <!-- Body -->
    <section class="body pt-10">
      {#if datasets === null}
        <h2>Loading</h2>
      {:else if datasets.length === 0}
        <NoDatasetFound />
      {:else if datasets.length > 0}
        {#each datasets as dataset}
          <DatasetListItem {dataset} />
        {/each}
      {/if}
    </section>
  </div>
</div>

{#if openModalNew}
  <DatasetModalNew onClose={handleClick} />
{/if}
