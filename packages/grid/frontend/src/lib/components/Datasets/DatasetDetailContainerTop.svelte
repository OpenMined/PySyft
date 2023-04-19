<script>
  import CaretLeft from '$lib/components/icons/CaretLeft.svelte';
  import PencilIcon from '$lib/components/icons/PencilIcon.svelte';
  import TableIcon from '$lib/components/icons/TableIcon.svelte';
  import ThreeDotsVertical from '$lib/components/icons/ThreeDotsVertical.svelte';
  import DatasetModalDelete from '$lib/components/Datasets/DatasetModalDelete.svelte';

  let openModalDelete = false;

  function handleClick() {
    openModalDelete = !openModalDelete;
  }

  export let dataset = {};
</script>

<section>
  <a href="/datasets" class="inline-flex gap-2 text-primary-500 items-center">
    <CaretLeft class="w-4 h-4" /> Back
  </a>
  <div class="p-4 pb-6">
    <div class="flex gap-4">
      <span class="inline-flex items-center gap-4 w-full">
        <h1 class="font-bold text-3xl leading-[1.4]">
          {dataset.name}
        </h1>
        <PencilIcon class="w-4 h-4 text-primary-500 flex-shrink-0" />
      </span>
      <button class="self-start" on:click={handleClick}>
        <ThreeDotsVertical class="w-6 h-6 desktop:w-8 desktop:h-8 text-gray-800" />
      </button>
    </div>
    <div class="flex gap-3 text-gray-600">
      <span>{dataset.owner}</span>
      <span class="text-[8px]">●</span>
      <span>{dataset.date}</span>
    </div>
    <div class="flex gap-3 text-gray-600">
      <span>UID:</span>
      <span>{dataset.id}</span>
    </div>
    <div class="pt-10 pb-6 flex gap-3 text-gray-600">
      <span class="inline-flex gap-2">
        <TableIcon class="w-5 h-5 text-gray-800" weight="fill" />
        <span>{dataset.assets}</span>
      </span>
      <span class="text-[8px]">●</span>
      <span>File Size ({dataset.filesize})</span>
    </div>
  </div>
</section>

{#if openModalDelete}
  <DatasetModalDelete onClose={handleClick} />
{/if}
