<script lang="ts">
  import debounce from "just-debounce-it"
  import Filter from "$lib/components/Filter.svelte"
  import Search from "$lib/components/Search.svelte"
  import Button from "$lib/components/Button.svelte"
  import LoadingDatasets from "$lib/components/LoadingDatasets.svelte"
  import Pagination from "$lib/components/Pagination.svelte"
  import DatasetListItem from "$lib/components/Datasets/DatasetListItem.svelte"
  import NoDatasetFound from "$lib/components/Datasets/DatasetNoneFound.svelte"
  import PlusIcon from "$lib/components/icons/PlusIcon.svelte"
  import DatasetModalNew from "$lib/components/Datasets/DatasetModalNew.svelte"
  import { invalidate } from "$app/navigation"
  import type { PageData } from "./$types"

  export let data: PageData

  let datasets = null
  let total = 0
  let paginators = [5, 10, 15, 20, 25]
  let page_size = 5,
    page_index = 0,
    page_row = 5
  let openModalNew = false
  let searchTerm = ""

  function handleClick() {
    openModalNew = !openModalNew
  }

  const search = debounce(async () => {
    if (searchTerm === "") return invalidate("dataset:list")
    const res = await fetch(
      `/_syft_api/datasets/search?name=${searchTerm}&page_size=${page_size}`
    )
    const json = await res.json()
    datasets = json.datasets
    total = json.total
  }, 300)

  const handleUpdate = async () => {
    const res = await fetch("/_syft_api/datasets")
    const json = await res.json()
    datasets = json.datasets
    total = json.total
  }
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
      <div class="flex items-center justify-between w-full gap-8">
        <div class="w-full max-w-[378px] pb-5">
          <Search
            type="text"
            placeholder="Search by name"
            bind:value={searchTerm}
            on:input={search}
          />
        </div>
        <div>
          <Filter
            variant="gray"
            filters={paginators}
            bind:filter={page_size}
            bind:index={page_index}
            on:setFilter={handleUpdate}
          >
            Filter:
          </Filter>
        </div>
      </div>

      {#if datasets === null}
        <LoadingDatasets />
      {:else if datasets.length === 0}
        <NoDatasetFound />
      {:else if datasets.length > 0}
        {#each datasets as dataset}
          <DatasetListItem {dataset} />
        {/each}
        <div
          class="flex justify-center items-center mt-8 mb-8 divide-y divide-gray-100"
        >
          <Pagination
            variant="gray"
            {total}
            {page_size}
            {page_row}
            bind:page_index
            on:setPagination={handleUpdate}
          />
        </div>
      {/if}
    </section>
  </div>
</div>

{#if openModalNew}
  <DatasetModalNew onClose={handleClick} />
{/if}
