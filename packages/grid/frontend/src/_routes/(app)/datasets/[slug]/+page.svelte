<script>
  import { page } from "$app/stores"
  import CaretLeft from "$lib/components/icons/CaretLeft.svelte"
  import DatasetModalDelete from "$lib/components/Datasets/DatasetModalDelete.svelte"
  import TableFillIcon from "$lib/components/icons/TableFillIcon.svelte"
  import TableIcon from "$lib/components/icons/TableIcon.svelte"
  import Tabs from "$lib/components/Tabs.svelte"
  import ThreeDotsVertical from "$lib/components/icons/ThreeDotsVertical.svelte"
  import { onMount } from "svelte"
  import { getDataset, deleteDataset } from "$lib/api/datasets"
  import { goto } from "$app/navigation"
  import UserList from "$lib/components/icons/UserList.svelte"

  let dataset = null
  let openModalDelete = false

  let currentTab
  let tabs = [
    { label: "Overview", id: "tab1", content: TableFillIcon },
    { label: "Assets", icon: TableFillIcon, id: "tab2", content: TableIcon },
  ]

  let openCitationsAccordion = false
  let openContributorsAccordion = false

  function handleClick() {
    openModalDelete = !openModalDelete
  }

  async function handleDelete() {
    try {
      await deleteDataset(dataset.id.value)
      goto("/datasets")
    } catch (error) {
      console.error(error)
    }
  }

  onMount(async () => {
    dataset = await getDataset($page.params.slug)
  })
</script>

<div class="p-6 flex flex-col gap-8">
  {#if dataset === null}
    <h2>Loading</h2>
  {:else if dataset}
    <section>
      <a
        href="/datasets"
        class="inline-flex gap-2 text-primary-500 items-center"
      >
        <CaretLeft class="w-4 h-4" /> Back
      </a>
      <div class="p-4 pb-6">
        <div class="flex gap-4">
          <span class="inline-flex items-center gap-4 w-full">
            <h1 class="font-bold text-3xl leading-[1.4]">
              {dataset.name}
            </h1>
            <!-- <PencilIcon class="w-4 h-4 text-primary-500 flex-shrink-0" /> -->
          </span>
          <button class="self-start" on:click={handleClick}>
            <ThreeDotsVertical
              class="w-6 h-6 desktop:w-8 desktop:h-8 text-gray-800"
            />
          </button>
        </div>
        <div class="flex gap-3 text-gray-600">
          <!-- <span>{dataset.owner}</span> -->
          <!-- <span class="text-[8px]">●</span> -->
          <span>{dataset.updated_at}</span>
        </div>
        <div class="flex gap-3 text-gray-600">
          <span>UID:</span>
          <span>{dataset.id.value}</span>
        </div>
        <div class="pt-10 pb-6 flex gap-3 text-gray-600">
          <span class="inline-flex gap-2">
            <TableIcon class="w-5 h-5 text-gray-800" weight="fill" />
            <span>{dataset.asset_list.length}</span>
          </span>
          <span class="dot">●</span>
          <span>File Size ({dataset.mb_size} MB)</span>
        </div>
      </div>
    </section>
    <Tabs {tabs} bind:active={currentTab} />
    {#if currentTab === tabs[0].id}
      <div class="w-full grid grid-cols-6">
        <div class="col-span-4 col-start-2">
          <section>
            <h5 class="text-2xl leading-[1.5]">Description</h5>
            <p class="pt-6">{dataset.description}</p>
            <div class="pt-24">
              <div class="flex flex-col">
                <button
                  class="w-full min-h-[60px] flex items-center gap-2 p-3 border border-gray-100 cursor-pointer accordion"
                  class:active={openCitationsAccordion}
                  on:click={() =>
                    (openCitationsAccordion = !openCitationsAccordion)}
                  disabled={!dataset.citation}
                >
                  <CaretLeft
                    class="w-5 h-5 transition transform duration-500 {openCitationsAccordion
                      ? '-rotate-90'
                      : '-rotate-180'}"
                  />
                  <h2 class="text-xl leading-[1.5]">Citation</h2>
                </button>
                <div
                  class="transition duration-500 px-12 pt-4 pb-8 bg-gray-50"
                  class:hidden={!openCitationsAccordion}
                  class:h-0={!openCitationsAccordion}
                  class:h-max={openCitationsAccordion}
                  class:block={openCitationsAccordion}
                >
                  <h3>{dataset.citation}</h3>
                </div>
                <button
                  class="w-full min-h-[60px] flex items-center gap-2 p-3 border border-gray-100 cursor-pointer accordion"
                  class:active={openContributorsAccordion}
                  on:click={() =>
                    (openContributorsAccordion = !openContributorsAccordion)}
                  disabled={!dataset.contributors?.length}
                >
                  <CaretLeft
                    class="w-5 h-5 transition transform duration-500 {openContributorsAccordion
                      ? '-rotate-90'
                      : '-rotate-180'}"
                  />
                  <h2 class="text-xl leading-[1.5]">Contributors</h2>
                </button>
                <div
                  class="transition duration-500 px-12 pt-4 pb-8 bg-gray-50"
                  class:hidden={!openContributorsAccordion}
                  class:h-0={!openContributorsAccordion}
                  class:h-max={openContributorsAccordion}
                  class:block={openContributorsAccordion}
                >
                  <ul class="flex flex-col gap-10 py-4">
                    {#each dataset.contributors as contributor}
                      <li class="flex gap-2 w-full">
                        <div class="flex flex-col py-4 flex-wrap w-full">
                          <h4 class="uppercase text-sm font-bold">Name</h4>
                          <p class="leading-[1] tracking-[0.75]">
                            {contributor.name}
                          </p>
                        </div>
                        <div class="flex flex-col py-4 flex-wrap w-full">
                          <h4 class="uppercase text-sm font-bold">Email</h4>
                          <p class="leading-[1] tracking-[0.75]">
                            {contributor.email}
                          </p>
                        </div>
                        <div class="flex flex-col py-4 flex-wrap w-full">
                          <h4 class="uppercase text-sm font-bold">Role</h4>
                          <p class="leading-[1] tracking-[0.75]">
                            {contributor.role}
                          </p>
                        </div>
                      </li>
                    {/each}
                  </ul>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    {:else if currentTab === tabs[1].id}
      <section class="py-20 px-10 gap-6 flex">
        {#each dataset.asset_list as asset}
          <div
            class="flex-col p-6 pb-0 min-w-[320px] max-w-[600px] rounded-[14px] flex-nowrap gap-4 border border-gray-200 p-6 gap-1 text-gray-800"
          >
            <div class="flex w-full">
              <h2
                class="whitespace-pre-wrap break-words text-2xl leading-[1.2] pl-2 pr-2.5 pb-4 w-full"
              >
                {asset.name}
              </h2>
            </div>
            <div class="w-full flex gap-3">
              {#if asset.contributors?.[0]}
                <p>{asset.contributors[0].name}</p>
                <!-- <span class="dot">●</span> -->
              {/if}
              <!-- <p>{asset.updated_at}</p> -->
              <!-- <span class="dot">●</span> -->
              <!-- <p>({asset.mb_size} MB)</p> -->
            </div>
            <div class="w-full flex gap-3">
              <p>UID:</p>
              <span>{asset.id.value}</span>
            </div>
            <div class="w-full flex gap-2 pt-6 pb-4 items-center">
              <UserList class="w-5 h-5" />
              <p>{asset.data_subjects?.length || 0}</p>
              <span class="dot">●</span>
              <p>Shape ({asset.shape?.join(" x ")})</p>
            </div>
            <div class="w-full border-t border-gray-100 gap-4 flex pt-6 pb-10">
              <p>Type</p>
              <span class="font-bold">
                {asset.mock_is_real ? "Mock Data" : "Asset"}
              </span>
            </div>
          </div>
        {/each}
      </section>
    {/if}
  {/if}
</div>

{#if openModalDelete}
  <DatasetModalDelete onClose={handleClick} onDelete={handleDelete} />
{/if}

<style lang="postcss">
  button[disabled] {
    @apply cursor-not-allowed opacity-50;
  }

  .accordion.active {
    @apply bg-gray-50;
  }
</style>
