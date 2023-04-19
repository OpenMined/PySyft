<script>
  import DatasetDetailContainerTop from '$lib/components/Datasets/DatasetDetailContainerTop.svelte';
  import TableIcon from '$lib/components/icons/TableIcon.svelte';
  import Tabs from '$lib/components/Tabs.svelte';
  import TableFillIcon from '$lib/components/icons/TableFillIcon.svelte';
  import CaretLeft from '$lib/components/icons/CaretLeft.svelte';

  export let dataset = {};

  let currentTab;
  let tabs = [
    { label: 'Overview', id: 'tab1', content: TableFillIcon },
    { label: 'Assets', icon: TableFillIcon, id: 'tab2', content: TableIcon }
  ];

  let openCitationsAccordion = false;
  let openContributorsAccordion = false;
</script>

<div class="p-6 flex flex-col gap-8">
  <DatasetDetailContainerTop {dataset} />
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
                on:click={() => (openCitationsAccordion = !openCitationsAccordion)}
                disabled={!dataset.citation}
              >
                <CaretLeft
                  class="w-5 h-5 transition transform duration-500 {openCitationsAccordion
                    ? '-rotate-90'
                    : '-rotate-180'}"
                />
                <h5 class="text-xl leading-[1.5]">Citation</h5>
              </button>
              <div
                class="transition px-12 pt-4 pb-8 bg-gray-100"
                class:hidden={!openCitationsAccordion}
                class:h-0={!openCitationsAccordion}
                class:h-max={openCitationsAccordion}
                class:block={openCitationsAccordion}
              >
                {dataset.citation}
              </div>
            </div>
            <div class="min-h-[60px] flex items-center gap-2 p-3 border border-gray-100">
              <CaretLeft
                class="w-5 h-5 transition transform duration-500 {openCitationsAccordion
                  ? '-rotate-90'
                  : '-rotate-180'}"
              />
              <h5 class="text-xl leading-[1.5]">Contributors</h5>
            </div>
          </div>
        </section>
      </div>
    </div>
  {:else if currentTab === tabs[1].id}
    <div>Assets</div>
  {/if}
</div>

<style lang="postcss">
  button[disabled] {
    @apply cursor-not-allowed opacity-50;
  }

  .accordion.active {
    @apply bg-gray-100;
  }
</style>
