<script>
  import { createEventDispatcher, onMount } from 'svelte';
  import { faCircle, faTableList, faChevronLeft, faTrash } from '@fortawesome/free-solid-svg-icons';
  import Button from '$lib/components/Button.svelte';
  import Fa from 'svelte-fa';
  import DeleteDatasetModal from './deleteDatasetModal.svelte';

  export let name;
  export let author;
  export let description;
  export let requests;
  export let assets;
  export let lastUpdated;
  export let fileSize;
  export let datasetId;

  export let activeTabValue = 'overview';

  let tabItems = [
    { label: 'Overview', value: 'overview' },
    { label: 'Assets', value: 'assets' }
  ];

  const handleTabClick = (tabValue) => () => (activeTabValue = tabValue);
  const dispatch = createEventDispatcher();

  let showModal = false;

  onMount(() => {
    window.scrollTo(0, 0);
  });
</script>

<div>
  <DeleteDatasetModal bind:showModal {datasetId} />

  <!-- Header -->
  <div class="flex justify-between">
    <Button variant="blue-back" action={() => dispatch('closeOpenCard')}
      ><Fa class="pr-2" icon={faChevronLeft} size="xs" />Back</Button
    >

    <!-- svelte-ignore a11y-click-events-have-key-events -->
    <div class="cursor-pointer" on:click={() => (showModal = true)}>
      <Fa class="px-2" icon={faTrash} size="xs" />
    </div>
  </div>

  <div class="detailHeader">
    <ul class="p-2 w-full">
      <div class="flex items-center justify-between">
        <li
          class="flex justify-left text-gray-800 font-rubik text-xl leading-normal font-medium pb-2"
        >
          {name}
        </li>
      </div>
      <div class="flex items-center pb-2">
        <li class="text-gray-600 font-small">Jana Doe</li>
        <Fa class="px-2" icon={faCircle} size="0.3x" />
        <li class="text-gray-600 font-small">{`Updated ${lastUpdated}`}</li>
      </div>
      <div>
        <li>UID: {datasetId}</li>
      </div>
      <div class="flex items-center py-8">
        <Fa class="px-2" icon={faTableList} size="sm" />
        <li class="text-gray-600 font-small">{assets}</li>
        <Fa class="px-2" icon={faCircle} size="0.3x" />
        <li class="text-gray-600 font-small">{`File Size: (${fileSize / 1000}kB)`}</li>
      </div>
    </ul>
  </div>

  <section>
    <ul class="tab-list">
      {#each tabItems as item}
        <li class={`list-item ${activeTabValue === item.value ? 'active' : ''}`}>
          <!-- svelte-ignore missing-declaration -->
          <!-- svelte-ignore a11y-click-events-have-key-events -->
          <span on:click={handleTabClick(item.value)}>{item.label}</span>
        </li>
      {/each}
    </ul>

    <!-- Dataset Detail Body -->
    {#if activeTabValue == 'overview'}
      <div class="py-4 px-32">
        <h2
          class="flex justify-left text-gray-800 font-rubik text-xl leading-normal font-medium pb-4"
        >
          Description
        </h2>
        <p>{description}</p>
      </div>
    {:else}
      <div class="p-2">Assets (Work in progress...)</div>
    {/if}
  </section>
</div>

<style lang="postcss">
  .detailHeader {
    @apply flex p-1 h-fit;
  }

  .tab-list {
    display: flex;
    flex-wrap: wrap;
    padding-left: 0;
    margin-bottom: 0;
    list-style: none;
    border-bottom: 1px solid #dee2e6;
  }

  .list-item {
    margin-bottom: -1px;
  }

  span {
    border: 1px solid transparent;
    border-top-left-radius: 0.25rem;
    border-top-right-radius: 0.25rem;
    display: block;
    padding: 0.5rem 1rem;
    cursor: pointer;
  }

  span:hover {
    border-color: #e9ecef #e9ecef #dee2e6;
  }

  li.active > span {
    color: #495057;
    background-color: #fff;
    border-color: #dee2e6 #dee2e6 #fff;
  }
</style>
