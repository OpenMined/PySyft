<script>
  import Fa from 'svelte-fa';
  import { createEventDispatcher } from 'svelte';
  import { faCircle, faTableList, faWaveSquare } from '@fortawesome/free-solid-svg-icons';

  export let name;
  export let author;
  export let datasetId;
  export let description;
  export let lastUpdated;
  export let assets;
  export let requests;
  export let fileSize;

  const dispatch = createEventDispatcher();

  function showDatasetDetail(e) {
    e.preventDefault();
    dispatch('hide', {
      openName: name,
      openDatasetId: datasetId,
      openAuthor: author,
      openDescription: description,
      openLastUpdated: lastUpdated,
      openAssets: assets,
      openRequests: requests,
      openFileSize: fileSize
    });
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<div on:click={showDatasetDetail} class="dataListItem">
  <ul class="p-2 w-full">
    <div class="flex items-center justify-between">
      <li
        class="flex justify-left text-gray-800 font-rubik text-xl leading-normal font-medium pb-2"
      >
        {name}
      </li>
      <li class="flex items-center text-gray-600 font-small">
        <Fa class="px-2" icon={faWaveSquare} size="sm" />
        {requests}
      </li>
    </div>
    <div class="flex items-center pb-2">
      <li class="text-gray-600 font-small">Jana Doe</li>
      <Fa class="px-2" icon={faCircle} size="0.3x" />
      <li class="text-gray-600 font-small">{`Updated ${lastUpdated}`}</li>
    </div>
    <div class="flex items-center">
      <Fa class="px-2" icon={faTableList} size="sm" />
      <li class="text-gray-600 font-small">{assets}</li>
      <Fa class="px-2" icon={faCircle} size="0.3x" />
      <li class="text-gray-600 font-small">{`File Size: (${fileSize / 1000}kB)`}</li>
    </div>
  </ul>
</div>

<style lang="postcss">
  .dataListItem {
    @apply flex shadow rounded-sm p-1 h-32;
  }

  .dataListItem:hover {
    @apply cursor-pointer;
  }
</style>
