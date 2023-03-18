<script>
  import { createEventDispatcher } from 'svelte';

  export let name;
  export let author;
  export let lastUpdated;
  export let fileSize;

  const dispatch = createEventDispatcher();

  function showDatasetDetail(e) {
    e.preventDefault();
    dispatch('hide', {
      openName: name,
      openAuthor: author,
      openLastUpdated: lastUpdated,
      openFileSize: fileSize
    });
  }

  function handlefav() {
    console.log(poi);
  }
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<div on:click={showDatasetDetail} class="dataListItem">
  <ul class="p-2">
    <li>{author}</li>
    <li>{name}</li>
    <li>{lastUpdated}</li>
    <li>{fileSize}</li>
  </ul>
</div>

<style>
  .dataListItem {
    display: flex;
    box-shadow: -2px 4px 8px rgba(13, 12, 17, 0.25);
    border-radius: 15px;
    padding: 1vh;
  }

  .dataListItem:hover {
    cursor: pointer;
  }
</style>
