<script>
  import { browser } from '$app/environment';
  import { getPath } from '../../../lib/utils.js';

  import GetStartedModal from './getStartedModal.svelte';
  import Datasets from './datasetList.svelte';
  import DatasetItem from './datasetItem.svelte';

  let modalFlag = true;

  let path = '';
  if (browser) {
    path = getPath();
  }

  const items = [{ value: 'I am red!', component: DatasetItem }];
</script>

<main>
  <div class="page-container">{path}</div>

  <GetStartedModal showModal={modalFlag} />

  <h1>Datasets</h1>
  <Datasets {items} />
</main>

<style>
  .page-container {
    width: 85%;
    position: absolute;
    height: 93%;
    top: 7%;
    left: 15%;
    display: flex;
    justify-content: center;
    align-items: center;
  }
</style>
