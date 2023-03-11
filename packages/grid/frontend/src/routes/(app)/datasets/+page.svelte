<script>
  // import { Modal } from 'flowbite-svelte';
  import { browser } from '$app/environment';
  import { getPath } from '../../../lib/utils.js';

  import Modal from '$lib/components/NewModal.svelte';
  import Button from '$lib/components/Button.svelte';
  import Datasets from './datasetList.svelte';
  import DatasetItem from './datasetItem.svelte';

  let modalFlag = true;

  let path = '';
  if (browser) {
    path = getPath();
  }

  function closeModal() {
    console.log(`modalFlag: ${modalFlag}`);
    modalFlag = false;
  }

  const items = [{ value: 'I am red!', component: DatasetItem }];
</script>

<main>
  <!-- <LargeModal bind:open={modalFlag} placement="center" size="md" class="w-full"> -->
  <!-- <Modal bind:open={modalFlag} placement="center" size="md" class="w-full"> -->
  <Modal bind:open={modalFlag} placement="center" size="md" class="w-full">
    <div class="flex flex-shrink-0 justify-between p-4 pb-0 flex-nowrap w-full h-min" slot="header">
      <!-- <span class="block text-center w-full">
        <p class="text-2xl font-bold text-gray-800">Apply for an account</p>
      </span> -->
      <h1 style="font-size: 28px; font-weight: bold; color:black;">Get Started!</h1>
      <h2 style="font-size: 15px;">
        Begin using your domain node with one of the key actions below.
      </h2>
    </div>

    <!-- <div class="space-y-4"> -->
    <div style="display:grid; grid-auto-rows: 40vh; grid-auto-columns: 25vh;" slot="body">
      <div
        style="cursor: pointer;grid-column: span 1; grid-row: 1 / span 1;margin-right: 3vh; border-width: 1px; border-color: #DDDDD1;	box-shadow: 2px 2px 9px #888888;"
      >
        <img
          style="height: 20vh; width: 26vh;"
          alt="upload-dataset.jpg"
          src="/assets/card_image.jpg"
        />
        <h1 style="margin-left: 7px;margin-top: 7px;color: black; font-size: 10; font-weight: bold">
          Upload a Dataset
        </h1>
        <h2
          style="font-size: 15px;margin-left: 7px;margin-right: 7px;margin-top: 3px;margin-bottom: 5px;"
        >
          Create a catalogue and begin uploading data so that your team and outside researchers can
          begin making impact.
        </h2>
      </div>
      <div
        style="cursor: pointer;grid-column: 2 / span 1; grid-row: 1 / span 1;margin-right: 3vh;border-width: 1px; border-color: #DDDDDD;box-shadow: 2px 2px 9px #888888;"
      >
        <img
          style="height: 20vh; width: 26vh;"
          alt="create-team-member.jpg"
          src="/assets/card_image.jpg"
        />
        <h1 style="margin-left: 7px;margin-top: 7px;color: black; font-size: 10; font-weight: bold">
          Create Team Member
        </h1>
        <h2
          style="font-size: 15px;margin-left: 7px;margin-right: 7px;margin-top: 3px;margin-bottom: 5px;"
        >
          Create accounts for team members so that they can help research, maintain, and contribute
          to your domain node.
        </h2>
      </div>
      <div
        style="cursor: pointer;grid-column: 3 / span 1; grid-row: 1 / span 1;margin-right: 3vh;border-width: 1px; border-color: #DDDDDD; box-shadow: 2px 2px 9px #888888;"
      >
        <img
          style="height: 20vh; width: 26vh;"
          alt="create-user.jpg"
          src="/assets/card_image.jpg"
        />
        <h1 style="margin-left: 7px;margin-top: 7px;color: black; font-size: 10; font-weight: bold">
          Create General User
        </h1>
        <h2
          style="font-size: 15px;margin-left: 7px;margin-right: 7px;margin-top: 3px;margin-bottom: 5px;"
        >
          Create accounts for outside researchers to safely study from your data without needing a
          copy or direct access.
        </h2>
      </div>
    </div>
    <h2 style="font-size: 15px;">
      Learn more about how to use your domain node by reading our "Data Owner" handbook.
    </h2>
    <!-- <br /><br /> -->
    <!-- <div style="display:flex; justify-content:right"><h1>Cancel</h1></div> -->
    <Button on:click={() => closeModal()} variant="secondary" slot="button-group">Cancel</Button>
    <!-- </div> -->
  </Modal>
  <div class="page-container">{path}</div>

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
