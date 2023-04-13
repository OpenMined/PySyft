<script>
  import Modal from '$lib/components/DynamicModal.svelte';
  import Button from '$lib/components/Button.svelte';
  import { onMount } from 'svelte';
  import { getClient } from '$lib/store';

  export let showModal;
  export let datasetId;

  let client = '';
  onMount(async () => {
    await getClient().then((response) => {
      client = response;
    });
  });
</script>

<main>
  {#if showModal}
    <Modal size="sm">
      <div slot="header" class="flex justify-center">
        <p class="text-center text-2xl font-bold pt-4">Are You Sure?</p>
      </div>
      <div slot="body" class="h-full">
        <p class="text-center py-4">
          If you delete this dataset all dataset assets will become unavailable to your users and
          all pending requests will close. If you want to continue with deleting this dataset please
          confirm by clicking "Delete Dataset" below, otherwise click "Cancel".
        </p>
      </div>
      <div slot="footer" class="flex justify-center pt-6">
        <Button action={() => (showModal = false)} variant="white">Cancel</Button>
        <Button
          action={() => {
            client.deleteDataset(datasetId);
            showModal = false;
            location.reload();
          }}
          variant="delete">Delete Dataset</Button
        >
      </div>
    </Modal>
  {/if}
</main>
