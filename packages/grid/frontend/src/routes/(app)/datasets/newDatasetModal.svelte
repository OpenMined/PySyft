<script>
  import Modal from '$lib/components/DynamicModal.svelte';
  import Button from '$lib/components/Button.svelte';

  export let showModal;
</script>

<main>
  {#if showModal}
    <Modal size="md">
      <div slot="header" class="flex justify-center">
        <p class="text-center text-2xl font-bold pt-4">Upload A Dataset</p>
      </div>
      <div slot="body" class="h-full">
        <p class="text-center py-4">
          Begin uploading data by following the steps below so that your team and outside
          researchers can begin making impact.
        </p>
        <ul class="flex flex-col">
          <li class="pb-2">Open a new Jupyter Notebook page</li>
          <li class="pb-2">Install HAGrid by running the code below in your Jupyter Notebook</li>
          <code class="flex items-center px-4 h-14">pip install -U hagrid</code>
          <li class="pb-2">
            Once HAGrid is installed open the "Upload Dataset" quickstart tutorial notebook by
            running the code below in your Jupyter Notebook.
          </li>
          <code class="flex items-center px-4 h-14">hagrid quickstart dataset_upload</code>
        </ul>
      </div>
      <div slot="footer" class="flex justify-center self-end pt-6">
        <Button action={() => (showModal = false)} variant="black" position="right">Finish</Button>
      </div>
    </Modal>
  {/if}
</main>

<style lang="postcss">
  ul {
    @apply list-disc mx-7;
  }

  code {
    @apply bg-gray-50;
  }
</style>
