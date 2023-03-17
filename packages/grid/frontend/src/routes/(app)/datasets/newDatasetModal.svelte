<script>
  import Modal from '$lib/components/DynamicModal.svelte';
  import Button from '$lib/components/Button.svelte';

  export let showModal;
</script>

<main>
  {#if showModal}
    <Modal size="large">
      <div slot="header" class="flex justify-center">
        <GreenCheck />
        <p class="text-center text-2xl font-bold">Your account has been deleted</p>
      </div>
      <div slot="body">
        <p class="text-center">
          To help us improve future experiences could you share with us any frustrations or
          suggestions you have with or for the PyGridUI Platform?
        </p>

        <form class="flex-shrink-0 pt-6">
          <div class="flex flex-col gap-y-4 gap-x-6">
            <FormControl
              label="Frustrations"
              id="frustrations"
              type="textarea"
              placeholder="What felt vague or cumbersome?"
              optional
            />
            <FormControl
              label="Suggestions"
              id="suggestions"
              type="textarea"
              placeholder="Did you have moments of thinking “I wish I could...”"
              optional
            />
          </div>
        </form>
      </div>
      <div slot="footer" class="flex justify-center pt-6">
        <Button
          variant="delete"
          action={() => {
            showDeleteConfirmModal = false;
          }}>Submit Response</Button
        >
        <a class="flex items-center no-underline pl-8 font-bold" href="/">Cancel</a>
      </div>
    </Modal>
  {/if}
</main>
