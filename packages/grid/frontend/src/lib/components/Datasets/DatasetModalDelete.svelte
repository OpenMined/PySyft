<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import XIcon from '$lib/components/icons/XIcon.svelte';
  import WarningIcon from '../icons/WarningIcon.svelte';
  import ButtonGhost from '../ButtonGhost.svelte';

  export let onClose: () => void;
  export let onDelete: () => void;
</script>

<div class="absolute top-0 left-0 w-screen h-screen bg-gray-800 bg-opacity-50">
  <div class="h-screen w-screen flex items-center justify-center">
    <div class="desktop:w-[562px]">
      <Modal>
        <div class="flex w-full" slot="header">
          <div class="flex flex-col justify-center items-center w-full gap-2 pt-4">
            <div class="w-min h-min rounded-full bg-yellow-500 text-gray-800 p-2">
              <WarningIcon class="w-6 h-6" />
            </div>
            <h3 class="text-2xl capitalize font-regular">Are you sure?</h3>
          </div>
          <button class="self-start" on:click={onClose}>
            <XIcon class="w-6 h-6" />
          </button>
        </div>
        <div class="flex flex-col w-full text-gray-400 gap-4 py-2" slot="body">
          <p>
            If you delete this dataset all dataset assets will become unavailable to your users and
            all pending requests will close. If you want to continue with deleting this dataset
            please confirm by clicking "Delete Dataset" below, otherwise click "Cancel".
          </p>
        </div>
        <div class="flex w-full justify-center px-4 gap-4" slot="button-group">
          <ButtonGhost variant="gray" on:click={onClose}>Cancel</ButtonGhost>
          <Button variant="error" on:click={onDelete}>Delete Dataset</Button>
        </div>
      </Modal>
    </div>
  </div>
</div>
