<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import Modal from '$lib/components/Modal.svelte';
  import TableFillIcon from '$lib/components/icons/TableFillIcon.svelte';
  import XIcon from '$lib/components/icons/XIcon.svelte';

  export let onClose: () => void;
</script>

<div class="absolute top-0 left-0 w-screen h-screen bg-gray-800 bg-opacity-50">
  <div class="h-screen w-screen flex items-center justify-center">
    <div class="desktop:w-[662px]">
      <Modal>
        <div class="flex w-full" slot="header">
          <div class="flex flex-col justify-center items-center w-full gap-2 pt-2">
            <div class="w-min h-min rounded-full bg-primary-500 text-gray-800 p-2">
              <TableFillIcon class="w-6 h-6" />
            </div>
            <h3 class="text-2xl capitalize font-regular">Upload a dataset</h3>
          </div>
          <button class="self-start" on:click={onClose}>
            <XIcon class="w-6 h-6" />
          </button>
        </div>
        <div class="flex flex-col w-full text-gray-400 gap-4" slot="body">
          <p class="text-center">
            Begin uploading data by following the steps below so that your team and outside
            researchers can begin making impact.
          </p>
          <ul class="w-full space-y-4">
            <li class="flex gap-3">
              <div
                class="rounded-full bg-gray-100 text-gray-800 w-6 h-6 flex items-center justify-center text-xs flex-shrink-0"
              >
                1
              </div>
              <p>Open a new Jupyter Notebook page</p>
            </li>
            <li class="flex flex-col gap-3">
              <div class="flex gap-3">
                <div
                  class="rounded-full bg-gray-100 text-gray-800 w-6 h-6 flex items-center justify-center text-xs flex-shrink-0"
                >
                  2
                </div>
                <p></p>
              </div>
              <div class="rounded bg-gray-100 text-gray-800 font-fira p-3 mx-8 mb-2">
                <p></p>
              </div>
            </li>
            <li class="flex flex-col gap-3">
              <div class="flex gap-3">
                <div
                  class="rounded-full bg-gray-100 text-gray-800 w-6 h-6 flex items-center justify-center text-xs flex-shrink-0"
                >
                  3
                </div>
                <p>
                </p>
              </div>
              <div class="rounded bg-gray-100 text-gray-800 font-fira p-3 mx-8 mb-2">
                <p></p>
              </div>
            </li>
          </ul>
        </div>
        <div class="flex w-full justify-end px-4" slot="button-group">
          <Button variant="primary" on:click={onClose}>Finish</Button>
        </div>
      </Modal>
    </div>
  </div>
</div>
