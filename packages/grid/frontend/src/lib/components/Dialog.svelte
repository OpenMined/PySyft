<script lang="ts">
  export let open = false;

  let dialog: HTMLDialogElement;

  $: if (dialog && open) dialog.showModal();
  $: if (dialog && !open) dialog.close();
</script>

<dialog bind:this={dialog} on:close={() => (open = false)} on:click|self={() => dialog.close()}>
  <slot />
</dialog>

<style lang="postcss">
  dialog {
    @apply bg-transparent border-none;
  }

  dialog::backdrop {
    @apply bg-gray-800 bg-opacity-50;
  }

  dialog[open] {
    @apply transition duration-500 ease-in-out;
  }

  dialog[open]::background {
    @apply transition duration-500 ease-in-out;
  }
</style>
