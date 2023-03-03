<script>
  import { createEventDispatcher } from 'svelte';
  import Close from '$lib/components/icons/Close.svelte';

  const dispatch = createEventDispatcher();
  const close = () => dispatch('close');

  const handle_keydown = (/** @type {{ key: string; }} */ e) => {
    if (e.key === 'Escape') {
      close();
      return;
    }
  };
</script>

<svelte:window on:keydown={handle_keydown} />

<div class="topModal visible" on:click={() => close()}>
  <div
    class="modal"
    on:click|stopPropagation={() => {
      // commenting here to bypass es6 lint
    }}
  >
    <Close onClick={() => close()} />
    <div class="modal-content flex flex-col space-y-3 mx-4">
      <slot name="icon" />
      <slot name="header" />
      <slot name="content" />
      <slot name="actions" />
    </div>
  </div>
</div>

<style lang="postcss">
  .topModal {
    visibility: hidden;
    z-index: 9999;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: #4448;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .modal {
    position: relative;
    border-radius: 2px;
    background: white;
    filter: drop-shadow(5px 5px 5px #555);
    padding: 1em;
    width: 30vw;
  }

  .visible {
    visibility: visible !important;
  }

  .modal-content {
    max-height: calc(100vh - 20px);
    overflow: auto;
  }
</style>
