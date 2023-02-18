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

<!-- svelte-ignore a11y-click-events-have-key-events -->
<div class="topModal visible" on:click={() => close()}>
  <div class="modal" on:click|stopPropagation={() => {}}>
    <Close onClick={() => close()} />
    <div class="modal-content flex flex-col">
      <slot name="top-actions" />
      <slot name="header" />
      <slot name="content" />
      <slot name="bottom-actions" />
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
    padding: 1em;
    width: 50vw;
    height: 80vh;
    background: #ffffff;
    box-shadow: -2px 4px 8px rgba(13, 12, 17, 0.25);
    border-radius: 15px;
  }

  .visible {
    visibility: visible !important;
  }

  .modal-content {
    max-height: calc(100vh - 20px);
    overflow: auto;
  }
</style>
