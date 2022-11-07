<script>
  import { createEventDispatcher, onDestroy } from 'svelte';

  const dispatch = createEventDispatcher();
  const close = () => dispatch('close');

  let modal;

  const handle_keydown = (/** @type {{ key: string; }} */ e) => {
    if (e.key === 'Escape') {
      close();
      return;
    }
  };
</script>

<svelte:window on:keydown={handle_keydown} />

<div class="topModal visible" on:click={() => close()}>
  <div class="modal" on:click|stopPropagation={() => {}}>
    <svg class="close" on:click={() => close()} viewBox="0 0 12 12">
      <line x1="3" y1="3" x2="9" y2="9" />
      <line x1="9" y1="3" x2="3" y2="9" />
    </svg>
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

  .close {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 24px;
    height: 24px;
    cursor: pointer;
    transition: transform 0.3s;
  }

  .close:hover {
    transform: scale(2);
  }

  .close line {
    stroke: darkgray;
  }

  .modal-content {
    /* max-width: calc(100vw - 40px); */
    /* max-width: 25vw; */
    max-height: calc(100vh - 20px);
    overflow: auto;
  }
</style>
