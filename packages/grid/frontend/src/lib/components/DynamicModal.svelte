<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  interface $$Props {
    size?: 'sm' | 'md' | 'lg';
  }

  export let size = 'sm';

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
  <div
    class={`modal ${size}`}
    on:click|stopPropagation={() => {
      // do nothing
    }}
  >
    <div class="modal-content flex flex-col">
      <slot name="header" />
      <slot name="body" />
      <slot name="footer" />
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
    background: #ffffff;
    box-shadow: -2px 4px 8px rgba(13, 12, 17, 0.25);
    border-radius: 15px;
  }

  .visible {
    visibility: visible !important;
  }

  .modal-content {
    height: -webkit-fill-available;
    overflow: auto;
    justify-content: space-between;
  }

  .sm {
    width: 60vw;
    height: 40vh;
  }

  .md {
    width: 60vw;
    height: 70vh;
  }

  .lg {
    max-width: 75vw;
    height: 80vh;
  }
</style>
