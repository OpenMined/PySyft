<script lang="ts">
  import type { HTMLButtonAttributes } from 'svelte/elements';

  interface $$Props extends HTMLButtonAttributes {
    disabled?: boolean;
    variant: 'gray';
  }

  export let variant = 'gray';
  export let disabled = false;
</script>

<button class={variant} aria-disabled={disabled} on:click {...$$props}>
  <slot />
</button>

<style lang="postcss">
  button {
    @apply capitalize font-roboto flex flex-nowrap justify-center items-center px-4 py-2 rounded-full font-normal;
    @apply transition-all duration-500;
  }

  [aria-disabled='true'] {
    @apply cursor-not-allowed opacity-50;
  }

  .gray {
    @apply text-gray-400;
    @apply hover:bg-pressedGray;
  }
</style>
