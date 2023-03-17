<script lang="ts">
  import type { HTMLButtonAttributes } from 'svelte/elements';

  interface $$Props extends HTMLButtonAttributes {
    disabled?: boolean;
    variant: 'secondary' | 'delete' | 'black';
    position?: 'left' | 'right' | 'none';
    action: Function;
  }

  export let variant = 'secondary';
  export let disabled = false;
  export let position = 'none';
  export let action: Function;
</script>

<button
  class="capitalize font-roboto flex flex-no-wrap justify-center items-center px-4 py-3 rounded-[20px] text-sm font-regular disabled:opacity-50 {variant} {position}"
  aria-disabled={disabled}
  on:click={action()}
>
  <slot />
</button>

<style lang="postcss">
  .secondary {
    @apply transition-all duration-500 bg-gradient-to-l from-primary-300 to-primary-300;
  }

  .secondary:hover {
    @apply to-marigold-400;
  }

  .delete {
    @apply transition-all duration-500 bg-gradient-to-l from-magenta-300 to-magenta-300;
  }

  .delete:hover {
    @apply to-marigold-400;
  }

  .black {
    @apply transition-all duration-500 bg-gradient-to-l from-black-900 to-black-900 text-white-50 rounded-3xl h-8;
  }

  .black:hover {
    @apply to-marigold-400;
  }

  .left {
    @apply float-left;
  }

  .right {
    @apply float-right;
  }

  .none {
    @apply float-none;
  }
</style>
