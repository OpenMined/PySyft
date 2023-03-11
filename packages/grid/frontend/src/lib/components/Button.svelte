<script lang="ts">
  import type { HTMLButtonAttributes } from 'svelte/elements';

  interface $$Props extends HTMLButtonAttributes {
    disabled?: boolean;
    variant: 'secondary';
  }

  export let variant: string;
  export let disabled = false;
</script>

<button
  class="capitalize font-roboto flex flex-no-wrap justify-center items-center px-4 py-3 rounded-[20px] text-sm font-regular disabled:opacity-50 {variant}"
  aria-disabled={disabled}
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
</style>
