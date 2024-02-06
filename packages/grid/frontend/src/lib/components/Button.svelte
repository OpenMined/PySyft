<script lang="ts">
  import type { HTMLButtonAttributes } from "svelte/elements"

  interface $$Props extends HTMLButtonAttributes {
    disabled?: boolean
    variant?: "primary" | "secondary" | "error"
    type?: "button" | "submit" | "reset"
  }

  export let variant = "primary"
  export let disabled = false
  export let type: $$Props["type"] = "button"
</script>

<button {type} class={variant} aria-disabled={disabled} on:click {...$$props}>
  <slot />
</button>

<style lang="postcss">
  button {
    @apply capitalize font-roboto flex flex-nowrap justify-center items-center px-4 py-2 rounded-full font-normal;
    @apply transition-all duration-500;
  }

  [aria-disabled="true"] {
    @apply cursor-not-allowed opacity-50;
  }

  .primary {
    @apply bg-gradient-to-r from-gray-800 to-gray-800 text-white;
    @apply hover:to-gray-500;
  }

  .secondary {
    @apply bg-gradient-to-l from-primary-300 to-primary-300;
    @apply hover:to-marigold-400;
  }

  .error {
    @apply bg-gradient-to-l from-magenta-400 to-magenta-400;
    @apply hover:to-yellow-400;
  }

  .primary-outline {
    @apply border border-primary-300 text-primary-300;
  }
</style>
