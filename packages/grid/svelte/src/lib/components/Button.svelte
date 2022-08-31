<script lang="ts">
  export let variant: 'solid-primary' | 'solid-gray' | 'outlined' | 'ghost' | 'link' =
    'solid-primary';
  export let size: 'xs' | 'sm' | 'md' | 'lg' = 'md';
  export let disabled: boolean = false;
</script>

<button class="{variant} {size}" aria-disabled={disabled}>
  <slot />
</button>

<style lang="postcss">
  button {
    @apply font-roboto flex gap-x-2 rounded justify-center items-center capitalize mx-auto;
  }

  [aria-disabled='true'] {
    @apply opacity-40;
  }

  .solid-primary {
    @apply bg-primary-500 text-white hover:bg-gradient-to-r hover:from-white/50 hover:to-white/0;
  }

  .md {
    @apply font-bold leading-normal px-3 py-2;
  }
</style>
