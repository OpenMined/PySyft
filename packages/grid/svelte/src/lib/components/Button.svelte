<script lang="ts">
  export let variant: 'solid-primary' | 'solid-gray' | 'outlined' | 'ghost' | 'link' | 'delete' =
    'solid-primary';
  export let size: 'xs' | 'sm' | 'md' | 'lg' = 'md';
  export let disabled: boolean = false;
  export let option: 'sm-pad' | 'none' = 'none';
  export let onClick: Function = () => {};
</script>

<button
  class="{variant} {size} {option}"
  on:click={(e) => {
    e.preventDefault();
    onClick();
  }}
  aria-disabled={disabled}
>
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

  .delete {
    @apply bg-magenta-500 text-white hover:bg-gradient-to-r hover:from-white/50 hover:to-white/0;
  }

  .md {
    @apply font-bold leading-normal px-3 py-2;
  }

  .sm-pad {
    @apply mx-1 !important;
  }

  .none {
  }
</style>
