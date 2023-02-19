<script lang="ts">
  export let variant:
    | 'solid-primary'
    | 'solid-gray'
    | 'outlined'
    | 'outlined-primary'
    | 'ghost'
    | 'link'
    | 'delete' = 'solid-primary';
  export let size: 'xs' | 'sm' | 'md' | 'lg' = 'md';
  export let disabled: boolean = false;
  export let option: 'right' | 'left' | 'margin-x-sm' | 'none' = 'none';
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
    @apply bg-primary-500 text-white;
  }

  .solid-primary:hover {
    @apply bg-gradient-to-r from-white/50 to-white/0;
  }

  .outlined-primary {
    @apply text-primary-500 border-solid border-2 border-primary-500 rounded;
  }

  .delete {
    @apply bg-magenta-500 text-white;
  }

  .delete:hover {
    @apply bg-gradient-to-r from-white/50 to-white/0;
  }

  .md {
    @apply font-bold leading-normal px-3 py-2;
  }

  .margin-x-sm {
    @apply mx-1;
  }

  .left {
    @apply ml-0;
  }

  .right {
    @apply mr-0;
  }

  .none {
  }
</style>
