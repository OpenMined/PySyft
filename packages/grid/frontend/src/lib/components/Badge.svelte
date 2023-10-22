<script lang="ts">
  export let variant: 'gray' | 'primary-light' | 'primary-dark' = 'gray';
</script>

<span class={variant} data-testid="badge">
  <slot />
</span>

<style lang="postcss">
  span {
    @apply h-5;
  }
  .gray {
    @apply bg-gray-100 text-gray-800 px-1.5 py-0.5 rounded-sm items-center inline-flex font-bold font-roboto text-xs leading-normal;
  }

  .primary-light {
    @apply bg-primary-100 text-primary-600 px-1.5 py-0.5 rounded-sm items-center inline-flex font-bold font-roboto text-xs leading-normal w-fit;
  }

  .primary-dark {
    @apply bg-gray-800 text-primary-200 px-1.5 py-0.5 rounded-sm items-center inline-flex font-bold font-roboto text-xs leading-normal w-fit;
  }
</style>
