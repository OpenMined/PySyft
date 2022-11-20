<script lang="ts">
  export let variant: 'gray' | 'blue' = 'gray';
</script>

<span class={variant}>
  <slot />
</span>

<style lang="postcss">
  .gray {
    @apply bg-gray-100
    text-gray-800
    px-1.5
    py-0.5
    rounded-sm
    items-center
    inline-flex
    font-bold
    font-roboto
    text-xs
    leading-normal;
  }

  .blue {
    @apply bg-gray-800
    text-primary-200
    px-1.5
    py-0.5
    rounded-sm
    items-center
    inline-flex
    font-bold
    font-roboto
    text-xs
    leading-normal
    w-fit;
  }
</style>
