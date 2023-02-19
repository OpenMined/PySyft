<script lang="ts">
  export let status: 'active' | 'inactive' = 'active';
</script>

<span class="circle {status}">
  <slot />
</span>

<style lang="postcss">
  .circle {
    @apply w-px h-px border-4 border-solid rounded;
  }

  .active {
    @apply border-green-500;
  }

  .inactive {
    @apply border-gray-500;
  }
</style>
