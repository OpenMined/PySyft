<script lang="ts">
  export let max: number;
  export let value: number;
</script>

<progress class="w-full appearance-none" {max} {value} />

<style lang="postcss">
  progress[value]::-moz-progress-bar {
    @apply bg-gray-50 h-2;
  }

  progress[value]::-webkit-progress-bar {
    @apply bg-gray-50 h-2;
  }

  progress[value]::-webkit-progress-value {
    @apply bg-primary-500;
  }
</style>
