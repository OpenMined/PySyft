<script lang="ts">
  export let onClick: () => void = () => {
    // commenting this to bypass es6 lint
  };
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<svg class="close" on:click={() => onClick()} viewBox="0 0 12 12">
  <line x1="3" y1="3" x2="9" y2="9" />
  <line x1="9" y1="3" x2="3" y2="9" />
</svg>

<style lang="postcss">
  .close {
    @apply absolute top-2 right-2 w-6 h-6 cursor-pointer transition-transform duration-300;
  }

  .close:hover {
    @apply scale-150;
  }
  .close line {
    @apply stroke-gray-300;
  }
</style>
