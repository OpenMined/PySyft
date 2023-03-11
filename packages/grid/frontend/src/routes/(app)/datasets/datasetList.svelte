<script>
  export let items;
</script>

{#each items as item}
  <svelte:component this={item.component} {...item} />
{/each}
