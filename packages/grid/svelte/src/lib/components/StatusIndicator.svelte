<script lang="ts" context="module">
  const statusMap = {
    active: 'text-green-500 bg-green-500',
    inactive: 'text-gray-500 bg-gray-500'
  };

  export type Status = keyof typeof statusMap;
</script>

<script lang="ts">
  export let status: Status;

  const statusClass = statusMap[status] || statusMap.active;
</script>

<div class="circle {statusClass}">
  <slot />
</div>

<style lang="postcss">
  .circle {
    width: 1px;
    height: 1px;
    border: 5px solid;
    border-radius: 5px;
  }
</style>
