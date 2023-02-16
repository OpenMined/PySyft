<script lang="ts">
  export let onClick: Function = () => {};
</script>

<svg class="close" on:click={() => onClick()} viewBox="0 0 12 12">
  <line x1="3" y1="3" x2="9" y2="9" />
  <line x1="9" y1="3" x2="3" y2="9" />
</svg>

<style lang="postcss">
  .close {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 24px;
    height: 24px;
    cursor: pointer;
    transition: transform 0.3s;
  }

  .close:hover {
    transform: scale(2);
  }

  .close line {
    stroke: darkgray;
  }
</style>
