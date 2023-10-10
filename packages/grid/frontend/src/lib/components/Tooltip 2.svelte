<script lang="ts">
  export let active = false;

  let variant: 'text' | 'profile' = 'text';
  let el: HTMLElement;
  let tooltip: HTMLElement;
  let innerWidth: number;
  let innerHeight: number;
  let scrollX: number;
  let scrollY: number;

  function handleResize() {}

  function handleMouseEnter() {
    active = true;
  }

  function handleMouseLeave() {
    active = false;
  }

  function left() {
    const elLeft = el.getBoundingClientRect().x + scrollX;
    const tooltipLeft = elLeft + el.offsetWidth / 2 - tooltip.offsetWidth / 2;
    const adjustedLeft = adjustXOverflow(tooltipLeft, tooltip.offsetWidth);
    return `${adjustedLeft}px`;
  }

  function adjustXOverflow(currentLeft: number, width: number) {
    const xOverflow = currentLeft + width - innerWidth + 16;
    let newLeft = currentLeft;

    if (xOverflow > 0) {
      newLeft = Math.max(currentLeft - xOverflow, 0);
    } else {
      newLeft = Math.max(currentLeft, 16);
    }

    return newLeft + scrollX;
  }

  const updateTooltipPosition = () => {
    tooltip.style.left = left();
  };

  const handleUpdate = () => ({
    update: () => {
      if (active) {
        updateTooltipPosition();
      }
    }
  });
</script>

<svelte:window
  bind:innerWidth
  bind:innerHeight
  bind:scrollX
  bind:scrollY
  on:resize={handleResize}
/>
<div on:mouseenter={handleMouseEnter} on:mouseleave={handleMouseLeave} bind:this={el}>
  <slot />
</div>
<span class="tooltip" class:active bind:this={tooltip} use:handleUpdate>
  <slot name="tooltip" />
</span>

<style lang="postcss">
  .tooltip {
    @apply absolute inline-block opacity-0 shadow-tooltip-1 flex flex-nowrap flex-col gap-1 rounded transition duration-200 ease-in-out;
  }

  .tooltip.active {
    @apply opacity-100;
  }
</style>
