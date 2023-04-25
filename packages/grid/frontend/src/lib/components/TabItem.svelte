<script lang="ts">
  import type { SvelteComponent } from 'svelte';

  export let icon: SvelteComponent | undefined;
  export let label: string;
  export let active: string;
  export let count: number | undefined;
  export let id: string;
  export let ariaControls: string;

  const iconClass = { class: 'w-4 h-4' };
  let isActive = active === id;

  $: isActive = active === id;
</script>

<button
  class="min-w-[153px] px-4 py-1 text-gray-300 hover:text-gray-800 transition duration-500 border-b border-gray-100"
  on:click
  role="tab"
  tabindex={isActive ? 0 : -1}
  aria-selected={isActive}
  aria-controls={ariaControls}
  {id}
>
  <div class="flex gap-1.5 items-center justify-center">
    {#if icon}
      <svelte:component this={icon} {...iconClass} />
    {/if}
    <span>{label}</span>
    {#if count || count === 0}
      <span class="text-sm">({count})</span>
    {/if}
  </div>
</button>

<style lang="postcss">
  [aria-selected='true'] {
    @apply text-gray-800 border-primary-500;
  }
</style>
