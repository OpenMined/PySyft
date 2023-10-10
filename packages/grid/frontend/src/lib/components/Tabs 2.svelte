<script lang="ts">
  import TabItem from './TabItem.svelte';
  import type { SvelteComponent } from 'svelte';

  interface Tabs {
    id: string;
    label: string;
    icon?: SvelteComponent;
    count?: number;
  }

  export let tabs: Tabs[] = [];
  export let active: string = tabs[0]?.id;

  function activateTab(id: string) {
    active = id;
  }
</script>

<div class="w-full h-min border-b border-gray-100 flex flex-nowrap" tabindex="0">
  {#each tabs as { id, count, icon, label }}
    <TabItem
      {id}
      {icon}
      {count}
      {label}
      {active}
      on:click={() => activateTab(id)}
      ariaControls="{id}-panel"
    >
      {label}
    </TabItem>
  {/each}
</div>
