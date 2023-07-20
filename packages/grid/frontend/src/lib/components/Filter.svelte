<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  export let variant: 'gray' | 'primary-light' | 'primary-dark' = 'gray';
  export let index: number;
  export let filter: number;
  export let filters: number[] = [];

  const onChange = () => {
    index = 0;
    dispatch('setFilter');
  };
</script>

<span class={variant}>
  <button
    type="button"
    title="filter"
    class="flex justify-center items-center"
    style=""
    aria-pressed="false"
    aria-label="Search filters"
  >
    <div class="relative">
      <svg
        enable-background="new 0 0 24 24"
        height="24"
        viewBox="0 0 24 24"
        width="24"
        focusable="false"
        style="fill: currentcolor; pointer-events: none; display: block; width: 24px; height: 24px;;"
      >
        <path
          d="M15 17h6v1h-6v-1zm-4 0H3v1h8v2h1v-5h-1v2zm3-9h1V3h-1v2H3v1h11v2zm4-3v1h3V5h-3zM6 14h1V9H6v2H3v1h3v2zm4-2h11v-1H10v1z"
        />
      </svg>
    </div>
    <slot />
    <div>
      <select class={variant} bind:value={filter} on:change={onChange}>
        {#each filters as filter (filter)}
          <option value={filter}>{filter}</option>
        {/each}
      </select>
    </div>
  </button>
</span>

<style lang="postcss">
  span {
    @apply h-5;
  }
  .gray {
    @apply bg-gray-100 text-gray-800 px-1.5 py-0.5 rounded-sm items-center inline-flex font-bold font-roboto text-xs leading-normal;
  }

  .primary-light {
    @apply bg-primary-100 text-primary-600 px-1.5 py-0.5 rounded-sm items-center inline-flex font-bold font-roboto text-xs leading-normal w-fit;
  }

  .primary-dark {
    @apply bg-gray-800 text-primary-200 px-1.5 py-0.5 rounded-sm items-center inline-flex font-bold font-roboto text-xs leading-normal w-fit;
  }
</style>
