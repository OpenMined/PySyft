<script lang="ts">
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  export let variant: 'gray' | 'primary-light' | 'primary-dark' = 'gray';
  export let total: number;
  export let page_size: number;
  export let page_index: number;
  export let page_row: number;

  const pageNumbers = (total: number, max: number, size: number, current: number) => {
    const half = Math.floor(max / 2);
    let to = max;

    if (current + half >= total) {
      to = total;
    } else if (current > half) {
      to = current + half;
    }

    let from = Math.max(to - max, 0);

    return Array.from({ length: Math.min(Math.ceil(total / size), max) }, (_, i) => i + 1 + from);
  };

  $: paginators = Math.ceil(total / page_size) || 0;
  $: paginations = pageNumbers(total, page_row, page_size, page_index);

  const handlePaginate = (index: number) => {
    page_index = index;
    dispatch('setPagination', page_index);
  };

  const handlePrev = () => {
    if (page_index - 1 < 0) return;
    else dispatch('setPagination', page_index--);
  };

  const handleNext = () => {
    if (page_index + 1 >= paginators) return;
    else dispatch('setPagination', page_index++);
  };
</script>

<span class="flex gap-2.5">
  <button
    type="button"
    title="Previous"
    class={`${variant} pagination-button`}
    style=""
    style:cursor={page_index === 0 ? 'not-allowed' : 'pointer'}
    aria-pressed="false"
    aria-label="LEFT-POINTING ANGLE"
    disabled={page_index === 0}
    on:click={handlePrev}
  >
    &#10094;
  </button>
  {#each paginations as pagination}
    <button
      type="button"
      title={`Page ${pagination}`}
      class={`${variant} pagination-button ${pagination - 1 === page_index ? 'primary-light' : ''}`}
      style=""
      aria-pressed="false"
      aria-label="Paginate"
      on:click={() => {
        handlePaginate(pagination - 1);
      }}
    >
      {pagination}
    </button>
  {/each}
  <button
    type="button"
    title="Next"
    class={`${variant} pagination-button`}
    style=""
    style:cursor={page_index + 1 === paginators ? 'not-allowed' : 'pointer'}
    aria-pressed="false"
    aria-label="RIGHT-POINTING ANGLE"
    disabled={page_index + 1 === paginators}
    on:click={handleNext}
  >
    &#10095;
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

  .pagination-button {
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 0.75rem;
    padding-bottom: 0.75rem;
    font-size: 1rem;
    line-height: 1.25rem;
    font-weight: 800;
  }
</style>
