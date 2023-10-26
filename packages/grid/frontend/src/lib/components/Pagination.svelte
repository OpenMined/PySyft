<script lang="ts">
  export let total: number
  export let page_size: number
  export let page_index: number

  $: totalPages = Math.ceil(total / page_size) || 0

  const handlePrev = () => page_index > 0 && page_index--
  const handleNext = () => page_index < totalPages - 1 && page_index++
</script>

<nav
  class="flex justify-between items-center gap-6 bg-gray-50 px-4 py-2 rounded"
  aria-label="page selector"
>
  <button
    class="prev"
    on:click={handlePrev}
    disabled={page_index === 0}
    aria-label="last page"
  >
    ←
  </button>
  <span>Page {page_index + 1} of {totalPages}</span>
  <button
    class="next"
    on:click={handleNext}
    disabled={page_index === totalPages - 1}
    aria-label="next page"
  >
    →
  </button>
</nav>

<style>
  button {
    background: none;
    border: none;
    font-size: 1.2rem;
    cursor: pointer;
  }

  button:disabled {
    color: #ccc;
    cursor: not-allowed;
  }
</style>
