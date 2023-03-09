<script lang="ts">
  import type { HTMLInputAttributes } from 'svelte/elements';

  interface $$Props extends HTMLInputAttributes {
    label: string;
    id: string;
    required?: boolean;
    error?: string;
    warning?: string;
  }

  export let label: string;
  export let id: string;
  export let required = false;
  export let error = '';
  export let warning = '';
</script>

<div class="flex flex-col gap-4 w-full">
  <label class="text-sm pb-0.5 flex gap-1" for={id}>
    <p class="font-semibold">{label}</p>
    {#if required}
      <span class="text-primary-600">*</span>
    {/if}
  </label>
  <input
    {...$$props}
    class="w-full pt-3 bg-gray-50 border border-gray-300/10 placeholder:text-gray-400 text-gray-800 p-3 font-base focus:border-primary-500 caret-gray-200 disabled:opacity-50 truncate rounded-lg"
    {id}
  />
</div>
