<script lang="ts">
  import type { HTMLInputAttributes } from 'svelte/elements';
  import MagnifyingGlass from './icons/MagnifyingGlass.svelte';

  interface $$Props extends HTMLInputAttributes {
    error?: string;
  }

  export let value: HTMLInputAttributes['value'];
  export const error = '';
</script>

<div class="w-full relative">
  <input
    {...$$props}
    on:input
    bind:value
    class="w-full pt-3 bg-gray-50 border border-gray-300/10 placeholder:text-gray-400 text-gray-800 p-3 pr-8 font-base focus:border-primary-500 caret-gray-200 disabled:opacity-50 truncate rounded-lg"
  />
  <MagnifyingGlass class="w-5 h-5 absolute text-gray-800 top-0 bottom-0 right-3 my-auto" />
</div>
