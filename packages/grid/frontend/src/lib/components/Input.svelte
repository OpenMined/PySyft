<script lang="ts">
  import type { HTMLInputElement } from "svelte/elements"

  export let label: string
  export let id: string
  export let required = false
  export let value: HTMLInputElement["value"] = ""
  export let error: boolean | undefined = undefined
  export const warning = ""
  $: borderColor = !error ? "border-gray-300/10" : "border-rose-500/100"
</script>

<div class="flex flex-col gap-4 w-full">
  <label class="text-sm pb-0.5 flex gap-1" for={id}>
    <p class="font-semibold">{label}</p>
    {#if required}
      <span class="text-primary-600">*</span>
    {/if}
  </label>
  <input
    {...$$restProps}
    {required}
    {id}
    bind:value
    class="w-full pt-3 bg-gray-50 border {borderColor} placeholder:text-gray-400 text-gray-800 p-3 font-base focus:border-primary-500 caret-gray-200 disabled:opacity-50 truncate rounded-lg"
  />
</div>
