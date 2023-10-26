<script lang="ts">
  export let options: { value: string | number; label: string }[] = []
  export let selected: string | number = ""
  export let name: string
</script>

<div>
  <select
    bind:value={selected}
    {name}
    class="w-full pt-3 bg-gray-50 border placeholder:text-gray-400 text-gray-800 p-3 font-base focus:border-primary-500 caret-gray-200 disabled:opacity-50 truncate rounded-lg border-gray-300/10"
  >
    {#each options as option}
      <option value={option.value}>{option.label}</option>
    {/each}
    asdf
  </select>
</div>
