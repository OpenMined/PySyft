<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import Capital from '$lib/components/Capital.svelte';
  import { defaultItems } from '$lib/checklist.json';

  let newItem = '';

  let checklistItems = defaultItems.map((item) => {
    return { text: item, status: false };
  });

  function addToList() {
    checklistItems = [...checklistItems, { text: newItem, status: false }];
    newItem = '';
  }

  function removeFromList(index: number) {
    checklistItems.splice(index, 1);
    checklistItems = checklistItems;
  }
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col h-full">
  <!-- Body content -->
  <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14 h-full">
    <div class="w-full">
      <div class="py-4 mt-2">
        <p class="text-xs">No. 001</p>
        <!-- Study name -->
        <h1 class="text-2xl leading-[1.1] font-medium text-gray-800 font-rubik mt-1">
          Is Recurrent Depression Mediated By Neuroinflammation?
        </h1>
      </div>

      <hr class="py-3" />

      <!-- Checklist info -->
      <h1 class="py-5 underline decoration-dotted underline-offset-8">General Checklist</h1>
      <!-- {#each Object.values(defaultItems) as item}
          <li>{item}</li>
          <input bind:checked={item.status} type="checkbox" />
          <span class:checked={item.status}>{item.text}</span>
          <span on:click={() => removeFromList(index)}>❌</span>
          <br />
        {/each} -->
      <ul class="space-y-0.25">
        {#each checklistItems as item, index}
          <li>
            <input bind:checked={item.status} type="checkbox" />
            <span class:checked={item.status}>{item.text}</span>
            <span on:click={() => removeFromList(index)}>new item</span>
          </li>
          <br />
        {/each}

        <input bind:value={newItem} type="text" placeholder="new todo item.." />
        <button on:click={addToList}>Add</button>
      </ul>
    </div>

    <Capital>
      <div slot="header">
        <h2
          class="flex justify-center text-gray-800 font-rubik text-2xl leading-normal font-medium"
        >
          Study Metadata
        </h2>
      </div>
    </Capital>
  </section>
  <!-- Footer -->
  <span>
    <img src="/images/empowered-by-openmined.png" alt="Empowered by OpenMined logo" />
  </span>
</main>

<style>
  .checked {
    text-decoration: line-through;
  }
</style>
