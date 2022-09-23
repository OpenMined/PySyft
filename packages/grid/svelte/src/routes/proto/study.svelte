<script lang="ts">
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

  function saveEntries() {
    alert('TODO: Save checklist items');
  }
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col h-full">
  <!-- Body content -->
  <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] my-5 h-full">
    <div class="w-full">
      <div class="py-4 mt-2">
        <!-- Study header -->
        <p class="text-xs">No. 001</p>
        <h1 class="text-2xl leading-[1.1] font-medium text-gray-800 font-rubik mt-1 mb-2">
          Is Recurrent Depression Mediated By Neuroinflammation?
        </h1>
        <div class="flex items-center">
          <div class="avatarImage">
            <slot />
          </div>
          <p class="text-xs pl-2">
            <strong>University of Oxford</strong>, Department of Psychiatry
          </p>
        </div>
      </div>

      <hr />

      <!-- Checklist components -->
      <h1 class="py-5 underline decoration-dotted underline-offset-8">General Checklist</h1>

      {#each checklistItems as item, index}
        <div class="flex items-center py-2">
          <input class="cursor-pointer large-checkbox" bind:checked={item.status} type="checkbox" />
          <label class="px-3" class:checked={item.status}>{item.text}</label>
          <span class="cursor-pointer" on:click={() => removeFromList(index)}>❌</span>
        </div>
      {/each}

      <div class="px-5 py-5">
        <button class="pr-2" on:click={addToList}>+ Add Item</button>
        <input
          class="border-solid border-2"
          bind:value={newItem}
          type="text"
          placeholder="new item.."
        />
      </div>

      <button class="submit-button" on:click={saveEntries}>Make A Decision</button>
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
    color: rgb(148 163 184);
  }

  input[type='checkbox'] {
    -ms-transform: scale(2);
    -moz-transform: scale(2);
    -webkit-transform: scale(2);
    -o-transform: scale(2);
    transform: scale(2);
  }

  .submit-button {
    color: white;
    background-color: black; /* Green */
    border-radius: 20px;
    font-size: 15px;
    height: 40px;
    width: 150px;
    padding: 10px;
  }

  .avatarImage {
    width: 40px;
    height: 40px;
    border: 20px solid;
    border-radius: 50%;
  }
</style>
