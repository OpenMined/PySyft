<script lang="ts">
  import { navItems } from '$lib/components/Navigation/NavItems.svelte'; // import application navigation

  // let selected = navItems[0].page; // keep track of the selected 'page' object (default to the about component since we must have local db connection established first)
  let selected = ''; // keep track of the selected 'page' object (default to the about component since we must have local db connection established first)

  function changeComponent(page: string) {
    selected = page;
  }
</script>

<nav class="flex flex-col min-w-[250px] h-auto bg-black-900 py-10">
  <p class="text-gray-50">Domain</p>
  <hr class="bg-gray-700" />

  {#each navItems as option}
    <span
      class={selected == option.page
        ? 'flex items-center h-12 px-2 text-gray-50 hover:text-gray-100 bg-gray-800 nav-span'
        : 'flex items-center h-12 px-2 text-gray-200 hover:text-gray-100 bg-black-900 nav-span'}
      on:click={() => changeComponent(option.page)}
      href={option.slug}
    >
      <i class="fa-solid fa-question-circle" />
      {option.page}
      <!-- <a href={option.slug} class="text-gray-200 hover:text-gray-100 no-underline">{option.page}</a> -->
    </span>
  {/each}

  <hr class="bg-gray-700" />
  <p>Profile</p>
</nav>

<style>
  .nav-span {
    cursor: pointer;
  }
</style>
