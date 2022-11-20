<script lang="ts">
  import { navigate } from 'svelte-routing';
  import { navItems } from '$lib/components/Navigation/NavItems.svelte';
  import { url } from '$lib/stores/nav';
  import Fa from 'svelte-fa';

  let activeRoute: string;

  url.subscribe((value) => (activeRoute = value));
</script>

<nav class="flex flex-col min-w-[270px] h-auto bg-black-900 py-10">
  <div class="h-1/6">
    <span class="font-roboto text-gray-50">Domain</span>
  </div>

  <hr />

  <div class="h-4/6 py-12">
    {#each navItems as option}
      <span
        class={activeRoute == option.slug
          ? 'flex items-center h-12 px-2 text-gray-50 hover:text-gray-100 bg-gray-800 nav-item-cursor'
          : 'flex items-center h-12 px-2 text-gray-200 hover:text-gray-100 bg-black-900 nav-item-cursor'}
        on:click={() => navigate(option.slug)}
      >
        <div class="w-10 p-2" style="color: #FFFFFF">
          <Fa icon={option.icon} size="sm" />
        </div>
        <span class="font-roboto">{option.title}</span>
      </span>
    {/each}
  </div>

  <hr />

  <div>
    <span class="font-roboto text-gray-50">Profile</span>
  </div>
</nav>

<style>
  .nav-item-cursor {
    cursor: pointer;
  }

  hr {
    border: 1px solid #454158;
  }
</style>
