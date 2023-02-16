<script lang="ts">
  import Badge from '$lib/components/Badge.svelte';
  import StatusIndicator from '$lib/components/StatusIndicator.svelte';

  import Fa from 'svelte-fa';
  import { faCircleUser, faChevronDown } from '@fortawesome/free-solid-svg-icons';
  import { Link, navigate } from 'svelte-routing';
  import { navItems } from '$lib/components/Navigation/NavItems.svelte';
  import { url } from '$lib/stores/nav';
  import { parseBadgeForNav } from '$lib/helpers';

  let activeRoute: string;

  url.subscribe((value) => (activeRoute = value));
</script>

<nav class="flex flex-col min-w-[320px] h-auto bg-black-900 py-6">
  <div class="h-1/6">
    <span class="flex">
      <div class="w-12 px-3" style="color: #FFFFFF">
        <Fa icon={faCircleUser} size="2x" />
      </div>
      <div class="flex flex-col gap-y-2 pl-3">
        <h4 class="text-gray-200 text-sm font-roboto">Canada Domain</h4>
        <Badge variant="blue">{parseBadgeForNav('ID#449f4f997a96467f90f7af8b396928f1')}</Badge>
        <Link class="text-gray-200 font-roboto text-sm" to="/login">Logout</Link>
      </div>
      <div class="flex justify-center w-12 px-3 pt-1" style="color: #8F8AA8">
        <Fa icon={faChevronDown} size="xs" />
      </div>
    </span>
  </div>

  <hr />

  <div class="h-4/5 py-12">
    <div class="flex flex-col justify-between h-full">
      <div>
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

      <div class="flex justify-left items-center pl-5">
        <StatusIndicator status="active" />
        <p class="pl-4 flex justify-center text-gray-200 text-sm font-roboto">Domain Online</p>
      </div>
    </div>
  </div>

  <hr />

  <div class="py-2">
    <span class="flex items-center pl-2">
      <div class="w-12 p-2" style="color: #FFFFFF">
        <Fa icon={faCircleUser} size="lg" />
      </div>
      <div class="flex flex-col gap-y-2">
        <span class="text-gray-200 text-sm font-roboto">Kyoko Eng</span>
      </div>
    </span>
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
