<script>
  import TopNav from '$lib/components/Navigation/TopNav.svelte';
  import SideNav from '$lib/components/Navigation/SideNav.svelte';
  export let data;
</script>

<SideNav />
<main class="desktop:pl-72 tablet:pl-[76px] h-full w-full">
  <TopNav />
  <slot />
</main>
