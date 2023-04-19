<script>
  import { page } from '$app/stores';
  import OnBoardModal from '$lib/components/onBoardModal.svelte';
  import TopNav from '$lib/components/Navigation/TopNav.svelte';
  import SideNav from '$lib/components/Navigation/SideNav.svelte';
  export let data;

  console.log({ data, p: $page });
</script>

<SideNav />
<main class="desktop:pl-72 tablet:pl-[76px] h-full w-full">
  <TopNav />
  <slot />
</main>
