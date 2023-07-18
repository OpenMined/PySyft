<script>
  import TopNav from '$lib/components/Navigation/TopNav.svelte';
  import SideNav from '$lib/components/Navigation/SideNav.svelte';
  import OnBoardModal from '$lib/components/onBoardModal.svelte';
  import { getMetadata } from '$lib/api/metadata';
  import { getSelf } from '$lib/api/users';
  import { onMount } from 'svelte';

  let open = false;
  onMount(async () => {
    const metadata = await getMetadata();
    const user = await getSelf();
    if (metadata?.on_board && user?.role?.value === 128) {
      setTimeout(function () {
        open = true;
      }, 2000);
    }
  });
</script>

<SideNav />
<main class="desktop:pl-72 tablet:pl-[76px] h-full w-full">
  <TopNav />
  <slot />
</main>
<OnBoardModal {open} />
