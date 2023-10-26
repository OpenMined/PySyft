<script lang="ts">
  import TopNav from "$lib/components/Navigation/TopNav.svelte"
  import SideNav from "$lib/components/Navigation/SideNav.svelte"
  import OnBoardModal from "$lib/components/OnboardingModal.svelte"
  import type { PageData } from "./$types"

  export let data: PageData

  let open = false
</script>

<SideNav metadata={data.metadata} />

<main
  class="desktop:pl-72 tablet:pl-[76px] h-full w-full flex flex-col flex-grow"
>
  <TopNav user={data.current_user} metadata={data.metadata} />
  <slot />
</main>

<OnBoardModal {open} metadata={data.metadata} />
