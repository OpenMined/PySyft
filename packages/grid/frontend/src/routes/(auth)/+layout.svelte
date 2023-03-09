<script lang="ts">
  import AuthCircles from '$lib/components/AuthCircles.svelte';
  import Nav from '$lib/components/authentication/Nav.svelte';
  import Footer from '$lib/components/authentication/Footer.svelte';
  import { getClient } from '$lib/store';
</script>

<div class="fixed top-0 right-0 w-full h-full max-w-[808px] max-h-[880px] z-[-1]">
  <AuthCircles />
</div>
<main class="flex flex-col p-10 gap-10 h-screen">
  {#await getClient() then client}
    {#await client.metadata then metadata}
      <Nav version={metadata.syft_version} />
      <div class="grow flex-shrink-0">
        <slot />
      </div>
    {/await}
  {/await}
  <Footer />
</main>
