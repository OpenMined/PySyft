<script lang="ts">
  import AuthCircles from "$lib/components/AuthCircles.svelte"
  import Nav from "$lib/components/authentication/Nav.svelte"
  import Footer from "$lib/components/authentication/Footer.svelte"
</script>

<div
  class="fixed top-0 right-0 w-full h-full max-w-[808px] max-h-[880px] z-[-1]"
>
  <AuthCircles />
</div>
<main class="flex flex-col p-10 gap-10 h-screen">
  <Nav />
  <div class="grow flex-shrink-0">
    <slot />
  </div>
  <Footer />
</main>
