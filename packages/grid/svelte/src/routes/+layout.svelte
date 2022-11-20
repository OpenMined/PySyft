<script>
  import '../app.css';
  import NavBar from '$lib/components/Navigation/NavBar.svelte';
  import { Router, Route, navigate } from 'svelte-routing';
  import { navItems } from '$lib/components/Navigation/NavItems.svelte';
  import AppRouter from '$lib/components/Router.svelte';

  export let url = '';
</script>

<div class="flex h-auto min-h-screen">
  <Router {url}>
    <NavBar />
    <AppRouter />
  </Router>

  <!-- Suppress console slot warning -->
  {#if false}<slot />{/if}
</div>
