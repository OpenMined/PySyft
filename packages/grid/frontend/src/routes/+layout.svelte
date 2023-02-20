<script>
  import '../app.css';
  import NavBar from '$lib/components/Navigation/NavBar.svelte';
  import AppRouter from '$lib/components/Router.svelte';
  import { Router } from 'svelte-routing';

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
