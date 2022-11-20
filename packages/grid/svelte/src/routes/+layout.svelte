<script>
  import '../app.css';
  import NavBar from '$lib/components/Navigation/NavBar.svelte';
  import { Router, Route, navigate } from 'svelte-routing';
  import { navItems } from '$lib/components/Navigation/NavItems.svelte';
  import Login from './login/+page.svelte';

  export let url = '';
</script>

<div class="flex h-auto min-h-screen">
  <Router {url}>
    <NavBar />

    <div class="w-full">
      <!-- {#each navItems as option} -->
      <Route path="Login" component={Login} />
      <!-- {/each} -->
    </div>
    <!-- </slot> -->
  </Router>

  <!-- Suppress console slot warning -->
  {#if false}<slot />{/if}
</div>
