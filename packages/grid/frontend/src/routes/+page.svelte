<script>
  import { store, getSerde } from '../lib/store.js';
  import { goto } from '$app/navigation';

  let credentials = null;

  store.subscribe((value) => {
    credentials = value.credentials;
  });
</script>

<main>
  <h1>PyGrid</h1>
  {#await getSerde() then jsserde}
    {#if JSON.stringify(credentials) === '{}'}
      {goto('/login')}
    {:else}
      {goto('/home')}
    {/if}
  {/await}
</main>
