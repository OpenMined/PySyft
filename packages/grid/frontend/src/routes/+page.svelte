<script>
  import { getClient } from '../lib/store.js';
  import { goto } from '$app/navigation';
</script>

<main>
  <h1>PyGrid</h1>
  {#await getClient() then client}
    {#if !client.access_token }
      {goto('/login')}
    {:else}
      {goto('/home')}
    {/if}
  {/await}
</main>
