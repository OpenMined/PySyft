<script>
  import { goto } from '$app/navigation';
  import { getClient } from '../lib/store.js';

  async function lazyLoad() {
    if (typeof window === 'undefined') return;

    if (!window.localStorage.getItem('key')) {
      goto('/login');
    } else {
      goto('/datasets');
    }
    return client;
  }
</script>

<main>
  {#await lazyLoad()}
    <title>PyGrid</title>
  {/await}
</main>
