<script>
  import '../../app.css';
  import OnBoardModal from '$lib/components/onBoardModal.svelte';
  import Sidebar from '$lib/components/Sidebar.svelte';
  import Navbar from '$lib/components/NavBar.svelte';
  import { getClient, store } from '$lib/store.js';

  let client;
  let activeUrl = '/home';

  $: metadata = '';
  $: user_info = '';

  async function loadGlobalInfos() {
    let newStore = {};
    // Load JSSerde from local Storage
    store.subscribe(async (value) => {
      newStore = value;
      // If we already have a client obj in store.
      if (newStore.client) {
        client = newStore.client;
      }
    });

    if (!client) {
      client = await getClient();
      client.access_token = window.sessionStorage.getItem('session_token');
    }

    // Load metadata from session Storage
    metadata = await client.metadata;
    newStore.metadata = metadata;
    // Load current user session info
    user_info = await client.user;
    newStore.user_info = user_info;

    store.set(newStore);
  }
</script>

<main>
  {#await loadGlobalInfos() then none}
    <Navbar bind:user_info bind:client />
    <Sidebar bind:activeUrl bind:metadata bind:user_info />
    <OnBoardModal {client} bind:user_info bind:metadata />
  {/await}
</main>
<slot />
