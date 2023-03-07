<script>
  import '../../app.css';
  import OnBoardModal from '$lib/components/onBoardModal.svelte';
  import Sidebar from '$lib/components/Sidebar.svelte';
  import Navbar from '$lib/components/NavBar.svelte';
  import { getClient } from '$lib/store.js';

  let client;
  let activeUrl = '/home';

  $: metadata = '';
  $: user_info = '';

  async function loadGlobalInfos() {
    // Get JSClient
    client = await getClient();

    // Load metadata from session Storage
    if (!metadata) {
      metadata = await client.metadata;
    }

    // Load current user session info
    if (!user_info) {
      user_info = await client.user;
    }
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
