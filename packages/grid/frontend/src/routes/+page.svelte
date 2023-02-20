<script>
  import { getClient } from '../lib/store.js';
  import { goto } from '$app/navigation';

  async function lazyLoad(){
    let client =  await getClient();
    if (!client.access_token){
      goto("/login")
    } else {
      goto("/home")
    }
    return client
  }
</script>

<main>
  {#await lazyLoad() then client}{/await}
</main>