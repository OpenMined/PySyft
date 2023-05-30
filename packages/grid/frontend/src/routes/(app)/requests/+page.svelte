<script lang="ts">
  import { onMount } from 'svelte';
  import PrismCode from '$lib/components/PrismCode.svelte';
  import { getAllCodeRequests } from '$lib/api/requests';

  let requests;

  onMount(async () => {
    requests = await getAllCodeRequests();
  });
</script>

<div class="p-8">
  {#if !requests}
    <div>Loading...</div>
  {:else if requests.length === 0}
    <div>Empty requests</div>
  {:else}
    <div class="flex flex-col gap-4">
      {#each requests as request}
        <!-- prettier-ignore -->
        <PrismCode code={request.raw_code} />
      {/each}
    </div>
  {/if}
</div>
