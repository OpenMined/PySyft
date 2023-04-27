<script lang="ts">
  import ky from 'ky';
  import { onMount } from 'svelte';
  import { API_BASE_URL } from '$lib/constants';
  import { onInterval } from '$lib/utils';
  import type { DomainOnlineStatus } from '../../types/domain/onlineIndicator';

  export let status: DomainOnlineStatus = 'pending';

  const checkStatus = async () => {
    if (status === 'offline') status = 'pending';
    try {
      await ky(`${API_BASE_URL}/new/ping`);
      status = 'online';
    } catch (error) {
      status = 'offline';
    }
  };

  onMount(async () => {
    await checkStatus();
  });

  onInterval(checkStatus, 5000);
</script>

<div class="flex justify-center items-center gap-2">
  <div
    class="w-2.5 h-2.5 rounded-full block"
    class:bg-green-500={status === 'online'}
    class:bg-red-500={status === 'offline'}
    class:bg-yellow-500={status === 'pending'}
  />
</div>
