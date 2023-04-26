<script lang="ts">
  import type { DomainOnlineStatus } from '../../types/domain/onlineIndicator';
  let status: DomainOnlineStatus = 'online';
</script>

<div class="flex justify-center items-center gap-2">
  <i
    class="w-2.5 h-2.5 rounded-full block"
    class:bg-green-500={status === 'online'}
    class:bg-red-500={status === 'offline'}
    class:bg-yellow-500={status === 'pending'}
  />
</div>
