<script lang="ts">
  import type { DatasiteOnlineStatus } from '../../types/datasite/onlineIndicator';
  let status: DatasiteOnlineStatus = 'online';
</script>

<div class="flex justify-center items-center gap-2">
  <i
    class="w-2.5 h-2.5 rounded-full block"
    class:bg-green-500={status === 'online'}
    class:bg-red-500={status === 'offline'}
    class:bg-yellow-500={status === 'pending'}
  />
</div>
