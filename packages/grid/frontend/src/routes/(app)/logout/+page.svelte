<script lang="ts">
  import { goto } from '$app/navigation';
  import { logout } from '$lib/utils';
  import { onMount } from 'svelte';

  onMount(() => {
    logout();
    goto('/login');
  });
</script>
