<script lang="ts">
  import { goto } from '$app/navigation';
  import { logout } from '$lib/utils';
  import { getMetadata } from '$lib/api/metadata';
  import { onMount } from 'svelte';

  onMount(() => {
    logout();
    getMetadata();
    goto('/login');
  });
</script>
