<script lang="ts">
  import { onMount } from 'svelte';
  import { url } from '$lib/stores/nav';
  import { parseActiveRoute } from '$lib/helpers';

  export let location: any;

  onMount(() => url.set(parseActiveRoute(location.pathname)));
</script>

<div class="flex flex-col p-2">
  <h1 class="text-3xl font-semibold">PyGrid UI</h1>
  <p>Svelte app, 0.8.0</p>
</div>
