<script>
  import { browser } from '$app/environment';
  import { getPath } from '../../../lib/utils.js';
  import { onMount } from 'svelte';
  import { url } from '$lib/stores/nav';
  import { parseActiveRoute } from '$lib/helpers';

  let path = '';
  if (browser) {
    path = getPath();
  }

  onMount(() => url.set(parseActiveRoute(location.pathname)));
</script>

<main class="h-full flex items-center justify-center">
  <span class="page-container">Welcome Home! Current Path: {path}</span>
</main>
