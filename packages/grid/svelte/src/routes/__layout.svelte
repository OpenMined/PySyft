<script>
  import '../app.css';
  import NavBar from '$lib/components/NavBar.svelte';
</script>

<div class="flex h-auto min-h-screen">
  <NavBar />
  <slot />
</div>
