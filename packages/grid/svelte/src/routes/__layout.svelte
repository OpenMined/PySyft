<script>
  import '../app.css';
  import NavBar from '$lib/components/NavBar.svelte';
</script>

<div class="flex h-full">
  <NavBar />
  <slot />
</div>
