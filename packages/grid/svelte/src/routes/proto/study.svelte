<script lang="ts">
  import { defaultItems } from '$lib/checklist.json';
</script>

<h1>General Checklist</h1>
<ol>
  {#each Object.values(defaultItems) as item}
    <li>{item}</li>
  {/each}
</ol>
