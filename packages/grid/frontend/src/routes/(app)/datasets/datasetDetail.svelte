<script>
  import { createEventDispatcher, onMount } from 'svelte';
  import Button from '$lib/components/Button.svelte';

  export let name;
  export let author;
  export let lastUpdated;
  export let fileSize;

  const dispatch = createEventDispatcher();

  onMount(() => {
    window.scrollTo(0, 0);
  });
</script>

<article>
  <div class="dataListItem">
    <ul>
      <li>{author}</li>
      <li>{name}</li>
      <li>{lastUpdated}</li>
      <li>{fileSize}</li>
    </ul>
  </div>
  <Button variant="secondary" action={() => dispatch('closeOpenCard')} id="back">Back</Button>
</article>
