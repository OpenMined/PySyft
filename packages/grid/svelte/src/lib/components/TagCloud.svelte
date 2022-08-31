<script lang="ts">
  import Tag, { type TagColor } from './Tag.svelte';

  export let tags: Array<string> = [];
  export let tagColor: TagColor = 'primary';
</script>

<div class="flex flex-wrap gap-2 w-full">
  {#each tags as tag}
    <Tag color={tagColor}>{tag}</Tag>
  {/each}
</div>
