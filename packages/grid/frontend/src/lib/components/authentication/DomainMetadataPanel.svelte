<script lang="ts">
  import Avatar from '$lib/components/Avatar.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import type { DomainMetadata } from '../../../types/domain/metadata';
  export let metadata: DomainMetadata;
  export let initials = 'OM';
</script>

<div
  class="py-11 w-full sm:w-[36%] sm:min-w-[544px] max-w-[784px] flex flex-col gap-4 xl:h-full px-8"
>
  <div class="w-[97.5px]">
    <Avatar {initials} />
  </div>
  <h2>{metadata.title}</h2>
  {#if metadata.organization}
    <p class="text-lg font-semibold flex-shrink-0">{metadata.organization}</p>
  {/if}
  {#if metadata.description}
    <p class="text-base">
      {metadata.description}
    </p>
  {/if}
  <hr />
  <div class="flex flex-col gap-2.5 flex-shrink-0 pt-2 pb-4">
    <div class="flex gap-1 items-center">
      <p class="uppercase text-gray-400">Id#:</p>
      <Badge>{metadata.id?.value}</Badge>
    </div>
    <div class="flex gap-1 items-center">
      <p class="uppercase text-gray-400">Deployed on:</p>
      <p class="font-mono">{metadata.deployed_on}</p>
    </div>
  </div>
</div>
