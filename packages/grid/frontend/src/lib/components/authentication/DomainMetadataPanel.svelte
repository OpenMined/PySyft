<script lang="ts">
  import Avatar from '$lib/components/Avatar.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import TagCloud from '$lib/components/TagCloud.svelte';
  import DomainOnlineIndicator from '$lib/components/DomainOnlineIndicator.svelte';
  import { prettyName } from '$lib/utils';
  import type { DomainMetadata } from '../../../types/domain/metadata';
  export let metadata: DomainMetadata;
  export let initials = 'OM';
</script>

<section class="flex flex-col w-full sm:w-[36%] sm:min-w-[544px] max-w-[784px] gap-4 py-11 px-8">
  <TagCloud tags={metadata.tags} />
  <div class="w-[97.5px] relative">
    <div class="absolute right-0">
      <DomainOnlineIndicator />
    </div>
    <Avatar {initials} />
  </div>
  <h2>{prettyName(metadata.name)}</h2>
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
    <div class="flex gap-1 items-center group relative">
      <span
        class="group-hover:opacity-100 bg-gray-800 px-2 text-sm text-gray-100 rounded-md absolute left-1/4 opacity-0 m-4 mx-auto -top-9"
        >Copy</span
      >
      <p class="uppercase text-gray-400">Id#:</p>
      <Badge>{metadata.id?.value}</Badge>
    </div>
    <div class="flex gap-1 items-center">
      <p class="uppercase text-gray-400">Deployed on:</p>
      <p class="font-mono">{metadata.deployed_on}</p>
    </div>
  </div>
</section>
