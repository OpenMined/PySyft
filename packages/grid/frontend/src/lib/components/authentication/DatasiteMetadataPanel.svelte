<script lang="ts">
  import Avatar from "$lib/components/Avatar.svelte"
  import Badge from "$lib/components/Badge.svelte"
  import DatasiteOnlineIndicator from "$lib/components/DatasiteOnlineIndicator.svelte"
  import { getInitials } from "$lib/utils"
  import type { DatasiteMetadata } from "../../../types/datasite/metadata"

  export let metadata: DatasiteMetadata

  $: datasiteInitials = getInitials(metadata?.name)
</script>

<section
  class="flex flex-col w-full sm:w-[36%] sm:min-w-[544px] max-w-[784px] gap-4 py-11 px-8"
  data-testid="datasite-metadata-panel"
>
  {#if metadata}
    <div class="w-[97.5px] relative">
      <div class="absolute right-0">
        <DatasiteOnlineIndicator />
      </div>
      <Avatar bigText initials={datasiteInitials} />
    </div>
    <h2>{metadata.name}</h2>
    {#if metadata.organization}
      <p class="text-lg font-semibold flex-shrink-0">{metadata.organization}</p>
    {/if}
    <hr />
    <div class="flex flex-col gap-2.5 flex-shrink-0 pt-2 pb-4">
      <div class="flex gap-1 items-center group relative">
        <p class="uppercase text-gray-400">Id#:</p>
        <Badge>{metadata.server_id}</Badge>
      </div>
      <div class="flex gap-1 items-center group relative">
        <p class="uppercase text-gray-400">Side:</p>
        <p class="font-mono uppercase text-gray-600">{metadata.server_side}</p>
      </div>
      <div class="flex gap-1 items-center group relative">
        <p class="uppercase text-gray-400">Type:</p>
        <p class="font-mono uppercase text-gray-600">{metadata.server_type}</p>
      </div>
    </div>
  {/if}
</section>
