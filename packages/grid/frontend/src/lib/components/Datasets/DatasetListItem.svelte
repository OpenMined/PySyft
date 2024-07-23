<script lang="ts">
  import FlowArrowIcon from '$lib/components/icons/FlowArrowIcon.svelte';
  import TableIcon from '$lib/components/icons/TableIcon.svelte';
  import Tooltip from '$lib/components/Tooltip.svelte';
  import TooltipText from '../TooltipText.svelte';
  import type { Dataset } from '../../../types/datasite/dataset';

  export let dataset: Dataset;
</script>

<a href="/datasets/{dataset?.id?.value}">
  <div class="p-4 pb-6 flex gap-4 hover:bg-primary-50">
    <div class="flex flex-col gap-1 w-full">
      <h2>{dataset.name}</h2>
      <div class="flex items-center gap-3 text-gray-600">
        {#if dataset?.contributors?.length > 0}
          <h3>{dataset.contributors?.[0]?.name}</h3>
          <span class="dot">●</span>
        {/if}
        <p>Updated {dataset.updated_at}</p>
      </div>
      <div class="flex items-center gap-3 text-gray-600">
        <div class="h-min">
          {#if dataset?.asset_list}
            <Tooltip>
              <div class="flex items-center gap-3">
                <TableIcon weight="fill" class="w-5 h-5 text-gray-800" />
                <p>{dataset.asset_list.length}</p>
              </div>
              <TooltipText slot="tooltip">
                There are <strong>{dataset.asset_list.length} assets</strong>
                linked to this dataset.
              </TooltipText>
            </Tooltip>
          {/if}
        </div>
        <span class="dot">●</span>
        <p>File Size ({dataset.mb_size} MB)</p>
      </div>
    </div>
    <div class="h-min">
      <Tooltip>
        <div class="flex items-top text-gray-800 gap-2 h-min flex-shrink-0">
          <FlowArrowIcon class="w-5 h-5" />
          <p>{dataset.requests}</p>
        </div>
        <TooltipText slot="tooltip">
          There are <strong>{dataset.requests} requests</strong>
          linked to this dataset.
        </TooltipText>
      </Tooltip>
    </div>
  </div>
</a>
<hr class="border-gray-100" />

<style lang="postcss">
  h2 {
    @apply text-lg font-medium leading-[1.5] text-gray-900;
  }
</style>
