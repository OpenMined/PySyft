<script lang="ts">
  import FlowArrowIcon from '$lib/components/icons/FlowArrowIcon.svelte';
  import TableIcon from '$lib/components/icons/TableIcon.svelte';
  import Tooltip from '$lib/components/Tooltip.svelte';
  import TooltipText from '../TooltipText.svelte';

  const dataset = {
    name: 'PNAS - Algorithmic Amplification of Politics On Twitter Dataset',
    author: 'Jane Doe',
    date: 'Dec 22, 2022',
    requests: 5,
    assets: 4,
    filesize: '80kb',
    id: '1823981723891729837'
  };
</script>

<a href="/datasets/{dataset.id}">
  <div class="p-4 pb-6 flex gap-4">
    <div class="flex flex-col gap-1 w-full">
      <h2>{dataset.name}</h2>
      <div class="flex items-center gap-3 text-gray-600">
        <h3>{dataset.author}</h3>
        <span class="dot">●</span>
        <p>Updated {dataset.date}</p>
      </div>
      <div class="flex items-center gap-3 text-gray-600">
        <div class="h-min">
          <Tooltip>
            <div class="flex items-center gap-3">
              <TableIcon weight="fill" class="w-5 h-5 text-gray-800" />
              <p>{dataset.assets}</p>
            </div>
            <TooltipText slot="tooltip">
              There are <strong>{dataset.assets} assets</strong>
              linked to this dataset.
            </TooltipText>
          </Tooltip>
        </div>
        <span class="dot">●</span>
        <p>File Size ({dataset.filesize})</p>
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

  .dot {
    @apply text-[8px] text-gray-200;
  }
</style>
