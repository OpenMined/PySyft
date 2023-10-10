<button
  class="flex flex-col shadow-callout-1 gap-3 rounded flex-nowrap overflow-hidden w-[287px] h-min group border border-gray-200"
  on:click
>
  <div class="w-full h-[150px] overflow-hidden group-hover:opacity-50">
    <slot name="image" />
  </div>
  <div class="flex flex-col gap-0.5 pb-6 pl-3 pr-2">
    <h3 class="font-bold capitalize font-sm text-gray-800"><slot name="title" /></h3>
    <p class="font-sm text-gray-600">
      <slot name="description" />
    </p>
  </div>
</button>
