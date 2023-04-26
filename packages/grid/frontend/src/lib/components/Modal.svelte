<div
  class="flex flex-col justify-center items-center gap-4 border border-gray-50 flex-nowrap bg-white rounded-lg w-full h-fit shadow-modal-1 desktop:w-[662px]"
  on:click|stopPropagation
>
  <!-- header -->
  <div class="flex flex-shrink-0 justify-between p-4 pb-0 flex-nowrap w-full h-min">
    <slot name="header" />
  </div>
  <!-- body -->
  <div class="w-full flex flex-col gap-6 px-12 py-2">
    <slot name="body" />
  </div>
  <!-- button group -->
  <div class="w-full py-8 flex justify-center items-center">
    <slot name="button-group" />
  </div>
</div>
