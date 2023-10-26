<script>
  let table = [1, 2, 3, 4, 5]
</script>

{#each table as element}
  <div class="p-4 pb-6 flex gap-4 hover:bg-primary-50">
    <div class="animate-pulse flex flex-grow space-x-4">
      <div class="flex flex-col space-y-2 w-full">
        <div class="flex gap-1 w-full">
          <div class="h-2 basis-1/2 bg-gray-400 rounded" />
        </div>
        <div class="flex gap-1 w-full space-x-2 items-center">
          <div class="h-2 basis-1/4 bg-gray-400 rounded" />
          <span class="dot">●</span>
          <div class="h-2 basis-1/4 bg-gray-400 rounded" />
        </div>
        <div class="flex gap-1 w-full space-x-2 items-center">
          <div class="h-2 basis-1/12 bg-gray-400 rounded" />
          <span class="dot">●</span>
          <div class="h-2 basis-3/12 bg-gray-400 rounded" />
        </div>
      </div>
    </div>
  </div>
{/each}
