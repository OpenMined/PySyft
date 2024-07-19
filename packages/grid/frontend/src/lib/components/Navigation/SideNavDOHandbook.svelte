<script lang="ts">
  import BookIcon from '../icons/BookIcon.svelte';
  import ChevronRightIcon from '../icons/ChevronRightIcon.svelte';
</script>

<a
  href="https://openmined.org"
  target="_blank"
  rel="noreferrer noopener"
  class="hidden desktop:block"
>
  <div
    class="flex flex-col gap-3 p-3 pb-6 rounded-[20px] bg-gradient-to-l from-gray-900 to-gray-900 hover:from-gray-700 hover:to-gray-900 active:to-gray-700 active:from-gray-700 transition"
  >
    <div class="flex justify-between items-center w-full">
      <!-- icon container -->
      <div class="rounded-full bg-primary-500 p-1.5 w-min">
        <span class="block w-3 h-3">
          <BookIcon />
        </span>
      </div>
      <span class="block w-4 h-4 text-primary-500">
        <ChevronRightIcon />
      </span>
    </div>
    <div class="gap-[2px] pl-0.5 pr-2">
      <p class="text-sm text-gray-50 font-bold">Data Owner handbook</p>
      <p class="text-sm text-gray-200">
        Check out the data owner handbook to learn more tips & tricks about how to manage your
        datasite server.
      </p>
    </div>
  </div>
</a>
