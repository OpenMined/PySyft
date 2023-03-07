<script lang="ts">
  export let initials = '';
  export let src = '';
</script>

<div
  class="bg-gradient-to-b from-primary-400 to-magenta-500 w-full h-full rounded-full flex justify-center items-center p-1 aspect-square"
>
  <div class="flex justify-center items-center bg-white w-full h-full rounded-full p-1">
    <div class="bg-primary-500/70 h-full w-full rounded-full flex items-center justify-center">
      {#if src}
        <img {src} alt="avatar" class="h-full w-full rounded-full" />
      {:else}
        <p class="font-bold leading-[1.2] text-white text-xl">{initials}</p>
      {/if}
    </div>
  </div>
</div>
