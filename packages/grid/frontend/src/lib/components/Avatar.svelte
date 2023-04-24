<script lang="ts">
  export let initials = '';
  export let src = '';
  export let blackBackground = false;
  export let big = false;
  export let noOutline = false;
  export let smallText = false;
  export let bigText = false;
</script>

<div
  class="bg-gradient-to-b from-primary-400 to-magenta-500 w-full h-full rounded-full flex justify-center items-center aspect-square"
  class:p-1={big}
  class:p-0.5={!big}
  class:bg-transparent={noOutline}
  class:bg-gradient-to-b={!noOutline}
>
  <div
    class="flex justify-center items-center bg-white w-full h-full rounded-full"
    class:p-1={big}
    class:p-0.5={!big}
  >
    <div
      class="h-full w-full rounded-full flex items-center justify-center"
      class:bg-gray-800={blackBackground}
      class:bg-primary-500={!blackBackground}
      class:bg-opacity-70={!blackBackground}
    >
      {#if src}
        <img {src} alt="avatar" class="h-full w-full rounded-full" />
      {:else}
        <p
          class="font-bold leading-[1.2] text-white"
          class:text-sm={smallText}
          class:text-xl={!smallText && !bigText}
          class:text-3xl={bigText}
        >
          {initials}
        </p>
      {/if}
    </div>
  </div>
</div>
