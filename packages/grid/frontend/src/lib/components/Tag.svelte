<script lang="ts" context="module">
  const tagColors = {
    gray: 'text-gray-600 bg-gray-100 hover:bg-gray-800 hover:text-primary-200',
    primary: 'text-primary-600 bg-primary-100 hover:bg-primary-500 hover:text-white',
    tertiary: 'text-violet-600 bg-violet-100 hover:bg-violet-500 hover:text-white',
    quaternaty: 'text-blue-600 bg-blue-100 hover:bg-blue-500 hover:text-white'
  };

  export type TagColor = keyof typeof tagColors;
</script>

<script lang="ts">
  export let color: TagColor = tagColors.primary as TagColor;
  export let disabled = false;

  const tagColor = tagColors[color] || tagColors.primary;
</script>

<span
  class="{tagColor} rounded-full gap-1.5 px-2 py-0.5 items-center inline-flex"
  aria-disabled={disabled}
>
  <slot />
</span>

<style lang="postcss">
  [aria-disabled='true'] {
    @apply opacity-50;
  }
</style>
