<script lang="ts">
  export let label: string;
  export let id: string;
  export let required = false;
  export let optional = false;
  export let disabled = false;
  export let error = '';
  export let hint = '';
  export let type = 'text';
  export let placeholder = '';
  export let rows = 4;
</script>

<div aria-disabled={disabled} class="space-y-2">
  <label for={id} class="flex gap-x-1">
    <span class="label">{label}</span>
    {#if required}
      <span class="required">*</span>
    {:else if optional}
      <span class="optional">(optional)</span>
    {/if}
  </label>
  {#if hint}
    <p class="hint">{hint}</p>
  {/if}
  {#if type == 'textarea'}
    <textarea {id} {placeholder} {rows} class="w-full border border-gray-500" />
  {:else}
    <input {type} {id} {placeholder} class="w-full border border-gray-500" />
  {/if}
  {#if error}
    <p class="error">{error}</p>
  {/if}
</div>

<style lang="postcss">
  [aria-disabled='true'] {
    @apply opacity-50 cursor-not-allowed;
  }

  .required {
    @apply text-primary-600 text-xs relative;
  }

  .label {
    @apply text-sm font-roboto font-bold leading-normal text-gray-500;
  }

  .optional {
    @apply text-primary-600 italic font-normal text-xs leading-160;
  }

  .hint {
    @apply text-gray-500 font-normal text-sm leading-normal;
  }

  .error {
    @apply text-red-600 text-right;
  }

  input {
    @apply w-full border border-gray-300 p-3 rounded;
    @apply text-sm leading-normal font-normal font-roboto;
  }

  input::placeholder {
    @apply text-gray-300;
  }
</style>
