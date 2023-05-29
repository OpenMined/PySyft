<script lang="ts">
  import { onMount } from 'svelte';
  import Prism from 'prismjs';
  import 'prismjs/components/prism-python';
  import 'prismjs/plugins/normalize-whitespace/prism-normalize-whitespace';
  import 'prismjs/plugins/show-invisibles/prism-show-invisibles';
  import 'prismjs/themes/prism-solarizedlight.css';
  import 'prismjs/plugins/show-invisibles/prism-show-invisibles.css';
  import { getAllCodeRequests } from '$lib/api/requests';

  let requests;

  onMount(async () => {
    Prism.highlightAll();
    requests = await getAllCodeRequests();
  });
</script>

<div class="p-8">
  {#if !requests}
    <div>Loading...</div>
  {:else if requests.length === 0}
    <div>Empty requests</div>
  {:else}
    <div class="flex flex-col gap-4">
      {#each requests as request}
        <pre class="bg-gray-800 rounded-lg p-6">
          <code class="language-python">
            {@html Prism.highlight(`\n${request.raw_code}\n`, Prism.languages.python, 'python')}
          </code>
        </pre>
      {/each}
    </div>
  {/if}
</div>
