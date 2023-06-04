<script lang="ts">
  window.Prism = window.Prism || {};
  window.Prism.manual = true;

  import { onMount } from 'svelte';
  import 'prismjs/themes/prism.css';
  import Prism from 'prismjs';
  import 'prismjs/plugins/line-numbers/prism-line-numbers';
  import 'prismjs/plugins/line-numbers/prism-line-numbers.css';
  import 'prismjs/plugins/normalize-whitespace/prism-normalize-whitespace';
  import 'prismjs/components/prism-python';
  import 'prismjs/plugins/toolbar/prism-toolbar';
  import 'prismjs/plugins/toolbar/prism-toolbar.css';
  import 'prismjs/plugins/copy-to-clipboard/prism-copy-to-clipboard';

  export let code = '';

  onMount(async () => {
    Prism.plugins.NormalizeWhitespace.setDefaults({
      'break-lines': 100,
      indent: 0,
      'tabs-to-spaces': 0,
      'spaces-to-tabs': 0
    });

    Prism.highlightAll();
  });
</script>

<!-- prettier-ignore -->
<pre class="bg-gray-200 line-numbers language-python"><code>{@html Prism.highlight(code, Prism.languages.python, 'python')}</code></pre>

<style>
  :global(.line-numbers .line-numbers-rows) {
    @apply border-primary-200;
  }
  pre {
    overflow: auto;
  }
</style>
