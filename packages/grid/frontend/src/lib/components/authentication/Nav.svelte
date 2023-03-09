<script lang="ts">
  import GithubOfficial from '$lib/components/icons/GithubOfficial.svelte';
  export let version = '0.0.1';
</script>

<nav class="flex justify-between items-center">
  <section class="flex gap-6 items-center w-min h-min">
    <img src="/assets/small-logo.png" alt="PyGrid logo" class="h-10 object-contain" />
    <p class="text-gray-400 whitespace-nowrap">Version {version}</p>
  </section>
  <section>
    <ul class="flex gap-6 items-center">
      <li><a href="https://openmined.github.io/PySyft">Docs</a></li>
      <li><a href="https://www.openmined.org/">Community</a></li>
      <li>
        <a href="https://github.com/OpenMined/PySyft">
          <span class="block w-6 h-7"><GithubOfficial /></span>
        </a>
      </li>
    </ul>
  </section>
</nav>
