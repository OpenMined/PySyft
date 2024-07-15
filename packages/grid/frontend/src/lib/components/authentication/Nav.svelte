<script lang="ts">
  import { page } from "$app/stores"
  import GithubOfficial from "$lib/components/icons/GithubOfficial.svelte"

  export let version: string = $page.data?.metadata?.syft_version
</script>

<nav class="flex justify-between items-center" data-testid="auth-nav">
  <section class="flex gap-6 items-center w-min h-min">
    <img
      src="/assets/small-logo.png"
      alt="Syft UI logo"
      class="h-10 object-contain"
    />
    {#if version}
      <p class="text-gray-400 whitespace-nowrap" data-testid="auth-nav-version">
        Version {version}
      </p>
    {/if}
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
