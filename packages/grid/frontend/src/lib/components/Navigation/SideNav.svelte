<script>
  import Avatar from '../Avatar.svelte';
  import CogIcon from '../icons/CogIcon.svelte';
  import SideNavDOHandbook from './SideNavDOHandbook.svelte';
  import SideNavItems from './SideNavItems.svelte';

  let metadata = {
    name: 'Domain name'
  };
</script>

<nav class="flex flex-col w-[76px] md:w-72 h-screen fixed border-r border-gray-100">
  <section
    class="relative flex flex-col justify-center items-center pt-6 px-8 pb-[46px] gap-3 flex-shrink-0"
  >
    <div class="absolute top-0.5 right-2 text-primary-500">
      <CogIcon />
    </div>
    <div class="flex flex-col items-center justify-center gap-2">
      <div class="w-16">
        <Avatar initials="L" blackBackground />
      </div>
      <p class="leading-[1.2] font-bold">{metadata.name}</p>
    </div>
    <button class="text-sm text-gray-600 underline w-min">logout</button>
  </section>
  <hr class="border-gray-100" />
  <section class="grow py-4 flex flex-col gap-6">
    <SideNavItems />
  </section>
  <hr class="border-gray-100" />
  <section class="flex flex-col gap-16 flex-shrink-0 p-8 pt-10">
    <a href="https://openmined.org" target="_blank" rel="noreferrer noopener">
      <SideNavDOHandbook />
    </a>
    <a class="text-center" href="https://openmined.org" target="_blank" rel="noreferrer noopener">
      <img
        src="/assets/branded/openmined.png"
        alt="Empowered by OpenMined"
        class="w-full h-6 object-contain"
      />
    </a>
  </section>
</nav>
