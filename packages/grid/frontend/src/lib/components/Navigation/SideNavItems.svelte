<script lang="ts">
  import InboxIcon from '../icons/InboxIcon.svelte';
  import TableIcon from '../icons/TableIcon.svelte';
  import UsersIcon from '../icons/UsersIcon.svelte';

  const items = [
    { title: 'Governance', sections: [{ icon: InboxIcon, label: 'Requests' }] },
    { title: 'Assets', sections: [{ icon: TableIcon, label: 'Datasets' }] },
    { title: 'Admin', sections: [{ icon: UsersIcon, label: 'Users' }] }
  ];
</script>

{#each items as item}
  <div>
    <p class="text-xs tracking-[0.75px] font-bold text-gray-400 px-8 py-1 uppercase">
      {item.title}
    </p>
    <ul class="flex flex-col gap-6 hover:bg-primary-50 cursor-pointer">
      {#each item.sections as section}
        <li class="px-8 pl-5 pr-2 py-1 flex items-center gap-1 hover:bg-primary-50 cursor-pointer">
          <div class="flex items-center justify-center px-3 py-2 text-primary-500">
            <svelte:component this={section.icon} />
          </div>
          <p>{section.label}</p>
        </li>
      {/each}
    </ul>
  </div>
{/each}
