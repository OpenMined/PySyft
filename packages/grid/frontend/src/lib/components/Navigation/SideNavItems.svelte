<script lang="ts">
  import { page } from "$app/stores"
  import TrayIcon from "../icons/TrayIcon.svelte"
  import TableIcon from "../icons/TableIcon.svelte"
  import UsersIcon from "../icons/UsersIcon.svelte"
  import TableFillIcon from "../icons/TableFillIcon.svelte"
  import TrayFillIcon from "../icons/TrayFillIcon.svelte"
  import UsersFillIcon from "../icons/UsersFillIcon.svelte"

  const items = [
    {
      title: "Governance",
      sections: [
        {
          icon: TrayIcon,
          iconFill: TrayFillIcon,
          label: "Requests",
          href: "/requests",
          disabled: true,
        },
      ],
    },
    {
      title: "Assets",
      sections: [
        {
          icon: TableIcon,
          iconFill: TableFillIcon,
          label: "Datasets",
          href: "/datasets",
          disabled: true,
        },
      ],
    },
    {
      title: "Admin",
      sections: [
        {
          icon: UsersIcon,
          iconFill: UsersFillIcon,
          label: "Users",
          href: "/users",
        },
      ],
    },
  ]

  let currentPathname = $page.url.pathname

  $: currentPathname = $page.url.pathname
</script>

{#each items as item}
  <div>
    <p
      class="text-xs tracking-[0.75px] font-bold text-gray-400 px-8 py-1 uppercase hidden desktop:inline-block"
    >
      {item.title}
    </p>
    <ul class="flex flex-col gap-6">
      {#each item.sections as section}
        <a href={section.href} aria-disabled={section.disabled}>
          <li
            class="justify-center desktop:justify-start desktop:px-8 desktop:pl-5 desktop:pr-2 desktop:py-1 py-0.5 flex items-center gap-1 hover:bg-primary-50"
            class:bg-primary-50={section.href === currentPathname}
            class:text-primary-600={section.href === currentPathname}
          >
            <div
              class="flex items-center justify-center px-3 py-2 text-primary-500"
            >
              <svelte:component
                this={section.href === currentPathname
                  ? section.iconFill
                  : section.icon}
                class="w-6 h-6"
              />
            </div>
            <p class="hidden desktop:inline-block">{section.label}</p>
          </li>
        </a>
      {/each}
    </ul>
  </div>
{/each}

<style lang="postcss">
  [aria-disabled] {
    @apply opacity-50 cursor-not-allowed pointer-events-none hover:bg-gray-200;
  }
</style>
