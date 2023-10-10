<script>
  import { metadata, user } from '$lib/store';
  import { getInitials } from '$lib/utils';
  import Avatar from '../Avatar.svelte';
  import BellIcon from '../icons/BellIcon.svelte';
  import CollegeIcon from '../icons/CollegeIcon.svelte';
  import QuestionIcon from '../icons/QuestionIcon.svelte';

  const links = [
    { href: 'https://github.com/OpenMined/PySyft', icon: QuestionIcon },
    { href: 'https://courses.openmined.org', icon: CollegeIcon },
    { href: '', icon: BellIcon }
  ];

  $: domainInitials = getInitials($metadata?.name);
  $: userInitials = getInitials($user?.name);
</script>

<div
  class="w-full py-2 px-6 flex items-center justify-between tablet:justify-end shadow-topbar-1 tablet:shadow-none"
>
  <div class="w-12 h-12 tablet:hidden">
    <Avatar initials={domainInitials} blackBackground smallText />
  </div>
  <ul class="flex w-min items-center text-primary-500">
    {#each links as link}
      <li class="items-center justify-center w-13 h-13 hidden tablet:flex">
        <a class="block w-6 h-6" href={link.href} target="_blank" rel="noopener noreferrer">
          <svelte:component this={link.icon} />
        </a>
      </li>
    {/each}
    <!-- <li class="w-[52px] h-[52px]">
      <a href="/account" class="cursor-pointer hover:opacity-90">
        <Avatar smallText noOutline initials={userInitials} />
      </a>
    </li> -->
  </ul>
</div>
