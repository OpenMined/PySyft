<script lang="ts">
  import { getUserRole } from '$lib/utils';
  import type { UserListView } from '../../../types/datasite/users';
  import Avatar from '../Avatar.svelte';
  import Badge from '../Badge.svelte';

  export let user: UserListView;

  $: userInitials = user?.name
    ?.split(' ')
    .map((name) => name[0])
    .join('');
</script>

<div class="w-full flex gap-3 pt-4 pb-6">
  {#if !user}
    Loading...
  {:else}
    <div class="w-20 h-20 p-2">
      <Avatar initials={userInitials} />
    </div>
    <div class="w-full text-gray-800 flex items-center">
      <div>
        <span class="inline-flex gap-2 items-center">
          <h2 class="font-roboto text-lg font-bold capitalize">{user.name}</h2>
          <Badge variant="gray" class="ml-2">{getUserRole(user.role.value)}</Badge>
        </span>
        <p class="text-gray-600">{user.email}</p>
      </div>
    </div>
  {/if}
</div>
