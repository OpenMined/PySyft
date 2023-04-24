<script lang="ts">
  import type { UserListView } from '../../../types/domain/users';
  import Avatar from '../Avatar.svelte';

  export let user: UserListView = {
    id: '1',
    name: 'John Doe',
    email: 'info@openmined.org'
  };

  $: userInitials = user?.name
    ?.split(' ')
    .map((name) => name[0])
    .join('');
</script>

<div class="w-full flex gap-3 pt-4 pb-6">
  <div class="w-20 h-20 p-2">
    <Avatar initials={userInitials} />
  </div>
  <div class="w-full text-gray-800 flex items-center">
    <div>
      <h2 class="font-roboto text-lg font-bold">{user.name}</h2>
      <p class="text-gray-600">{user.email}</p>
    </div>
  </div>
</div>
