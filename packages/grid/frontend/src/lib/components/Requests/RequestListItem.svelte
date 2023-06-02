<script>
  import Avatar from '$lib/components/Avatar.svelte';
  import Button from '$lib/components/Button.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { getInitials } from '$lib/utils';
  export let user = {};
  export let request = {};
</script>

<div>
  <div class="flex w-full pt-4 px-6 gap-2">
    <div class="flex h-16 flex-shrink-0">
      <!-- Avatar -->
      <Avatar big initials={getInitials(user.name)} />
    </div>
    <div class="flex w-full flex-col">
      <div class="flex items-center gap-2">
        <p class="text-lg font-bold">{user.name}</p>
        <div class="flex-shrink-0">
          <Badge variant="gray">{user.role}</Badge>
        </div>
      </div>
      <p class="text-gray-600">{user.organization}</p>
      <div class="flex flex-col w-full pt-4 gap-y-2">
        <p>{request.reason}</p>
        {#each request.replies as reply}
          <p
            class="border-l-4 pl-4"
            class:border-cyan-500={reply.uuid === '123456'}
            class:border-red-300={reply.uuid != '123456'}
          >
            {reply.text}
          </p>
        {/each}
      </div>
    </div>
    <div class="flex flex-shrink-0">
      <p>{request.date}</p>
    </div>
  </div>
  <div class="flex w-full justify-end gap-4 pt-6 pb-2">
    <Button>Accept</Button>
    <Button>Deny</Button>
  </div>
</div>
