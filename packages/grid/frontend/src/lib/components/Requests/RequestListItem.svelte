<script lang="ts">
  import Avatar from '$lib/components/Avatar.svelte';
  import Button from '$lib/components/Button.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { getInitials } from '$lib/utils';
  import { syftRoles } from '$lib/constants';
  export let user = {};
  export let request = {};
  export let message = {};

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    const day = date.getDate();
    const month_year = date.toLocaleString('en-us', { month: 'short', year: 'numeric' });
    return day + ' , ' + month_year;
  };

  const checkKeys = (request_key, comment_key) => {
    console.log(request_key);
    if (request_key && comment_key) {
      return request_key.join() == comment_key.join();
    } else {
      return false;
    }
  };
</script>

<div class="hover:bg-primary-50">
  <div class="flex w-full pt-4 px-6 gap-2">
    <div class="flex h-16 flex-shrink-0">
      <!-- Avatar -->
      <Avatar big initials={getInitials(user.name)} />
    </div>
    <div class="flex w-full flex-col">
      <div class="flex items-center gap-2">
        <p class="text-lg font-bold">{user.name}</p>
        <div class="flex-shrink-0">
          <Badge variant="gray">{syftRoles[user.role.value]}</Badge>
        </div>
      </div>
      <p class="text-gray-600">{user?.institution || ''}</p>
      <div class="flex flex-col w-full pt-4 gap-y-2">
        <p>{message.subject}</p>
        {#each message.replies as reply}
          <p
            class="border-l-4 pl-4"
            class:border-cyan-500={checkKeys(
              reply.from_user_verify_key.verify_key.key,
              request.requesting_user_verify_key.verify_key.key
            )}
            class:border-red-300={!checkKeys(
              reply.from_user_verify_key.verify_key.key,
              request.requesting_user_verify_key.verify_key.key
            )}
          >
            {reply.text}
          </p>
        {/each}
      </div>
    </div>
    <div class="flex flex-shrink-0">
      <p>{formatDate(request.request_time.utc_timestamp)}</p>
    </div>
  </div>
  <div class="flex w-full justify-end gap-4 pt-6 pb-2 pr-3">
    <Button>Accept</Button>
    <Button>Deny</Button>
  </div>
</div>
