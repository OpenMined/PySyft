<script lang="ts">
  import Avatar from "$lib/components/Avatar.svelte"
  import Badge from "$lib/components/Badge.svelte"
  import CaretLeft from "$lib/components/icons/CaretLeft.svelte"
  import { getInitials, getUserRole } from "$lib/utils"
  import type { UserView } from "../../../../types/datasite/users"
  import type { PageData } from "./$types"

  export let data: PageData

  let user: UserView = data.user_requested

  $: initials = getInitials(user.name)
</script>

<div class="p-6 flex flex-col gap-8">
  {#if user === null}
    <h2>Loading...</h2>
  {:else if user}
    <section>
      <a href="/users" class="inline-flex gap-2 text-primary-500 items-center">
        <CaretLeft class="w-4 h-4" /> Back
      </a>
      <div class="divide-y divide-gray-100">
        <div class="p-4 pb-6">
          <div class="w-full flex flex-col justify-center items-center gap-2">
            <span class="w-[120px] h-[120px] p-2">
              <Avatar {initials} bigText />
            </span>
            <h2>{user.name}</h2>
            {#if user.institution}
              <p class="text-lg text-gray-600">{user.institution}</p>
            {/if}
            {#if user.role}
              <div class="py-2">
                <Badge variant="gray">{getUserRole(user.role)}</Badge>
              </div>
            {/if}
          </div>
        </div>
        <div class="flex flex-col items-center justify-center gap-2 pt-4">
          <h3 class="all-caps">Contact</h3>
          <div class="space-y-1">
            <p>{user.email}</p>
            {#if user.website}
              <p>{user.website}</p>
            {/if}
          </div>
        </div>
      </div>
    </section>
  {/if}
</div>
