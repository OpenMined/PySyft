<script lang="ts">
  import { invalidate } from "$app/navigation"
  import debounce from "just-debounce-it"
  import Badge from "$lib/components/Badge.svelte"
  import Filter from "$lib/components/Filter.svelte"
  import Search from "$lib/components/Search.svelte"
  import Pagination from "$lib/components/Pagination.svelte"
  import UserListItem from "$lib/components/Users/UserListItem.svelte"
  import PlusIcon from "$lib/components/icons/PlusIcon.svelte"
  import UserNewModal from "$lib/components/Users/UserNewModal.svelte"
  import UserCreateModal from "$lib/components/Users/UserCreateModal.svelte"
  import type { UserListView } from "../../../types/domain/users"
  import type { PageData } from "./$types"
  import { throwIfError } from "$lib/api/syft_error_handler"

  export let data: PageData

  let searchTerm = ""
  let userList: UserListView[] = data.list || []
  let total: number = data.total || 0
  let paginators: number[] = [5, 10, 15, 20, 25]
  let page_size: number = 5
  let page_index: number = 0
  let page_row: number = 5

  let openModal: string | null = null

  const closeModal = () => {
    openModal = null
  }

  const onCreateGeneralUser = () => {
    openModal = "step1"
  }

  const search = debounce(async () => {
    console.log("search", { search })
    const url = searchTerm
      ? `/_syft_api/users/search?name=${searchTerm}&page_size=${page_size}`
      : "/_syft_api/users"
    const res = await fetch(url)
    const json = await res.json()
    userList = json.list
    total = json.total
    console.log("search end", { userList, total })
  }, 300)

  const createUser = async (newUser) => {
    try {
      const res = await fetch("/_syft_api/users", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(newUser),
      })
      if (res.ok) {
        const json = await res.json()
        throwIfError(json)
        invalidate("/users")
      }
    } catch (err) {
      console.log(err)
      return { failed: true }
    }
  }

  const handleUpdate = async () => {
    try {
      const res = await fetch(
        `/_syft_api/users?page_size=${page_size}&page_index=${page_index}`
      )
      const json = await res.json()

      console.log("hU", { json })
      userList = json.list
      total = json.total
    } catch (err) {
      console.error({ err })
      invalidate("/users")
    }
  }

  $: console.log({ userList })
</script>

<div class="pt-8 desktop:pt-2 pl-16 pr-[140px] flex flex-col gap-[46px]">
  <div class="flex justify-center w-full">
    <div class="w-[438px] h-[263px]">
      <img
        src="images/illustrations/user-main.png"
        alt="User main alt"
        class="w-full h-full object-contain"
      />
    </div>
  </div>
  <div class="flex items-center justify-between w-full gap-8">
    <div class="w-full max-w-[378px]">
      <Search
        type="text"
        placeholder="Search by name"
        bind:value={searchTerm}
        on:input={search}
      />
    </div>
    <div class="flex-shrink-0">
      <div class="flex gap-2.5">
        <Badge variant="gray">Total: {total || 0}</Badge>
        <Filter
          variant="gray"
          filters={paginators}
          bind:filter={page_size}
          bind:index={page_index}
          on:setFilter={handleUpdate}
        >
          Filter:
        </Filter>
      </div>
    </div>
  </div>
  <div class="divide-y divide-gray-100">
    {#each userList as user}
      <a
        class="block hover:bg-primary-100 cursor-pointer"
        href={`/users/${user.id.value}`}
      >
        <UserListItem {user} />
      </a>
    {/each}
  </div>
  <div class="flex justify-center items-center mb-8 divide-y divide-gray-100">
    <Pagination
      variant="gray"
      {total}
      {page_size}
      {page_row}
      bind:page_index
      on:setPagination={handleUpdate}
    />
  </div>
</div>
<div class="fixed bottom-10 right-12">
  <button
    class="bg-black text-white rounded-full w-14 h-14 flex items-center justify-center"
    on:click={() => (openModal = "newUser")}
  >
    <PlusIcon class="w-6 h-6" />
  </button>
</div>

<UserNewModal
  open={openModal === "newUser"}
  onClose={closeModal}
  {onCreateGeneralUser}
/>
<UserCreateModal
  open={openModal === "step1"}
  onClose={closeModal}
  onCreateUser={createUser}
/>
