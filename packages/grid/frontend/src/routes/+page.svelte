<script lang="ts">
  import { goto } from "$app/navigation"
  import { onMount } from "svelte"
  import type { PageData } from "./$types"

  export let data: PageData

  onMount(() => {
    if (data?.current_user && data.current_user?.email) {
      goto("/users")
    } else {
      goto("/login")
    }
  })
</script>

<main>
  <title>Syft UI</title>
</main>
