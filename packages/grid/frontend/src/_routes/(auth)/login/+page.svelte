<script lang="ts">
  import { enhance } from "$app/forms"
  import Button from "$lib/components/Button.svelte"
  import Modal from "$lib/components/Modal.svelte"
  import DatasiteMetadataPanel from "$lib/components/authentication/DatasiteMetadataPanel.svelte"
  import Input from "$lib/components/Input.svelte"
  import DatasiteOnlineIndicator from "$lib/components/DatasiteOnlineIndicator.svelte"
  import type { DatasiteOnlineStatus } from "../../../types/datasite/onlineIndicator"
  import type { PageData, ActionData } from "./$types"

  export let data: PageData
  export let form: ActionData

  const { metadata } = data

  let status: DatasiteOnlineStatus = "online"
</script>

<div
  class="flex flex-col xl:flex-row w-full h-full xl:justify-around items-center gap-12"
>
  <DatasiteMetadataPanel {metadata} />
  <form method="POST" class="contents" use:enhance>
    <section class="w-full max-w-[681px]">
      <Modal>
        <div
          class="flex flex-shrink-0 justify-between p-4 pb-0 flex-nowrap w-full h-min"
          slot="header"
        >
          <span class="block text-center w-full">
            <p class="text-2xl font-bold text-gray-800">Welcome</p>
          </span>
        </div>
        <div class="contents" slot="body">
          {#if form?.invalid}
            <p class="w-full text-center text-red-600">
              Invalid credentials! Try again.
            </p>
          {/if}
          <div class="flex justify-center items-center gap-2">
            <DatasiteOnlineIndicator />
            <p class="text-600">
              {#if status === "pending"}
                Checking connection
              {:else}
                Datasite {status}
              {/if}
            </p>
          </div>
          <input hidden name="server_id" value={metadata?.server_id} />
          <Input
            label="Email"
            type="email"
            id="email"
            name="email"
            placeholder="info@openmined.org"
            autocomplete="username"
            error={form?.invalid}
            required
            data-testid="email"
          />
          <Input
            label="Password"
            type="password"
            id="password"
            name="password"
            placeholder="******"
            error={form?.invalid}
            autocomplete="current-password"
            required
            data-testid="password"
          />
          {#if metadata?.signup_enabled}
            <p class="text-center">
              Don't have an account yet? Apply for an account <a
                href="/signup"
                class="text-primary-600 underline hover:opacity-50"
              >
                here
              </a>
            </p>
          {/if}
        </div>
        <Button type="submit" variant="secondary" slot="button-group">
          Login
        </Button>
      </Modal>
    </section>
  </form>
</div>
