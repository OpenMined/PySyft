<script lang="ts">
  import { enhance } from "$app/forms"
  import Button from "$lib/components/Button.svelte"
  import DatasiteMetadataPanel from "$lib/components/authentication/DatasiteMetadataPanel.svelte"
  import Input from "$lib/components/Input.svelte"
  import Modal from "$lib/components/Modal.svelte"
  import type { ActionData, PageData } from "./$types"

  export let data: PageData
  export let form: ActionData

  const metadata = data.metadata || {}
</script>

<div
  class="flex flex-col xl:flex-row w-full h-full xl:justify-around items-center gap-12"
>
  <DatasiteMetadataPanel {metadata} />
  <form class="contents" method="POST" use:enhance>
    <Modal>
      <div
        class="flex flex-shrink-0 justify-between p-4 pb-0 flex-nowrap w-full h-min"
        slot="header"
      >
        <span class="block text-center w-full">
          <p class="text-2xl font-bold text-gray-800">Apply for an account</p>
        </span>
      </div>
      <div class="contents" slot="body">
        <div class="w-full gap-6 flex flex-col tablet:flex-row">
          <Input
            label="Full name"
            id="fullName"
            name="fullName"
            placeholder="Jane Doe"
            required
            data-testid="full_name"
          />
          <Input
            label="Company/Institution"
            name="organization"
            id="organization"
            placeholder="OpenMined University"
            data-testid="institution"
          />
        </div>
        <Input
          label="Email"
          id="email"
          name="email"
          placeholder="info@openmined.org"
          required
          data-testid="email"
        />
        <div class="w-full gap-6 flex flex-col tablet:flex-row">
          <Input
            type="password"
            label="Password"
            id="password"
            name="password"
            placeholder="******"
            required
            data-testid="password"
          />
          <Input
            type="password"
            label="Confirm Password"
            id="confirm_password"
            name="confirm_password"
            placeholder="******"
            required
            data-testid="confirm_password"
          />
        </div>
        <Input
          label="Website/Profile"
          id="website"
          name="website"
          placeholder="https://openmined.org"
          data-testid="website"
        />
        <p class="text-center text-rose-500" hidden={!form?.invalid}>
          {form?.message}
        </p>
        <p class="text-center">
          Already have an account? Sign in <a
            class="text-primary-600 underline hover:opacity-50"
            href="/login"
          >
            here
          </a>
        </p>
      </div>
      <Button
        type="submit"
        variant="secondary"
        slot="button-group"
        data-testid="submit"
      >
        Sign up
      </Button>
    </Modal>
  </form>
</div>
