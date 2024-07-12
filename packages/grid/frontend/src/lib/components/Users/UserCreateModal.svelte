<script lang="ts">
  import { page } from "$app/stores"
  import Button from "$lib/components/Button.svelte"
  import Modal from "$lib/components/Modal.svelte"
  import XIcon from "$lib/components/icons/XIcon.svelte"
  import UserGearIcon from "../icons/UserGearIcon.svelte"
  import ButtonGhost from "../ButtonGhost.svelte"
  import Progress from "../Progress.svelte"
  import MinusCircle from "../icons/MinusCircle.svelte"
  import Input from "../Input.svelte"
  import CheckIcon from "../icons/CheckIcon.svelte"
  import Dialog from "../Dialog.svelte"

  const cardsContent = [
    {
      roleId: 2,
      title: "Data Scientist",
      description:
        "This role is for users who will be performing computations on your datasets. They may be users you know directly or those who found your datasite through search and discovery. By default this user can see a list of your datasets and can request to get results.",
    },
    {
      roleId: 32,
      title: "Data Owner",
      description:
        "This role is for users on your team who will be responsible for uploading data to the datasite.",
    },
    {
      roleId: 128,
      title: "Admin",
      description:
        "This role is for users who will help you manage your server. This should be users you trust as they will be users who will have full permissions to the server.",
    },
  ]

  const { metadata } = $page.data

  export let open = false
  export let onClose: () => void
  export let onCreateUser: (newUser: any) => void

  let selectedRole: number | null = null
  let currentStep = 1
  let email = ""
  let name = ""
  let institution = ""

  const handleRoleSelection = (id: number) => {
    selectedRole = id
    currentStep = 2
  }

  const handleBack = () => {
    currentStep = 1
    selectedRole = null
  }

  const handleCreateUser = async () => {
    try {
      const res = onCreateUser({
        email,
        password: "changethis",
        password_verify: "changethis",
        name,
        institution,
        role: selectedRole ?? 0,
      })

      if (res?.failed) throw new Error("failed")

      currentStep = 3
    } catch (error) {
      console.error(error)
    }
  }

  const href = $page.url.href.replace("/users", "")

  $: if (!open && currentStep === 3) currentStep = 1
</script>

<Dialog bind:open>
  {#if currentStep === 1}
    <Modal>
      <div class="flex w-full" slot="header">
        <div
          class="flex flex-col justify-center items-center w-full gap-2 pt-4"
        >
          <div
            class="w-min h-min rounded-full bg-primary-500 text-gray-800 p-2"
          >
            <UserGearIcon class="w-6 h-6" />
          </div>
          <div class="text-center space-y-2">
            <h3 class="text-2xl capitalize font-bold">Select a role</h3>
            <p class="text-primary-500">Step 1 of 3</p>
          </div>
        </div>
        <button class="self-start" on:click={onClose}>
          <XIcon class="w-6 h-6" />
        </button>
      </div>
      <div class="w-full flex flex-col gap-4" slot="body">
        <div class="w-full py-2">
          <Progress max={3} value={1} />
        </div>
        <p class="text-gray-400 py-2 text-center">
          To begin let's select the role this user is going to have on your
          datasite server.
        </p>
        <div class="grid grid-cols-2 gap-6">
          {#each cardsContent as { title, description, roleId }}
            <button
              class="flex flex-col pt-3 pl-3 pr-1.5 pb-6 shadow-roles-1 border border-gray-200 rounded gap-3 text-start hover:bg-primary-50"
              on:click={() => handleRoleSelection(roleId)}
            >
              <MinusCircle class="w-6 h-6 text-primary-500" />
              <div class="space-y-0.5">
                <h3 class="text-sm font-bold">{title}</h3>
                <p class="text-sm text-gray-400">{description}</p>
              </div>
            </button>
          {/each}
        </div>
      </div>
      <div class="flex w-full justify-end px-4 gap-4" slot="button-group">
        <ButtonGhost variant="gray" on:click={onClose}>Cancel</ButtonGhost>
      </div>
    </Modal>
  {:else if currentStep === 2}
    <Modal>
      <div class="flex w-full" slot="header">
        <div
          class="flex flex-col justify-center items-center w-full gap-2 pt-4"
        >
          <div
            class="w-min h-min rounded-full bg-primary-500 text-gray-800 p-2"
          >
            <UserGearIcon class="w-6 h-6" />
          </div>
          <div class="text-center space-y-2">
            <h3 class="text-2xl capitalize font-bold">
              Determine Account Details
            </h3>
            <p class="text-primary-500">Step 2 of 3</p>
          </div>
        </div>
        <button class="self-start" on:click={onClose}>
          <XIcon class="w-6 h-6" />
        </button>
      </div>
      <div class="w-full flex flex-col gap-4" slot="body">
        <div class="w-full py-2">
          <Progress max={3} value={2} />
        </div>
        <p class="text-gray-400 py-2">
          Now that we have selected our user's role, let's describe some basic
          information about our user.
        </p>
        <div class="py-2 flex flex-col gap-6">
          <Input
            label="Email"
            id="email"
            required
            bind:value={email}
            placeholder="info@openmined.org"
          />
          <Input
            label="Full Name"
            id="name"
            required
            bind:value={name}
            placeholder="Full Name"
          />
          <Input
            label="Organization"
            required
            bind:value={institution}
            id="institution"
            placeholder="Organization name here"
          />
        </div>
      </div>
      <div class="flex w-full justify-between px-4 gap-4" slot="button-group">
        <Button on:click={handleBack} type="button">Back</Button>
        <div class="flex gap-4">
          <ButtonGhost variant="gray" on:click={onClose}>Cancel</ButtonGhost>
          <Button variant="primary" on:click={handleCreateUser}>Finish</Button>
        </div>
      </div>
    </Modal>
  {:else if currentStep === 3}
    <Modal>
      <div class="flex w-full" slot="header">
        <div
          class="flex flex-col justify-center items-center w-full gap-2 pt-4"
        >
          <div
            class="w-min h-min rounded-full bg-primary-500 text-gray-500 p-2"
          >
            <CheckIcon class="w-6 h-6" />
          </div>
          <div class="text-center space-y-2">
            <h3 class="text-2xl capitalize font-bold">Account created!</h3>
          </div>
        </div>
        <button class="self-start" on:click={onClose}>
          <XIcon class="w-6 h-6" />
        </button>
      </div>
      <div class="w-full flex flex-col gap-4" slot="body">
        <div class="w-full py-2">
          <Progress max={3} value={3} />
        </div>
        <p class="text-gray-400 py-2">
          User account created! Copy and paste the text below and email the
          account credentials to your user so they can get started.
        </p>
        <div class="bg-gray-50 p-4 pt-6 rounded-[12px] gap-4 flex flex-col">
          <h3 class="capitalize font-bold">Email invitation template</h3>
          <p>
            Welcome to {metadata?.name}
            {name},
            <br />
            You are formally invited you to join {metadata?.name} Datasite. Below is
            your login credentials and the URL to the datasite. After logging in you
            will be prompted to customize your account.
          </p>
          <a {href}>{href}</a>
          <p class="font-bold">{email}</p>
          <p class="font-bold">
            Password: <span class="font-medium">changethis</span>
          </p>
        </div>
      </div>
      <div class="flex w-full justify-end px-4 gap-4" slot="button-group">
        <Button variant="primary" on:click={onClose}>Close</Button>
      </div>
    </Modal>
  {/if}
</Dialog>
