<script lang="ts">
  import Dialog from "$lib/components/Dialog.svelte"
  import Modal from "$lib/components/Modal.svelte"
  import UserGearIcon from "$lib/components/icons/UserGearIcon.svelte"
  import XIcon from "$lib/components/icons/XIcon.svelte"
  import Button from "$lib/components/Button.svelte"
  import Progress from "$lib/components/Progress.svelte"
  import Input from "$lib/components/Input.svelte"
  import ButtonGhost from "$lib/components/ButtonGhost.svelte"
  import ServerIcon from "$lib/components/icons/ServerIcon.svelte"
  import CheckIcon from "$lib/components/icons/CheckIcon.svelte"

  export let metadata

  export let open = true
  let currentStep = 1

  let userSettings = {
    name: "",
    email: "",
    password: "",
    institution: "",
    website: "",
  }

  let datasiteSettings = {
    name: "",
    description: "",
    organization: "",
    on_board: false,
  }

  let checkRequiredDatasiteFields = () => {
    return datasiteSettings.name !== "" ? true : false
  }

  let checkRequiredUserFields = () => {
    return userSettings.name !== "" &&
      userSettings.email !== "" &&
      userSettings.password !== ""
      ? true
      : false
  }

  let handleUpdate = async () => {
    try {
      await fetch("/_syft_api/metadata", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(datasiteSettings),
      })

      await fetch(`/_syft_api/users/${userSettings.id}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(userSettings),
      })

      currentStep = currentStep + 1
    } catch (err) {
      console.log(err)
    }
  }

  let handleForward = () => {
    currentStep = currentStep + 1
  }

  let handleBack = () => {
    currentStep = currentStep - 1
  }

  let onClose: () => void = () => {
    open = false
    currentStep = 1

    userSettings = {
      name: "",
      email: "",
      password: "",
      institution: "",
      website: "",
    }

    datasiteSettings = {
      name: "",
      description: "",
      organization: "",
    }
  }
</script>

<Dialog bind:open>
  {#if currentStep === 1}
    <Modal>
      <div class="flex w-full" slot="header">
        <div
          class="flex flex-col justify-center items-center w-full gap-2 pt-4"
        >
          <div class="flex justify-center items-center w-full">
            <img
              width="264px"
              height="224px"
              src="/assets/2023_welcome_to_syft_ui.png"
              alt="Welcome to Syft UI"
            />
          </div>
          <div class="text-center space-y-2">
            <h3 class="text-2xl capitalize font-bold">
              Welcome to Syft UI!
            </h3>
            <p class="text-primary-500">Step 1 of 4</p>
          </div>
        </div>
        <button class="self-start" on:click={onClose}>
          <XIcon class="w-6 h-6" />
        </button>
      </div>
      <div class="w-full flex flex-col gap-4" slot="body">
        <div class="w-full py-2">
          <Progress max={4} value={1} />
        </div>
        <p class="text-gray-400 py-2">
          Congratulations on logging into {metadata?.name ?? ""} server. This wizard
          will help get you started in setting up your user account. You can skip
          this wizard by pressing “Cancel” below. You can edit any of your responses
          later by going to "Account Settings" indicated by your avatar in the top
          right corner of the navigation.
        </p>
      </div>
      <div class="flex w-full justify-end px-4 gap-4" slot="button-group">
        <div class="flex gap-4">
          <ButtonGhost variant="gray" on:click={onClose}>Cancel</ButtonGhost>
          <Button variant="primary" on:click={handleForward}>Next</Button>
        </div>
      </div>
    </Modal>
  {:else if currentStep == 2}
    <Modal>
      <div class="flex w-full" slot="header">
        <div
          class="flex flex-col justify-center items-center w-full gap-2 pt-4"
        >
          <div
            class="w-min h-min rounded-full bg-primary-500 text-gray-800 p-2"
          >
            <ServerIcon class="w-6 h-6" />
          </div>
          <div class="text-center space-y-2">
            <h3 class="text-2xl capitalize font-bold">Datasite Profile</h3>
            <p class="text-primary-500">Step 2 of 4</p>
          </div>
        </div>
        <button class="self-start" on:click={onClose}>
          <XIcon class="w-6 h-6" />
        </button>
      </div>
      <svg
        width="48"
        height="48"
        viewBox="0 0 24 24"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="m21 7.702-8.5 4.62v9.678c1.567-.865 6.379-3.517 7.977-4.399.323-.177.523-.519.523-.891zm-9.5 4.619-8.5-4.722v9.006c0 .37.197.708.514.887 1.59.898 6.416 3.623 7.986 4.508zm-8.079-5.629 8.579 4.763 8.672-4.713s-6.631-3.738-8.186-4.614c-.151-.085-.319-.128-.486-.128-.168 0-.335.043-.486.128-1.555.876-8.093 4.564-8.093 4.564z"
          fill-rule="nonzero"
        />
      </svg>
      <div class="flex w-full justify-between px-4 gap-4" slot="button-group">
        <Button on:click={handleBack} type="button">Back</Button>
        <div class="flex gap-4">
          <ButtonGhost variant="gray" on:click={onClose}>Cancel</ButtonGhost>
          <Button variant="primary" on:click={handleForward}>Next</Button>
        </div>
      </div>
      <div class="w-full flex flex-col gap-4" slot="body">
        <div class="w-full py-2">
          <Progress max={4} value={2} />
        </div>
        <p class="text-gray-400 py-2">
          Let's begin by describing some basic information about this datasite
          server. This information will be shown to outside users to help them
          find and understand what your datasite offers.
        </p>
        <Input
          label="Datasite Name"
          id="datasiteName"
          required
          bind:value={datasiteSettings.name}
          placeholder="ABC University Datasite"
        />
        <Input
          label="Organization"
          id="datasiteOrganization"
          bind:value={datasiteSettings.organization}
          placeholder="ABC University"
        />
        <Input
          label="Description"
          id="datasiteDescription"
          bind:value={datasiteSettings.description}
          placeholder="Describe your datasite here ..."
        />
      </div>
    </Modal>
  {:else if currentStep == 3}
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
            <h3 class="text-2xl capitalize font-bold">User Account</h3>
            <p class="text-primary-500">Step 3 of 4</p>
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
          Now that we have described our datasite, let's update our password and
          describe some basic information about ourselves for our "User
          Profile". User profile information will be shown to teammates and
          collaborators when working on studies together.
        </p>
        <div class="py-2 flex flex-col gap-6">
          <Input
            label="Email"
            id="email"
            required
            bind:value={userSettings.email}
            placeholder="info@openmined.org"
          />
          <Input
            label="Password"
            id="password"
            type="password"
            required
            bind:value={userSettings.password}
            placeholder="*****"
          />
          <div>
            <p class="text-gray-400 font-bold">Profile Information</p>
            <p class="text-gray-400">
              Now, some profile information to help your teammates and
              collaborators get to know you better.
            </p>
          </div>
          <Input
            label="Full Name"
            id="name"
            required
            bind:value={userSettings.name}
            placeholder="Full Name"
          />
          <Input
            label="Organization"
            bind:value={userSettings.institution}
            id="institution"
            placeholder="Organization name here"
          />
          <Input
            label="Website"
            bind:value={userSettings.website}
            id="website"
            placeholder="www.abc.com"
          />
        </div>
      </div>
      <div class="flex w-full justify-between px-4 gap-4" slot="button-group">
        <Button on:click={handleBack} type="button">Back</Button>
        <div class="flex gap-4">
          <ButtonGhost variant="gray" on:click={onClose}>Cancel</ButtonGhost>
          <Button variant="primary" on:click={handleUpdate}>Finish</Button>
        </div>
      </div>
    </Modal>
  {:else if currentStep == 4}
    <Modal>
      <div class="flex w-full" slot="header">
        <div
          class="flex flex-col justify-center items-center w-full gap-2 pt-4"
        >
          <div
            class="w-min h-min rounded-full bg-primary-500 text-gray-800 p-2"
          >
            <CheckIcon class="w-6 h-6" />
          </div>
          <div class="text-center space-y-2">
            <h3 class="text-2xl capitalize font-bold">Setup Complete!</h3>
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
          Congratulations on setting up your account. To edit any of your
          responses you can go to "Account Settings" indicated by your avatar in
          the top right corner of the navigation.
        </p>
      </div>
      <div class="flex w-full justify-end px-4 gap-4" slot="button-group">
        <Button variant="primary" on:click={onClose}>Close</Button>
      </div>
    </Modal>
  {/if}
</Dialog>
