<script lang="ts">
  import ButtonGhost from "../ButtonGhost.svelte"
  import CalloutCard from "../CalloutCard.svelte"
  import Dialog from "../Dialog.svelte"
  import XIcon from "../icons/XIcon.svelte"
  import Modal from "../Modal.svelte"

  export let open = false
  export let onClose: () => void
  export let onCreateGeneralUser: () => void
</script>

<Dialog bind:open>
  <Modal>
    <div slot="header" class="flex w-full">
      <h3 class="text-2xl capitalize font-regular w-full">
        Create a user account
      </h3>
      <button class="self-start" on:click={onClose}>
        <XIcon class="w-6 h-6" />
      </button>
    </div>
    <div slot="body" class="flex w-full justify-center py-2">
      <CalloutCard on:click={onCreateGeneralUser}>
        <img
          src="images/illustrations/create-general-user.png"
          alt="Create a general user"
          class="w-full h-full object-cover object-center"
          slot="image"
        />
        <span slot="title">Create general user</span>
        <span slot="description">
          Create accounts for outside researchers to safely study from your data
          without needing a copy or direct access.
        </span>
      </CalloutCard>
    </div>
    <div class="flex w-full justify-end" slot="button-group">
      <ButtonGhost variant="gray" on:click={onClose}>Cancel</ButtonGhost>
    </div>
  </Modal>
</Dialog>
