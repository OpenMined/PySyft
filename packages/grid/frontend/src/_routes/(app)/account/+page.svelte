<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import Modal from '$lib/components/DynamicModal.svelte';
  import YellowWarn from '$lib/components/icons/YellowWarn.svelte';
  import GreenCheck from '$lib/components/icons/GreenCheck.svelte';
  import Info from '$lib/components/icons/Info.svelte';

  let showDeleteServerModal = false;
  let showDeleteAccountModal = false;
  let showDeleteConfirmModal = false;

  function placeholderAction() {
    alert('TODO');
  }
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col">
  <!-- Header -->
  <div
    class="flex items-center bg-primary-100 border-t border-b border-primary-500 text-black-700 px-4 py-3"
    role="alert"
  >
    <Info />
    <p>
      Your profile information is public-facing information that other users and server owners can
      see.
    </p>
  </div>

  <!-- Body content -->
  <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14 h-full">
    <!-- Account settings form -->
    <form class="w-[572px] flex-shrink-0">
      <div>
        <h2
          class="flex justify-left text-gray-800 font-rubik text-2xl leading-normal font-medium pb-4"
        >
          Profile
        </h2>
      </div>
      <div class="flex flex-col gap-y-4 gap-x-6">
        <FormControl label="Full Name" id="fullName" type="text" required />
        <FormControl label="Email" id="email" type="email" required />
        <FormControl label="Company/Institution" id="company" type="text" optional />
        <p class="text-left text-gray-500 text-xs">
          Which company, organization or institution are you affiliated with?
        </p>
        <FormControl label="Website/Profile" id="website" type="text" optional />
        <p class="text-left text-gray-500 text-xs">
          Provide a link to your personal or university webpage or a social media profile to help
          others get to know you.
        </p>
      </div>
      <div class="inline-flex py-6">
        <Button variant="secondary" action={placeholderAction}>Save Changes</Button>
        <a class="flex items-center no-underline pl-8 font-bold" href="/">Cancel</a>
      </div>

      <hr class="my-10" />

      <div>
        <h2
          class="flex justify-left text-gray-800 font-rubik text-2xl leading-normal font-medium pb-4"
        >
          Password
        </h2>
        <div class="flex flex-col gap-y-4 gap-x-6">
          <FormControl label="Current Password" id="currentPassword" type="password" required />
          <FormControl label="New Password" id="newPassword" type="password" required />
        </div>
        <div class="inline-flex py-6">
          <Button variant="secondary" action={placeholderAction}>Change Password</Button>
          <a class="flex items-center no-underline pl-8 font-bold" href="/">Cancel</a>
        </div>
      </div>

      <hr class="my-10" />

      <div>
        <h2
          class="flex justify-left text-gray-800 font-rubik text-2xl leading-normal font-medium pb-4"
        >
          Delete Account
        </h2>
        <p class="text-left text-gray-500">
          When you delete your user account all information relating to you will be deleted as well
          as any permissions and requests. If you are the datasite owner the datasite server will be
          deleted as well and will be closed to all users. To transfer ownership of a datasite server
          before deleting your account you can follow the instructions <a href="/">here</a>
        </p>
        <div class="inline-flex py-6">
          <Button variant="delete" action={() => (showDeleteAccountModal = true)}>
            Delete Account
          </Button>
        </div>
      </div>
    </form>
  </section>

  {#if showDeleteAccountModal}
    <Modal>
      <div slot="header" class="flex justify-center">
        <YellowWarn />
        <p class="text-center text-2xl font-bold">Are you sure you want to delete your account?</p>
      </div>
      <p slot="body" class="text-center">
        If deleted all uploaded documents will be deleted and all open requests will be closed. Keep
        in mind any legal agreements pertaining to the use of your data requests will still apply
        according to the terms of the agreement signed. If you would like to proceed press “Delete
        Account” if not you can click “Cancel”.
      </p>
      <div slot="footer" class="flex justify-center pt-6">
        <Button
          variant="delete"
          action={() => {
            showDeleteAccountModal = false;
            showDeleteConfirmModal = true;
          }}
        >
          Delete Account
        </Button>
        <a class="flex items-center no-underline pl-8 font-bold text-magenta-500" href="/">
          Cancel
        </a>
      </div>
    </Modal>
  {/if}

  {#if showDeleteServerModal}
    <Modal>
      <div slot="header" class="flex justify-center">
        <YellowWarn />
        <p class="text-center text-2xl font-bold">Are you sure you want to delete your server?</p>
      </div>
      <p slot="body" class="text-center">
        Because you are the datasite owner, the datasite server along with all uploaded datasets, user
        accounts, and requests will be deleted. All network memberships will also be removed. If you
        would like to keep this datasite server but no longer want to be an owner press “cancel” and
        follow the instructions here to transfer ownership of your datasite server.
      </p>
      <div slot="footer" class="flex justify-center pt-6">
        <Button
          variant="delete"
          action={() => {
            showDeleteServerModal = false;
            showDeleteConfirmModal = true;
          }}
        >
          Delete Server
        </Button>
        <a class="flex items-center no-underline pl-8 font-bold text-magenta-500" href="/">
          Cancel
        </a>
      </div>
    </Modal>
  {/if}

  {#if showDeleteConfirmModal}
    <Modal size="lg">
      <div slot="header" class="flex justify-center">
        <GreenCheck />
        <p class="text-center text-2xl font-bold">Your account has been deleted</p>
      </div>
      <div slot="body">
        <p class="text-center">
          To help us improve future experiences could you share with us any frustrations or
          suggestions you have with or for the Syft UI Platform?
        </p>

        <form class="flex-shrink-0 pt-6">
          <div class="flex flex-col gap-y-4 gap-x-6">
            <FormControl
              label="Frustrations"
              id="frustrations"
              type="textarea"
              placeholder="What felt vague or cumbersome?"
              optional
            />
            <FormControl
              label="Suggestions"
              id="suggestions"
              type="textarea"
              placeholder="Did you have moments of thinking “I wish I could...”"
              optional
            />
          </div>
        </form>
      </div>
      <div slot="footer" class="flex justify-center pt-6">
        <Button
          variant="delete"
          action={() => {
            showDeleteConfirmModal = false;
          }}
        >
          Submit Response
        </Button>
        <a class="flex items-center no-underline pl-8 font-bold" href="/">Cancel</a>
      </div>
    </Modal>
  {/if}
</main>
