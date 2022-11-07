<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import Modal from '$lib/components/Modal.svelte';

  let showDeleteNodeModal = false;
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col">
  <!-- Header Logo -->

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
        <Button>Save Changes</Button>
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
          <Button>Change Password</Button>
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
          as any permissions and requests. If you are the domain owner the domain node will be
          deleted as well and will be closed to all users. To transfer ownership of a domain node
          before deleting your account you can follow the instructions <a href="/">here</a>
        </p>
        <div class="inline-flex py-6">
          <Button onClick={() => (showDeleteNodeModal = true)}>Delete Account</Button>
        </div>
      </div>
    </form>
  </section>

  {#if showDeleteNodeModal}
    <Modal on:close={() => (showDeleteNodeModal = false)}>
      <div slot="icon" class="flex justify-center">
        <svg
          class="w-6 h-6 fill-current text-yellow-500"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
        >
          <path d="M0 0h24v24H0V0z" fill="none" /><path
            d="M12 5.99L19.53 19H4.47L12 5.99M12 2L1 21h22L12 2zm1 14h-2v2h2v-2zm0-6h-2v4h2v-4z"
          />
        </svg>
      </div>
      <p slot="header" class="text-center text-2xl">
        Are you sure you want to delete your account?
      </p>
      <p slot="content" class="text-center">
        Because you are the domain owner, the domain node along with all uploaded datasets, user
        accounts, and requests will be deleted. All network memberships will also be removed. If you
        would like to keep this domain node but no longer want to be an owner press “cancel” and
        follow the instructions here to transfer ownership of your domain node.
      </p>
    </Modal>
  {/if}
</main>
