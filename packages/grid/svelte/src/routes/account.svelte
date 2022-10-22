<script lang="ts">
  import Button from '$lib/components/Button.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
</script>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col h-full">
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
        <!-- <div class="flex w-full gap-x-6">
          <span class="w-full">
            <FormControl label="Password" id="password" type="password" required />
          </span>
        </div> -->
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
        <Button>Login</Button>
      </div>
    </form>
  </section>
</main>
