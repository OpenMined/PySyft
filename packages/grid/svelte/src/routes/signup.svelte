<script lang="ts">
  import AuthCircles from '$lib/components/AuthCircles.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import Button from '$lib/components/Button.svelte';
  import Capital from '$lib/components/Capital.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import TagCloud from '$lib/components/TagCloud.svelte';
</script>

<!-- <img src="/images/auth-circles.svg" aria-hidden="true" /> -->
<div class="fixed top-0 right-0 w-full h-full max-w-[808px] max-h-[880px] z-[-1]">
  <AuthCircles />
</div>
<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col h-full">
  <!-- Header Logo -->
  <span>
    <img src="/images/pygrid-logo.png" alt="PyGrid logo" />
  </span>
  <!-- Body content -->
  <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14 h-full">
    <div class="w-full">
      <TagCloud tags={['Commodities', 'Trade', 'Canada']} />
      <div class="space-y-6 mt-2">
        <h1 class="text-5xl leading-[1.1] font-medium text-gray-800 font-rubik">Canada Domain</h1>
        <p>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis in vulputate enim. Morbi
          scelerisque, ante eu ultrices semper, ipsum nisl malesuada ligula, non faucibus libero
          purus et ligula.
        </p>
      </div>
      <!-- List (Domain information) -->
      <ul class="mt-[42px] space-y-4">
        <li>
          <span class="font-bold">ID:</span>
          <!-- Badge -->
          <Badge>ID#449f4f997a96467f90f7af8b396928f1</Badge>
        </li>
        <li>
          <span class="font-bold">Hosted datasets:</span>
          <span>2</span>
        </li>
        <li>
          <span class="font-bold">Deployed on:</span>
          <span>09.07.2010</span>
        </li>
        <li>
          <span class="font-bold">Owner:</span>
          <span>Kyoko Eng, United Nations</span>
        </li>
        <li>
          <span class="font-bold">Network(s):</span>
          <span>United Nations</span>
        </li>
      </ul>
      <hr class="mt-10" />
      <!-- Support -->
      <div class="space-y-1 mt-2">
        <span>For further assistance please email:</span><br />
        <a href="mailto:support@abc.com">support@abc.com</a>
      </div>
    </div>
    <!-- Signup form -->
    <form class="w-[572px] flex-shrink-0">
      <!-- Capital -->
      <Capital>
        <!-- Capital Header (slot: header) -->
        <h2 class="text-gray-800 font-rubik text-2xl leading-normal font-medium" slot="header">
          Apply for an account
        </h2>
        <!-- Capital Body (slot: body) -->
        <div class="flex flex-col gap-y-4 gap-x-6" slot="body">
          <div class="flex w-full gap-x-6">
            <span class="max-w-1/2 w-full">
              <FormControl label="Full name" id="fullname" required />
            </span>
            <span class="max-w-1/2 w-full">
              <FormControl label="Company/Institution" id="company" optional />
            </span>
          </div>
          <FormControl label="Email" id="email" type="email" required />
          <div class="flex w-full gap-x-6">
            <span class="max-w-1/2 w-full">
              <FormControl label="Password" id="password" type="password" required />
            </span>
            <span class="max-w-1/2 w-full">
              <FormControl label="Confirm Password" id="confirm" type="password" required />
            </span>
          </div>
          <FormControl label="Website/Profile" id="website" optional />
        </div>
        <!-- Capital Footer (slot: footer) -->
        <div class="space-y-6" slot="footer">
          <Button>Submit application</Button>
          <p class="text-center">Have an account already? <a href="/login">Login here</a></p>
        </div>
      </Capital>
    </form>
  </section>
  <!-- Footer -->
  <span>
    <img src="/images/empowered-by-openmined.png" alt="Empowered by OpenMined logo" />
  </span>
</main>
