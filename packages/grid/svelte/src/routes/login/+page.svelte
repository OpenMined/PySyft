<script lang="ts">
  import AuthCircles from '$lib/components/AuthCircles.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import Button from '$lib/components/Button.svelte';
  import Capital from '$lib/components/Capital.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import StatusIndicator from '$lib/components/StatusIndicator.svelte';
</script>

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
      <div class="space-y-6 mt-2">
        <h1 class="text-5xl leading-[1.1] font-medium text-gray-800 font-rubik">Canada Domain</h1>
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
    </div>

    <!-- Login form -->
    <form class="w-[572px] flex-shrink-0">
      <!-- Capital -->
      <Capital>
        <!-- Capital Header (slot: header) -->
        <div slot="header">
          <h2
            class="flex justify-center text-gray-800 font-rubik text-2xl leading-normal font-medium"
          >
            Welcome Back
          </h2>
          <div class="flex justify-center items-center">
            <StatusIndicator status="active" />
            <p class="pl-2 flex justify-center">Domain Online</p>
          </div>
        </div>
        <!-- Capital Body (slot: body) -->
        <div class="flex flex-col gap-y-4 gap-x-6" slot="body">
          <FormControl label="Email" id="email" type="email" required />
          <div class="flex w-full gap-x-6">
            <span class="w-full">
              <FormControl label="Password" id="password" type="password" required />
            </span>
          </div>
        </div>

        <!-- Capital Footer (slot: footer) -->
        <div class="space-y-6" slot="footer">
          <p class="text-center">
            Don't have an account yet? <a href="/signup">Apply for an account here</a>
          </p>
          <Button>Login</Button>
        </div>
      </Capital>
    </form>
  </section>
  <!-- Footer -->
  <span>
    <img src="/images/empowered-by-openmined.png" alt="Empowered by OpenMined logo" />
  </span>
</main>
