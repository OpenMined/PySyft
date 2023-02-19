<script lang="ts">
  import AuthCircles from '$lib/components/AuthCircles.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import Button from '$lib/components/Button.svelte';
  import Capital from '$lib/components/Capital.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import StatusIndicator from '$lib/components/StatusIndicator.svelte';
  import RegisterModal from '../../lib/components/RegisterModal.svelte';

  import { onMount } from 'svelte';
  import { Link } from 'svelte-routing';
  import { url } from '$lib/stores/nav';
  import { parseActiveRoute } from '$lib/helpers';
  import { prettyName } from '../../lib/utils.js';
  import { getClient, store } from '../../lib/store.js';
  import { goto } from '$app/navigation';

  export let location: any;
  let inputColor = 'base';
  let displayError = 'none';
  let errorText = '';
  let localStore;
  let email = '';
  let password = '';
  $: formModal = false;

  onMount(() => url.set(parseActiveRoute(location.pathname)));

  store.subscribe((value) => {
    localStore = value;
  });

  // TODO: On submit for login button
  async function login(
    client: { login: (arg0: any, arg1: any) => Promise<any> },
    email: any,
    password: any
  ) {
    await client
      .login(email, password)
      .then(() => {
        goto('/home');
      })
      .catch((error) => {
        errorText = error.message;
        inputColor = 'red';
        displayError = 'block';
      });
  }

  // TODO: Use for copying domain ID to clipboard
  // const copyToClipBoard = () => {
  //   // Get the text field
  //   var copyText = document.getElementById('gridUID');

  //   // Copy the text inside the text field
  //   navigator.clipboard.writeText(copyText?.textContent);

  //   // Alert the copied text
  //   alert('Domain UID copied!');
  // };
</script>

<div class="fixed top-0 right-0 w-full h-full max-w-[808px] max-h-[880px] z-[-1]">
  <AuthCircles />
</div>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col h-full w-full">
  {#await getClient() then client}
    <!-- {#await client.metadata then metadata} -->
    <!-- Header Logo -->
    <span>
      <img src="/images/pygrid-logo.png" alt="PyGrid logo" />
    </span>

    <!-- Register Modal -->
    <!-- <RegisterModal bind:formModal {client} nodeId={metadata.get('id').get('value')} /> -->

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
            <Badge variant="gray">ID#449f4f997a96467f90f7af8b396928f1</Badge>
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
              Don't have an account yet?<br />
              <Link to="/signup">Apply for an account here</Link>
            </p>
            <Button onClick={login}>Login</Button>
          </div>
        </Capital>
      </form>
    </section>
    <!-- Footer -->
    <span>
      <img src="/images/empowered-by-openmined.png" alt="Empowered by OpenMined logo" />
    </span>
  {/await}
  <!-- {/await} -->
</main>
