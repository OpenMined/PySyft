<script>
  import { getClient } from '$lib/store';
  import Github from '$lib/components/icons/Github.svelte';
  import AuthCircles from '$lib/components/AuthCircles.svelte';
  import Button from '$lib/components/Button.svelte';
  import Capital from '$lib/components/Capital.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import { prettyName } from '$lib/utils.js';
  import { goto } from '$app/navigation';
  import Badge from '$lib/components/Badge.svelte';

  let inputColor = 'base';
  let displayError = 'none';
  let errorText = '';


  async function login(client) {
    let password = document.getElementById('password').value;
    let email = document.getElementById('email').value;

    await client
      .login(email, password)
      .then(() => {goto('/home')})
      .catch((error) => {
        errorText = error.message;
        inputColor = 'red';
        displayError = 'block';
      });
  }

  const copyToClipBoard = () => {
    // Get the text field
    var copyText = document.getElementById('gridUID');

    // Copy the text inside the text field
    navigator.clipboard.writeText(copyText?.textContent);

    // Alert the copied text
    alert('Domain UID copied!');
  };
</script>

<div class="fixed top-0 right-0 w-full h-full max-w-[808px] max-h-[880px] z-[-1]">
  <AuthCircles />
</div>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col h-full w-full">
  {#await getClient() then client}
    {#await client.metadata then metadata}
      <!-- Header Logo -->
      <div class="w-full flex justify-between ">
        <div class="flex items-center gap-2">
          <img width="100px" src="../../public/assets/small-logo.png" alt="pygrid-logo.png" />
          <span class="font-roboto">Version: {metadata.syft_version}</span>
        </div>
        <div class="flex justify-end gap-5">
          <a
            href="https://openmined.github.io/PySyft/index.html"
            style="text-decoration:none;color:black"
          >
            Docs
          </a>
          <a href="https://www.openmined.org/" style="text-decoration:none;color:black">Community</a
          >
          <a href="https://github.com/OpenMined/PySyft" style="text-decoration:none;color:black"
            ><Github /></a
          >
        </div>
      </div>

      <!-- Body content -->
      <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14 h-full">
        <div class="w-full">
          <div class="mt-2">
            <h1 class="text-5xl leading-[1.1] font-medium text-gray-800 font-rubik">
              {prettyName(metadata.name)}
            </h1>
          </div>
          <div class="mt-2 ">
            <h1 class="text-2xl leading-[1.1] font-medium text-gray-500 font-rubik">
              {metadata.organization}
            </h1>
          </div>
          <div class="mt-5 h-2/5">
            <p class="text-medium leading-[1.1] font-medium text-gray-800 font-roboto">
              {metadata.description}
            </p>
          </div>

          <!-- List (Domain information) -->
          <div class="flex flex-col py-7 px-3 border-t border-gray-300 ">
            <button
              on:click={() => {
                copyToClipBoard();
              }}
              class="flex items-center gap-2"
            >
              <h1 class="font-bold">ID:</h1>
              <Badge variant="gray">{metadata.id.value}</Badge>
            </button>

            <div class="flex items-center gap-2">
              <span class="font-bold"> DEPLOYED ON: </span>
              <span>{metadata.deployed_on.split(' ')[0]}</span>
            </div>
          </div>

          <!---
        <ul class="flex mt-[42px]">
          <li>
            <span class="font-bold">ID:</span>

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
        -->
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
                Don't have an account yet? Apply for an account <a
                  class="font-medium text-blue-600 hover:underline dark:text-blue-500"
                  href="/signup">here</a
                >
              </p>
              <Button
                onClick={() => {
                  login(client);
                }}>Login</Button
              >
            </div>
          </Capital>
        </form>
      </section>
      <!-- Footer -->
      <span>
        <img src="/images/empowered-by-openmined.png" alt="Empowered by OpenMined logo" />
      </span>
    {/await}
  {/await}
</main>

<style>
</style>
