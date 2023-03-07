<script lang="ts">
  import AuthCircles from '$lib/components/AuthCircles.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import Button from '$lib/components/Button.svelte';
  import Capital from '$lib/components/Capital.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import TagCloud from '$lib/components/TagCloud.svelte';
  import { getClient } from '$lib/store';
  import { prettyName } from '$lib/utils';

  async function createUser(client) {
    let email = document.getElementById('email') ? document.getElementById('email').value : null;
    let password = document.getElementById('password')
      ? document.getElementById('password').value
      : null;
    let passwordConfirmation = document.getElementById('password')
      ? document.getElementById('confirm').value
      : null;
    let name = document.getElementById('fullname')
      ? document.getElementById('fullname').value
      : null;
    let institution = document.getElementById('company')
      ? document.getElementById('company').value
      : null;
    let website = document.getElementById('website')
      ? document.getElementById('website').value
      : null;

    if (password !== passwordConfirmation) {
      throw Error('Password and password confirmation mismatch');
    }

    let newUser = {
      email: email,
      password: password,
      password_verify: passwordConfirmation,
      name: name,
      institution: institution,
      website: website
    };

    // Filter attributes that doesn't exist
    Object.keys(newUser).forEach((k) => newUser[k] == null && delete newUser[k]);
    await client.register(newUser); // This will return a success message and the new user info
  }
</script>

<div class="fixed top-0 right-0 w-full h-full max-w-[808px] max-h-[880px] z-[-1]">
  <AuthCircles />
</div>

<main class="px-4 py-3 md:12 md:py-6 lg:px-36 lg:py-10 z-10 flex flex-col h-full w-full">
  {#await getClient() then client}
    {#await client.metadata then metadata}
      <!-- Header Logo -->
      <span>
        <img src="/images/pygrid-logo.png" alt="PyGrid logo" />
      </span>
      <!-- Body content -->
      <section class="md:flex md:gap-x-[62px] lg:gap-x-[124px] mt-14 h-full">
        <div class="w-full">
          <TagCloud tags={['Commodities', 'Trade', 'Canada']} />
          <div class="space-y-6 mt-2">
            <h1 class="text-5xl leading-[1.1] font-medium text-gray-800 font-rubik">
              {prettyName(metadata.name)}
            </h1>
            <div class="mt-2 ">
              <h1 class="text-2xl leading-[1.1] font-medium text-gray-500 font-rubik">
                {metadata.organization}
              </h1>
            </div>
            <p>{metadata.description}</p>
          </div>
          <!-- List (Domain information) -->
          <ul class="mt-[42px] space-y-4">
            <li>
              <span class="font-bold">ID:</span>
              <!-- Badge -->
              <Badge variant="gray">{metadata.id.value}</Badge>
            </li>
            <li>
              <span class="font-bold">Deployed on:</span>
              <span>{metadata.deployed_on.split(' ')[0]}</span>
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
              <Button
                onClick={() => {
                  createUser(client);
                }}>Submit application</Button
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
