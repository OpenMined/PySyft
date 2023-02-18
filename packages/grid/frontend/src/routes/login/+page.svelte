<script>
  import LoginHeader from '../../components/LoginHeader.svelte';
  import Button from '$lib/components/Button.svelte';
  import Capital from '$lib/components/Capital.svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import Badge from '$lib/components/Badge.svelte';
  import { getClient, store } from '../../lib/store.js';
  import { prettyName } from '../../lib/utils.js';
  import { goto } from '$app/navigation';

  $: formModal = false;
  let inputColor = 'base';
  let displayError = 'none';
  let errorText = '';

  async function login(client) {
    let password = document.getElementById('password').value
    let email = document.getElementById('email').value
    console.log("Here!")
    await client
      .login(email, password)
      .then(() => {
        goto('/signup');
      })
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

<main>
  {#await getClient() then client}
    {#await client.metadata then metadata}
      <!-- Login Screen Header -->
      <LoginHeader version={metadata.get('version')} />

      <!-- Login Screen Body -->
      <div id="login-screen">
        <!-- Background Circles -->
        <div id="login-orange-circle-bg" />
        <div id="login-blue-circle-bg" />

        <form style='position:absolute;top:24%;left:50%;z-index:2' class="w-[35%] flex-shrink-0">
          <Capital>
              <!-- Capital Header (slot: header) -->
              <div slot="header">
                <h2 class="flex justify-center text-gray-800 font-rubik text-2xl leading-normal font-medium">
                  Welcome
                </h2>
                <br>
                <div class="flex justify-center items-center">
                  <span class="dot" />
                  <p class="pl-2 flex justify-center">Domain Online</p>
                </div>
              </div>
              <!-- Capital Body (slot: body) -->
              <div class="flex flex-col gap-y-4 gap-x-6" slot="body">
                <FormControl placeholder='info@openmined.org' label="Email" id="email" type="email" required />
                <div class="flex w-full gap-x-6">
                  <span class="w-full">
                    <FormControl  placeholder='*******' label="Password" id="password" type="password" required />
                  </span>
                </div>
              </div>

              <!-- Capital Footer (slot: footer) -->
              <div class="space-y-6" slot="footer">
                <p class="text-center">
                  Don't have an account yet? Apply for an account <a class="font-medium text-blue-600 hover:underline dark:text-blue-500" href='/signup'>here</a>
                </p>
                <div style='display:flex;justify-content:center'>
                  <Button onClick={() => {login(client)}}>Login</Button>
                </div>
              </div>
          </Capital>
        </form>

        <!-- Domain Info Text -->
        <div id="domain-info">
          <div style="border-bottom: solid; height: 45vh;">
            <h1 style="font-size: 45px;">
              <b>{prettyName(metadata.get('name'))}</b>
              <h1>
                <h5 style="font-size: 15px;"><b> {metadata.get('organization')} </b></h5>
                <p style="font-size:17px;">{metadata.get('description')}</p>
              </h1>
            </h1>
          </div>
          <h3 class="info-foot">
            <b> ID# </b>
            <button
              id="gridUID"
              on:click={() => copyToClipBoard()}
              style="margin-left: 5px; color: black;padding-left:10px; padding-right:10px; background-color: #DDDDDD"
            ><Badge variant="gray">{metadata.get('id').get('value')}</Badge></button>
          </h3>
          <h3 class="info-foot"><b> DEPLOYED ON:&nbsp;&nbsp;</b> {metadata.get('deployed_on')}</h3>
        </div>
      </div>

      <a href="https://www.openmined.org/">
        <div id="login-footer">
          <h3>Empowered by</h3>
          <img
            style="margin-left: 10px; margin-right: 8vh;"
            alt="openmined-logo.png"
            width="120"
            height="120"
            src="../../public/assets/small-om-logo.png"
          />
        </div></a
      >
    {/await}
  {/await}
</main>

<svelte:window />

<style>
  #login-screen {
    height: 100%;
    width: 100%;
    position: absolute;
    overflow: hidden;
  }

  #login-footer {
    height: 10%;
    width: 100%;
    position: absolute;
    top: 90%;
    display: flex;
    justify-content: right;
    align-items: center;
    z-index: 2;
  }

  #domain-info {
    position: absolute;
    z-index: 2;
    top: 24%;
    left: 10%;
    width: 30%;
    height: 55%;
  }

  #login-orange-circle-bg {
    height: 100vh;
    width: 100vh;
    border-radius: 50%;
    position: absolute;
    z-index: 1;
    background: linear-gradient(90deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0) 100%),
      #ec9913;
    filter: blur(50px);
    left: 50%;
    z-index: 1;
  }

  #login-blue-circle-bg {
    height: 40vh;
    width: 40vh;
    border-radius: 50%;
    position: absolute;
    z-index: 3;
    background: linear-gradient(90deg, rgba(255, 255, 255, 0.5) 0%, rgba(255, 255, 255, 0) 100%),
      rgb(13, 110, 237);
    filter: blur(50px);
    top: -15%;
    left: 90%;
    z-index: 1;
  }

  .info-foot {
    margin: 15px;
    display: flex;
    color: grey;
    font-size: 11px;
  }

  .dot {
    height: 10px;
    width: 10px;
    background-color: green;
    border-radius: 50%;
    display: inline-block;
    margin-right: 5px;
    box-shadow: 0px 0px 2px 2px green;
    animation: glow 1.5s linear infinite alternate;
  }

  @keyframes glow {
    to {
      box-shadow: 0px 0px 1px 1px greenyellow;
    }
  }
</style>
