<script>
  import { Modal, Button } from 'flowbite-svelte';
  import { shortName } from '$lib/utils.ts';
  import FormControl from '$lib/components/FormControl.svelte';
  import YellowWarn from '$lib/components/icons/YellowWarn.svelte';
  export let user_info;
  export let client;
  export let userModal;

  let modalActiveList = [false, false, false, false, false, false];
  $: changeModal = !(
    !modalActiveList[0] &&
    !modalActiveList[1] &&
    !modalActiveList[2] &&
    !modalActiveList[3] &&
    !modalActiveList[4] &&
    !modalActiveList[5]
  );
  $: unfocusModal = !changeModal
    ? 'background-color: white; filter: brightness(100%)'
    : 'background-color: white; filter: brightness(60%)';

  async function submitUserChanges() {
    let email = document.getElementById('email') ? document.getElementById('email').value : null;
    let password = document.getElementById('password')
      ? document.getElementById('password').value
      : null;
    let name = document.getElementById('name') ? document.getElementById('name').value : null;
    let institution = document.getElementById('team')
      ? document.getElementById('team').value
      : null;
    let website = document.getElementById('website')
      ? document.getElementById('website').value
      : null;

    let userInfo = {
      email: email,
      password: password,
      name: name,
      institution: institution,
      website: website
    };

    // Filter attributes that doesn't exist
    Object.keys(userInfo).forEach((k) => userInfo[k] == null && delete userInfo[k]);

    // Update user info
    user_info = await client.updateCurrentUser(userInfo);

    // Cleaning Element Values
    const elementsList = [
      [document.getElementById('name'), 0],
      [document.getElementById('team'), 1],
      [document.getElementById('website'), 2],
      [document.getElementById('email'), 3],
      [document.getElementById('password'), 4]
    ];

    elementsList.map((element) => {
      if (element[0]) {
        element[0].value = null;
        modalActiveList[element[1]] = !modalActiveList[element[1]];
      }
    });
  }
</script>

<main>
  <Modal bind:open={userModal} placement="top-center" size="md" class="w-full" style={unfocusModal}>
    <div style="display: flex;">
      <div id="account-tab" style="display:flex; justify-content:center; width:25%;">
        <h1 class="px-4 py-3 font-bold">Account</h1>
      </div>
    </div>

    <div style="display: flex; flex-direction:row">
      <div style="width:30%;display:flex;justify-content:center;">
        <div id="user-profile-circle" style="height:10vh;width:10vh;">
          <h3 style="color:white"><b>{shortName(user_info.name)}</b></h3>
        </div>
      </div>
      <div style="width:70%">
        <form class="space-y-3 px-3" style="padding-bottom: 10%;">
          <div class="space-y-3" style="padding-bottom: 5%;">
            <span class="font-bold">PROFILE INFORMATION</span>
            <div class="space-y-1" style="display:flex;flex-direction:column">
              <span class="font-bold small">Name</span>
              <span class="font-roboto small">{user_info.name}</span>
              <button
                style="text-align: left"
                on:click={() => {
                  modalActiveList[0] = !modalActiveList[0];
                }}
              >
                <span class="font-roboto small change-link-text">Change name</span>
              </button>
            </div>
            <div class="space-y-1" style="display:flex;flex-direction:column">
              <span class="font-bold small">Team</span>
              <span class="font-roboto small">{user_info.institution}</span>
              <button
                style="text-align: left"
                on:click={() => {
                  modalActiveList[1] = !modalActiveList[1];
                }}
              >
                <span class="font-roboto small change-link-text">Change Team</span>
              </button>
            </div>
            <div class="space-y-1" style="display:flex;flex-direction:column">
              <span class="font-bold small">Website</span>
              <span class="font-roboto small">{user_info.website}</span>
              <button
                style="text-align: left"
                on:click={() => {
                  modalActiveList[2] = !modalActiveList[2];
                }}
              >
                <span class="font-roboto small change-link-text">Change website</span>
              </button>
            </div>
          </div>
          <div class="space-y-3" style="padding-bottom: 5%;">
            <span class="font-bold">AUTHENTICATION</span>
            <div class="space-y-1" style="display:flex;flex-direction:column">
              <span class="font-bold small">Email</span>
              <span class="font-roboto small">{user_info.email}</span>
              <button
                style="text-align: left"
                on:click={() => {
                  modalActiveList[3] = !modalActiveList[3];
                }}
              >
                <span class="font-roboto small change-link-text">Change Email</span>
              </button>
            </div>
            <div class="space-y-1" style="display:flex;flex-direction:column">
              <span class="font-bold small">Password</span>
              <button
                style="text-align: left"
                on:click={() => {
                  modalActiveList[4] = !modalActiveList[4];
                }}
              >
                <span class="font-roboto small change-link-text">Change Password</span>
              </button>
            </div>
          </div>
          <!--
          <span class="font-bold"> CAUTION ZONE </span>
          <div class="space-y-2" style="display:flex;flex-direction:column">
            <span class="font-bold small">Account</span>
            <button
              style="text-align: left"
              on:click={() => {
                modalActiveList[5] = !modalActiveList[5];
              }}><span style="color: red;">Delete Account</span></button
            >
          </div>
          -->
        </form>
      </div>
    </div>
  </Modal>

  <Modal
    bind:open={modalActiveList[0]}
    placement="center"
    size="xs"
    class="w-full"
    style="background-color: white"
  >
    <FormControl placeholder="Jana Doe" label="Full name" id="name" required />
    <div class="space-x-3" style="display:flex;justify-content:right">
      <button
        on:click={() => {
          modalActiveList[0] = !modalActiveList[0];
        }}
      >
        Cancel
      </button>
      <Button pill={true} on:click={() => submitUserChanges()} color="dark" style="color: white">
        Confirm
      </Button>
    </div>
  </Modal>

  <Modal
    bind:open={modalActiveList[3]}
    placement="center"
    size="xs"
    class="w-full"
    style="background-color: white"
  >
    <FormControl placeholder="info@openmined.org" label="Email" id="email" required />
    <div class="space-x-3" style="display:flex;justify-content:right">
      <button
        on:click={() => {
          modalActiveList[3] = !modalActiveList[3];
        }}
      >
        Cancel
      </button>
      <Button pill={true} on:click={() => submitUserChanges()} color="dark" style="color: white">
        Confirm
      </Button>
    </div>
  </Modal>

  <Modal
    bind:open={modalActiveList[2]}
    placement="center"
    size="xs"
    class="w-full"
    style="background-color: white"
  >
    <FormControl placeholder="www.openmined.org" label="Website" id="website" required />
    <div class="space-x-3" style="display:flex;justify-content:right">
      <button
        on:click={() => {
          modalActiveList[2] = !modalActiveList[2];
        }}
      >
        Cancel
      </button>
      <Button pill={true} on:click={() => submitUserChanges()} color="dark" style="color: white">
        Confirm
      </Button>
    </div>
  </Modal>

  <Modal
    bind:open={modalActiveList[1]}
    placement="center"
    size="xs"
    class="w-full"
    style="background-color: white"
  >
    <FormControl placeholder="OpenMined Team" label="Team" id="team" required />
    <div class="space-x-3" style="display:flex;justify-content:right">
      <button
        on:click={() => {
          modalActiveList[1] = !modalActiveList[1];
        }}
      >
        Cancel
      </button>
      <Button pill={true} on:click={() => submitUserChanges()} color="dark" style="color: white">
        Confirm
      </Button>
    </div>
  </Modal>
  <Modal
    bind:open={modalActiveList[4]}
    placement="center"
    size="xs"
    class="w-full"
    style="background-color: white"
  >
    <FormControl placeholder="********" label="Password" id="password" type="password" required />
    <FormControl
      placeholder="********"
      label="Password Confirmation"
      type="password"
      id="password-confirmation"
      required
    />

    <div class="space-x-3" style="display:flex;justify-content:right">
      <button
        on:click={() => {
          modalActiveList[4] = !modalActiveList[4];
        }}
      >
        Cancel
      </button>
      <Button pill={true} on:click={() => submitUserChanges()} color="dark" style="color: white">
        Confirm
      </Button>
    </div>
  </Modal>

  <Modal
    bind:open={modalActiveList[5]}
    placement="center"
    size="xs"
    class="w-full"
    style="background-color: white"
  >
    <div style="display:flex; justify-content:center">
      <div
        style="background-color: black; border-radius: 50%; width: 30px;height:30px;align-items:center;justify-content:center;display:flex"
      >
        <YellowWarn />
      </div>
    </div>
    <h2>
      When you delete your user account all information relating to you will be deleted as well as
      any permissions and requests. If you are the datasite owner the datasite server will be deleted as
      well and will be closed to all users. To transfer ownership of a datasite server before deleting
      your account you can follow the instructions here.
    </h2>
    <div class="space-x-3" style="display:flex;justify-content:center">
      <button
        on:click={() => {
          modalActiveList[5] = !modalActiveList[5];
        }}
      >
        Cancel
      </button>
      <Button pill={true} color="red">Delete Account</Button>
    </div>
  </Modal>
</main>

<style>
  .small {
    font-size: 80%;
  }

  .change-link-text {
    color: rgb(25, 179, 230);
    cursor: pointer;
  }
  #account-tab {
    border-bottom: solid rgb(25, 179, 230);
  }

  #user-profile-circle {
    height: 5vh;
    cursor: pointer;
    width: 5vh;
    border-radius: 50%;
    margin-right: 20px;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: rgb(25, 179, 230);
  }
</style>
