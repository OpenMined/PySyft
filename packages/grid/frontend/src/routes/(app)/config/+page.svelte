<script>
  import { shortName, prettyName } from '$lib/utils.js';
  import { Modal, Button } from 'flowbite-svelte';
  import FormControl from '$lib/components/FormControl.svelte';
  import { getClient } from '$lib/store.js';
  import Badge from '$lib/components/Badge.svelte';
  import { onMount } from 'svelte';

  let client = '';
  let metadata = '';

  onMount(async () => {
    await getClient()
      .then((response) => {
        client = response;

        client.metadata.then((responseMetadata) => {
          metadata = responseMetadata;
        });
      })
      .catch((error) => {
        console.log(error);
      });
  });

  let modalActiveList = [false, false, false];
  let selectedTab = ['border-bottom: solid rgb(25, 179, 230)', '', ''];
  let display = ['block', 'none', 'none'];
  let focus = ['font-bold', 'font-roboto', 'font-roboto'];

  function updateSelectedBar(index) {
    selectedTab.map((element, i) => {
      if (i === index) {
        selectedTab[i] = 'border-bottom: solid rgb(25, 179, 230)';
        focus[i] = 'font-bold';
        display[i] = 'block';
      } else {
        selectedTab[i] = '';
        focus[i] = 'font-roboto';
        display[i] = 'none';
      }
    });
  }

  async function submitDomainChanges(client) {
    let name = document.getElementById('domain_name')
      ? document.getElementById('domain_name').value
      : null;
    let organization = document.getElementById('organization')
      ? document.getElementById('organization').value
      : null;
    let description = document.getElementById('description')
      ? document.getElementById('description').value
      : null;

    let domainInfo = {
      name: name,
      organization: organization,
      description: description
    };

    // Filter attributes that doesn't exist
    Object.keys(domainInfo).forEach((k) => domainInfo[k] == null && delete domainInfo[k]);

    await client.updateMetadata(domainInfo);
    metadata = await client.metadata;

    // Cleaning Element Values
    const elementsList = [
      [document.getElementById('domain_name'), 0],
      [document.getElementById('organization'), 1],
      [document.getElementById('description'), 2]
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
  {#if metadata}
    <div class="page-container">
      <div class="flex space-x-7 px-4" style="height: 30px;">
        <button
          on:click={() => {
            updateSelectedBar(0);
          }}
          style={selectedTab[0]}
          class="flex w-20 justify-center cursor-pointer"
        >
          <span class={focus[0]}>Domain</span>
        </button>
        <!--
        <button
          on:click={() => {
            updateSelectedBar(1);
          }}
          style={selectedTab[1]}
          class="flex w-20 justify-center cursor-pointer"
        >
          <span class={focus[1]}> Connection </span>
        </button>
        <button
          on:click={() => {
            updateSelectedBar(2);
          }}
          style={selectedTab[2]}
          class="flex w-20 justify-center cursor-pointer"
        >
          <span class={focus[2]}> Permissions </span>
        </button>
        -->
      </div>

      <div class="px-6">
        <div style="display:{display[0]}">
          <div class="grid grid-cols-6 gap-3">
            <div class="flex justify-center py-10">
              <div>
                <div id="domain-profile-border">
                  <div id="domain-profile-circle">
                    <h3 style="color:white">
                      <b>{shortName(prettyName(metadata.name))}</b>
                    </h3>
                  </div>
                </div>
              </div>
            </div>

            <div class="border-b col-span-5 space-y-6 py-12">
              <span class="font-bold">PROFILE INFORMATION</span>
              <div class="flex flex-col space-y-2">
                <span class="font-bold">Domain Name</span>
                <span class="font-roboto">{metadata.name}</span>
                <button
                  on:click={() => {
                    modalActiveList[0] = !modalActiveList[0];
                  }}
                  style="text-align: left"
                >
                  <span class="font-roboto change-link-text ">Change Domain Name</span>
                </button>
              </div>

              <div class="flex flex-col space-y-2">
                <span class="font-bold">Organization</span>
                <span class="font-roboto">{metadata.organization}</span>
                <button
                  on:click={() => {
                    modalActiveList[1] = !modalActiveList[1];
                  }}
                  style="text-align: left"
                >
                  <span class="font-roboto change-link-text ">Change Organization</span>
                </button>
              </div>

              <div class="flex flex-col space-y-2">
                <span class="font-bold">Description</span>
                <span class="font-roboto">{metadata.description}</span>
                <button
                  on:click={() => {
                    modalActiveList[2] = !modalActiveList[2];
                  }}
                  style="text-align: left;"
                >
                  <span class="font-roboto change-link-text ">Change Description</span>
                </button>
              </div>
            </div>
            <div class="col-start-2 col-span-5">
              <span class="font-bold">SYSTEM INFORMATION</span>
              <div class="grid grid-cols-4 py-5">
                <!--
                <div class="flex justify-center">
                  <div id="user-profile-circle" style="height:10vh;width:10vh;">
                    <h3 style="color:white">
                      <b> {shortName('None')} </b>
                    </h3>
                  </div>
                  <div style="justify-content:center;display:flex;flex-direction:column">
                    <span class="font-bold">{'None'}</span>
                    <span class="font-roboto">Domain Owner</span>
                  </div>
                </div>
                -->
                <div class="flex flex-col justify-center">
                  <span class="font-bold">
                    ID # <Badge variant="gray">{metadata.id.value}</Badge>
                  </span>
                  <span>
                    <b>DEPLOYED ON:</b>
                    {metadata.deployed_on}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div style="display:{display[1]}" />
        <div style="display:{display[2]}" />
      </div>
    </div>

    <Modal
      bind:open={modalActiveList[0]}
      placement="center"
      size="xs"
      class="w-full"
      style="background-color: white"
    >
      <FormControl placeholder="USA Domain" label="Domain Name" id="domain_name" required />
      <div class="space-x-3" style="display:flex;justify-content:right">
        <button
          on:click={() => {
            modalActiveList[0] = !modalActiveList[0];
          }}
        >
          Cancel
        </button>
        <Button
          pill={true}
          on:click={() => submitDomainChanges(client)}
          color="dark"
          style="color: white"
        >
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
      <FormControl placeholder="UCSF University" label="Organization" id="organization" required />
      <div class="space-x-3" style="display:flex;justify-content:right">
        <button
          on:click={() => {
            modalActiveList[0] = !modalActiveList[0];
          }}
        >
          Cancel
        </button>
        <Button
          pill={true}
          on:click={() => submitDomainChanges(client)}
          color="dark"
          style="color: white"
        >
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
      <FormControl
        placeholder="This domain was created for research purposes"
        label="Description"
        id="description"
        type="textarea"
        required
      />
      <div class="space-x-3" style="display:flex;justify-content:right">
        <button
          on:click={() => {
            modalActiveList[0] = !modalActiveList[0];
          }}
        >
          Cancel
        </button>
        <Button
          pill={true}
          on:click={() => submitDomainChanges(client)}
          color="dark"
          style="color: white"
        >
          Confirm
        </Button>
      </div>
    </Modal>
  {/if}
</main>

<style>
  .page-container {
    width: 85%;
    position: absolute;
    height: 93%;
    top: 7%;
    left: 15%;
  }

  #domain-profile-circle {
    height: 15vh;
    width: 15vh;
    border-width: 2px;
    border-color: whitesmoke;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #222222;
  }

  #domain-profile-border {
    padding: 1rem;
    position: relative;
    background: linear-gradient(to bottom, rgb(37, 169, 246), red);
    padding: 3px;
    border-radius: 50%;
  }

  .change-link-text {
    color: rgb(25, 179, 230);
    cursor: pointer;
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
