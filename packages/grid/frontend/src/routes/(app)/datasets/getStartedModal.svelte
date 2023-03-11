<script>
  import Modal from '$lib/components/DynamicModal.svelte';
  import Button from '$lib/components/Button.svelte';

  export let showModal;
</script>

<main>
  {#if showModal}
    <Modal size="large">
      <div
        class="flex flex-shrink-0 justify-between p-4 pb-0 flex-nowrap w-full h-min"
        slot="header"
      >
        <h1 style="font-size: 28px; font-weight: bold; color:black;">Get Started!</h1>
        <h2 style="font-size: 15px;">
          Begin using your domain node with one of the key actions below.
        </h2>
      </div>

      <div
        style="display:grid; grid-auto-columns: 25vh; align-items:center; height: inherit"
        slot="body"
      >
        <div
          style="cursor: pointer;grid-column: span 1; grid-row: 1 / span 1;margin: 1vh; border-width: 1px; border-color: #DDDDD1;	box-shadow: 2px 2px 9px #888888; height: 50vh;"
        >
          <img
            style="height: 20vh; width: 26vh;"
            alt="upload-dataset.jpg"
            src="/assets/card_image.jpg"
          />
          <h1
            style="margin-left: 7px;margin-top: 7px;color: black; font-size: 10; font-weight: bold"
          >
            Upload a Dataset
          </h1>
          <h2
            style="font-size: 15px;margin-left: 7px;margin-right: 7px;margin-top: 3px;margin-bottom: 5px;"
          >
            Create a catalogue and begin uploading data so that your team and outside researchers
            can begin making impact.
          </h2>
        </div>
        <div
          style="cursor: pointer;grid-column: 2 / span 1; grid-row: 1 / span 1;margin: 1vh;border-width: 1px; border-color: #DDDDDD;box-shadow: 2px 2px 9px #888888; height: 50vh;"
        >
          <img
            style="height: 20vh; width: 26vh;"
            alt="create-team-member.jpg"
            src="/assets/card_image.jpg"
          />
          <h1
            style="margin-left: 7px;margin-top: 7px;color: black; font-size: 10; font-weight: bold"
          >
            Create Team Member
          </h1>
          <h2
            style="font-size: 15px;margin-left: 7px;margin-right: 7px;margin-top: 3px;margin-bottom: 5px;"
          >
            Create accounts for team members so that they can help research, maintain, and
            contribute to your domain node.
          </h2>
        </div>
        <div
          style="cursor: pointer;grid-column: 3 / span 1; grid-row: 1 / span 1;margin: 1vh;border-width: 1px; border-color: #DDDDDD; box-shadow: 2px 2px 9px #888888; height: 50vh;"
        >
          <img
            style="height: 20vh; width: 26vh;"
            alt="create-user.jpg"
            src="/assets/card_image.jpg"
          />
          <h1
            style="margin-left: 7px;margin-top: 7px;color: black; font-size: 10; font-weight: bold"
          >
            Create General User
          </h1>
          <h2
            style="font-size: 15px;margin-left: 7px;margin-right: 7px;margin-top: 3px;margin-bottom: 5px;"
          >
            Create accounts for outside researchers to safely study from your data without needing a
            copy or direct access.
          </h2>
        </div>
      </div>
      <div slot="footer">
        <h2 style="font-size: 15px; text-align: center; padding-bottom: 12vh">
          Learn more about how to use your domain node by reading our "Data Owner" handbook.
        </h2>
        <Button action={() => (showModal = false)} variant="secondary" position="right"
          >Cancel</Button
        >
      </div>
    </Modal>
  {/if}
</main>
