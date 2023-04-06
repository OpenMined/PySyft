<script>
  import Info from '$lib/components/icons/Info.svelte';
  import { createEventDispatcher } from 'svelte';

  import { MapRoles } from '$lib/utils';

  const dispatch = createEventDispatcher();
  import { getClient } from '$lib/store';
  import { onMount } from 'svelte';

  let client = '';
  export let selectedUser;


  onMount(async () => {
    await getClient()
      .then((response) => {
        client = response;
      })
      .catch((error) => {
        console.log(error);
      });
  });

  const setPage = async () => {
    dispatch('setPage', 'isList');
  };

  let sections = {
    isActivity: true,
    isMessage: false
  };

  const setSection = async (current) => {
    if (sections[current]) return;
    else {
      for (let section in sections) {
        sections[section] = section === current ? true : false;
      }
    }
  };

  const filters = ['Recent Activity', 'Most dataflows'];

</script>

<div class="user-container">
  <div class="user-control">
    <button type="button" class="contol-button" on:click={setPage}>&#10094; Back</button>
  </div>
  <!-- <div class="user-header"> -->
  <div class="user-profile">
    <div class="user-avatar">
      <img
        src="https://framerusercontent.com/images/kFml68vMjYxCIgVrL63SRwDEiwU.jpg"
        alt="JD"
        class="user-image"
      />
    </div>
    <div class="user-name">{selectedUser.name}</div>
    <div class="user-organisation">{(selectedUser.institution) ? selectedUser.institution: ''}</div>
    <div class="user-roles">
      <div class="user-role">
          <div class="role">{MapRoles(selectedUser.role.value)}</div>
      </div>
    </div>
  </div>
  <hr style="width: 100%" />
  <div class="user-property">
    <!-- <div>
      <hr>
    </div>
    <hr /> -->
    <div class="user-privacy">
      <h3 style="font-weight: 700;">PRIVACY LIMIT</h3>
      <div style="display: flex; gap: 10px;">0&#603; <Info /></div>
    </div>
    <div class="property-divider" />
    <div class="user-contact">
      <h3 style="font-weight: 700;">CONTACT</h3>
      <div>
        <a href="mailto:{selectedUser.email}">{selectedUser.email}</a>
      </div>
      <div>
        <a href="http://www.google.scholar.com/995749874597-abarrategui"
          >{(selectedUser.website) ? selectedUser.website: ''}</a
        >
      </div>
    </div>
  </div>
  <hr style="width: 100%; margin-bottom: 24px;" />
  <div class="user-buttons">
    <button
      type="button"
      class="user-button"
      class:active={sections.isActivity}
      on:click={() => {
        setSection('isActivity');
      }}>Activity</button
    >
    <button
      type="button"
      class="user-button"
      class:active={sections.isMessage}
      on:click={() => {
        setSection('isMessage');
      }}>Messages</button
    >
  </div>

  <!-- </div> -->
  <div class="user-content">
    <!-- <div class="user-options">
      <div class="user-search">
        <label for="search" class="search-label">
          <input type="search" class="search-input" name="search" id="search" placeholder="Search by name">
          <span class="search-icon">&#9906;</span>
        </label>
      </div>
      <div class="user-filter">
        <div class="user-activity">
          <button type="button">Recent Activity <span class="activity-arrow">&#9660;</span></button>
        </div>
        <div class="user-count">Total: 3</div>
      </div>
    </div> -->
    {#if sections.isActivity}
      <div class="user-activities">
        <div class="user-activity">
          <div class="activity-check flex justify-center">
            <div
              class="flex items-center justify-center w-[60px] h-[60px] mb-12 rounded-full bg-gray-200"
            >
              <label for="toggleB" class="flex items-center cursor-pointer">
                <!-- toggle -->
                <div class="relative">
                  <!-- input -->
                  <input type="checkbox" id="toggleB" class="sr-only" />
                  <!-- line -->
                  <div class="block bg-black-900 w-12 h-6 rounded-full" />
                  <!-- dot -->
                  <div
                    class="dot absolute left-1 top-1 bg-gray-200 w-4 h-4 rounded-full transition"
                  />
                </div>
                <!-- label -->
                <!-- <div class="ml-3 text-gray-700 font-medium">
                  Toggle Me!
                </div> -->
              </label>
            </div>
          </div>
          <div class="activity-detail">
            <div class="activity-date">Feb 14, 2023</div>
            <div class="activity-request">
              would like permission to view asset "Media Sources Daily Audience".
            </div>
            <div class="activity-response">
              <div class="response-details">
                <div class="response-status">Denied</div>
                <div class="response-separator">&bull;</div>
                <div class="response-user">Jane Doe</div>
              </div>
              <div class="response-text">
                Would be a breach of privacy to give Dr. Javier view access
              </div>
            </div>
          </div>
        </div>
        <div class="user-activity">
          <div class="activity-check flex justify-center">
            <div
              class="flex items-center justify-center w-[60px] h-[60px] mb-12 rounded-full bg-gray-200"
            >
              <label for="toggleB" class="flex items-center cursor-pointer">
                <!-- toggle -->
                <div class="relative">
                  <!-- input -->
                  <input type="checkbox" id="toggleB" class="sr-only" />
                  <!-- line -->
                  <div class="block bg-black-900 w-12 h-6 rounded-full" />
                  <!-- dot -->
                  <div
                    class="dot absolute left-1 top-1 bg-gray-200 w-4 h-4 rounded-full transition"
                  />
                </div>
                <!-- label -->
                <!-- <div class="ml-3 text-gray-700 font-medium">
                  Toggle Me!
                </div> -->
              </label>
            </div>
          </div>
          <div class="activity-detail">
            <div class="activity-date">Feb 11, 2023</div>
            <div class="activity-request">
              would like permission to view asset "Media Sources Daily Audience".
            </div>
            <div class="activity-response">
              <div class="response-details">
                <div class="response-status">Denied</div>
                <div class="response-separator">&bull;</div>
                <div class="response-user">Jane Doe</div>
              </div>
              <div class="response-text">
                Would be a breach of privacy to give Dr. Javier view access
              </div>
            </div>
          </div>
        </div>
      </div>
    {/if}
    {#if sections.isMessage}
      <div class="user-messages">
        <div class="user-message">
          <div class="message-properties">
            <div class="message-avatar">
              <img
                src="https://framerusercontent.com/images/kFml68vMjYxCIgVrL63SRwDEiwU.jpg"
                alt="JD"
                class="message-profile"
              />
            </div>
            <div class="message-author">Jane Doe</div>
            <div class="message-separator">&bull;</div>
            <div class="message-date">Yesterday</div>
          </div>
          <div class="message-text">
            Dr. Alegre-Abarrategui, unfortunately I cannot give you view access to this dataset as
            that would be a breach of privacy for our users. However, we would be open to reviewing
            your project via a code or project submission. This would allow you to do some
            computations on the data without needing view access and would allow us to protect the
            privacy of our users.
          </div>
          <div class="message-request">98798798797897978787...</div>
        </div>
        <div class="message-box">
          <form class="message-form" action="">
            <div class="form-item">
              <label for="message">
                <input
                  type="text"
                  style="width: 100%; border-radius: 5px; background-color: #f1f0f4;"
                  name="message"
                  id="message"
                  placeholder="Write message..."
                />
              </label>
            </div>
            <div class="form-item" style="align-self: flex-end;">
              <button type="submit">Send &#10148;</button>
            </div>
          </form>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .user-container {
    display: flex;
    flex-direction: column;
    width: 100%;
    position: relative;
  }
  .user-control {
    /* align-self: start; */
    margin-top: 16px;
    margin-bottom: 16px;
  }
  .contol-button {
    color: rgb(25, 179, 230);
  }
  /* .user-header {
    display: flex;
    flex-direction: column;
    width: 100%;
    gap: 16px;
    border-radius: 0px 0px 0px 0px;
  } */
  .user-profile {
    display: flex;
    flex-direction: column;
    width: 100%;
    height: auto;
    margin-top: 16px;
    margin-bottom: 16px;
    /* border-radius: 50%; */
  }
  .user-avatar {
    align-self: center;
    height: 10vh;
    width: 10vh;
    border-radius: 50%;
    margin-bottom: 8px;
  }
  .user-image {
    height: 10vh;
    width: 10vh;
    border-radius: 50%;
  }
  .user-name {
    align-self: center;
    margin-top: 8px;
    margin-bottom: 8px;
    flex-shrink: 0;
    width: 100%;
    height: auto; /* 40px */
    white-space: pre-wrap;
    word-wrap: break-word;
    word-break: break-word;
    flex: 1 0 0px;
    position: relative;
    line-height: 1.2;
    font-size: 32px;
    font-weight: 700;
    text-align: center;
  }
  .user-organisation {
    color: #8a859c;
    font-size: 20px;
    align-self: center;
    margin-top: 8px;
    margin-bottom: 8px;
  }
  .user-roles {
    align-self: center;
    /* background-color: #e3e1e9; */
    /* padding: 3px; */
    /* border-radius: 5px; */
    /* font-size: 16px; */
    margin-top: 8px;
    margin-bottom: 8px;
  }
  .user-role {
    display: flex;
    gap: 10px;
  }
  .role {
    /* align-self: center; */
    background-color: #e3e1e9;
    padding: 3px;
    border-radius: 5px;
    font-size: 16px;
    /* margin-top: 8px; */
    /* margin-bottom: 8px; */
  }
  .user-property {
    display: flex;
    justify-content: space-between;
    font-size: 20px;
    margin-top: 16px;
    margin-bottom: 16px;
  }
  .user-privacy,
  .user-contact {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 45%;
    gap: 10px;
  }
  .property-divider {
    margin-left: 10px;
    margin-right: 10px;
    flex-shrink: 0;
    width: 2px;
    height: inherit;
    display: block;
    background-color: #e3e1e9;
    overflow: hidden;
    /* flex: 1 0 0px; */
    position: relative;
    border-radius: 0px 0px 0px 0px;
  }
  .user-buttons {
    align-self: center;
    background-color: #f1f0f4;
    border-radius: 5px;
    padding: 0.2rem;
    margin-top: 32px;
    margin-bottom: 32px;
    /* width: 100%; */
  }
  .user-button {
    min-width: 150px;
    padding: 0.5rem;
    background-color: inherit;
    color: #aba6be;
    border-radius: 5px;
    /* opacity: 0.5; */
    /* box-shadow: 0px 2px 6px rgba(0,0,0,0.1), 0px 4px 16px rgba(0,0,0,0.1); */
  }
  .user-button.active {
    background-color: #fff;
    color: rgb(25, 179, 230);
    /* opacity: 1; */
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1), 0px 4px 16px rgba(0, 0, 0, 0.1);
  }
  .user-content {
    margin-top: 32px;
    margin-bottom: 32px;
  }
  .user-activity {
    display: flex;
    gap: 10px;
    padding-bottom: 32px;
    margin-top: 32px;
    margin-bottom: 32px;
    border-color: var(--Gray_100, #e3e1e9);
    border-style: solid;
    border-top-width: 0px;
    border-bottom-width: 1px;
    border-left-width: 0px;
    border-right-width: 0px;
  }
  .activity-check {
    width: auto;
  }
  input:checked ~ .dot {
    transform: translateX(1.5rem);
    background-color: #48bb78;
  }
  .activity-detail {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .activity-response {
    position: relative;
    padding-left: 15px;
  }
  .activity-response:before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    /* border-top-left-radius: 4px; */
    /* border-bottom-left-radius: 4px; */
    background-color: #d22d80;
  }
  .response-details {
    display: flex;
    gap: 10px;
  }
  .response-status {
    background-color: #e481b3;
    border-radius: 10px;
    padding: 0 5px;
  }
  .response-separator,
  .message-separator {
    color: #c7c4d3;
    font-size: 24px;
  }
  /* messages */
  .user-message {
    display: flex;
    flex-direction: column;
    gap: 10px;
    width: 70%;
    margin: 0 auto;
    margin-top: 32px;
    margin-bottom: 32px;
    padding-bottom: 32px;
    border-color: var(--Gray_100, #e3e1e9);
    border-style: solid;
    border-top-width: 0px;
    border-bottom-width: 1px;
    border-left-width: 0px;
    border-right-width: 0px;
  }
  .message-properties {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 15px;
  }
  .message-avatar {
    height: 10vh;
    width: 10vh;
    border-radius: 50%;
  }
  .message-profile {
    height: 10vh;
    width: 10vh;
    border-radius: 50%;
  }
  .message-box {
    width: 70%;
    padding: 16px;
    margin: 0 auto;
    margin-top: 32px;
    margin-bottom: 32px;
    box-shadow: 0px 2px 4px 0px rgba(0, 0, 0, 0.15);
  }
  .message-form {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
</style>
