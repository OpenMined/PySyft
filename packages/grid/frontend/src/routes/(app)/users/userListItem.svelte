<script>
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  import { getClient } from '$lib/store';
  import { onMount } from 'svelte';

  let client = '';

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
    dispatch('setPage', 'isDetail');
  };

  let sections = {
    isMember: true,
    isUser: false
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

  const members = [
    {
      name: 'Tamara Jones',
      email: 'tamarajones@openmined.org',
      role: 'Data Owner',
      avatar: '',
      active: 'Feb 14, 2023',
      filter: 'Recent Activity'
    },
    {
      name: 'John Doe',
      email: 'johndoe@openmined.org',
      role: 'Data Scientist',
      avatar: '',
      active: 'Feb 13, 2023',
      filter: 'Recent Activity'
    },
    {
      name: 'Tamara Jones',
      email: 'tamarajones@openmined.org',
      role: 'Admin',
      avatar: '',
      active: 'Feb 10, 2023',
      filter: 'Recent Activity'
    },
    {
      name: 'Jane Doe',
      email: 'janedoe@openmined.org',
      role: 'Data Scientist',
      avatar: '',
      active: 'Jan 10, 2023',
      filter: 'Recent Activity'
    }
  ];

  const users = [
    {
      name: 'Dr. Javier Alegre-Abarrategui',
      organisation: 'Institute of Neurology, London',
      role: 'Data Scientist',
      avatar: '',
      dataflow: 1,
      filter: 'Most dataflows'
    }
  ];
</script>

<div class="user-container w-full">
  <div
    class="user-header flex flex-col justify-start content-center items-center flex-nowrap gap-4 overflow-visible relative p-0 rounded-none"
  >
    <div class="user-illustration block shrink-0 w-[438px] h-[263px]" />
    <div class="user-buttons">
      <button
        type="button"
        class="user-button"
        class:active={sections.isMember}
        on:click={() => {
          setSection('isMember');
        }}>Team Members</button
      >
      <button
        type="button"
        class="user-button"
        class:active={sections.isUser}
        on:click={() => {
          setSection('isUser');
        }}>Users</button
      >
    </div>
  </div>
  <div class="user-content">
    <div class="user-options">
      <div class="user-search">
        <label for="search" class="search-label">
          <input
            type="search"
            class="search-input"
            name="search"
            id="search"
            placeholder="Search by name"
          />
          <span class="search-icon">&#9906;</span>
        </label>
      </div>
      <div class="user-filter">
        <div class="user-activity">
          <button type="button">Recent Activity <span class="activity-arrow">&#9660;</span></button>
        </div>
        <div class="user-count">Total: {sections.isMember ? members.length : users.length}</div>
      </div>
    </div>
    {#if sections.isMember}
      <div class="user-list">
        {#each members as member}
          <div class="hover:bg-gray-100 user-card">
            <div class="user-avatar">
              <img
                src="https://framerusercontent.com/images/kFml68vMjYxCIgVrL63SRwDEiwU.jpg"
                alt="JD"
                class="user-profile"
              />
            </div>
            <div class="user-details">
              <div class="user-identity">
                <div class="user-name">{member.name}</div>
                <div class="user-role">{member.role}</div>
              </div>
              <div class="user-email">{member.email}</div>
            </div>
            <div class="user-timestamp">Active: {member.active}</div>
          </div>
        {/each}
      </div>
    {/if}
    {#if sections.isUser}
      <div class="user-list">
        {#each users as user}
          <!-- svelte-ignore a11y-click-events-have-key-events -->
          <div class="hover:bg-gray-100 cursor-pointer user-card" on:click={setPage}>
            <div class="user-avatar">
              <img
                src="https://framerusercontent.com/images/kFml68vMjYxCIgVrL63SRwDEiwU.jpg"
                alt="JD"
                class="user-profile"
              />
            </div>
            <div class="user-details">
              <div class="user-identity">
                <div class="user-name">{user.name}</div>
                <div class="user-role">{user.role}</div>
              </div>
              <div class="user-organisation">{user.organisation}</div>
            </div>
            <div class="user-dataflows">
              <div class="user-dataflow">&rarrc; {user.dataflow}</div>
              <div class="dataflow-divider" />
              <div class="user-epsilon">0&#603;</div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  .user-container {
    width: 100%;
  }
  .user-header {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
    align-content: center;
    flex-wrap: nowrap;
    flex: 1 0 0px;
    gap: 16px;
    /* width: 100%px; */
    overflow: visible;
    position: relative;
    height: min-content;
    padding: 0px 0px 0px 0px;
    border-radius: 0px 0px 0px 0px;
  }
  .user-illustration {
    display: block;
    flex-shrink: 0;
    width: 438px;
    height: 263px;
    position: relative;
    overflow: visible;
    /* background-image: url(452660f55c3675f41b85c5d997767b8a.jpg); */
    background-color: aquamarine;
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center;
    border-radius: 0px 0px 0px 0px;
  }
  .user-buttons {
    background-color: #f1f0f4;
    border-radius: 5px;
    padding: 5px;
    margin-top: 32px;
    margin-bottom: 32px;
  }
  .user-button {
    color: #aba6be;
    min-width: 150px;
    padding: 0.5rem;
    background-color: inherit;
    border-radius: 5px;
  }
  .user-button.active {
    color: rgb(25, 179, 230);
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1), 0px 4px 16px rgba(0, 0, 0, 0.1);
    background-color: #fff;
  }
  .user-content {
    margin-top: 32px;
    margin-bottom: 32px;
  }
  .user-options {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 2rem;
    padding-bottom: 2rem;
    margin-top: 32px;
    margin-bottom: 32px;
  }
  .user-search {
    width: 50%;
  }
  .search-label {
    position: relative;
  }
  .search-input {
    background-color: #fbfbfc;
    border-radius: 50px;
    width: 100%;
  }
  .search-icon {
    position: absolute;
    color: #aba6be;
    transform: rotate(-45deg);
    right: 10px;
    top: 0;
    font-size: 2rem;
  }
  .user-filter {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 10px;
    width: 50%;
  }
  .user-activity {
    padding: 10px;
    border-radius: 5px;
    transition: all 0.5s linear;
    box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1), 0px 4px 16px rgba(0, 0, 0, 0.1);
  }
  .user-activity:hover {
    background-color: #f1f0f4;
    -webkit-transition: all 0.5s linear;
    transition: all 0.5s linear;
  }
  .user-activity:hover .activity-arrow {
    color: #aba6be;
    transition: all 0.5s linear;
  }
  .activity-arrow {
    color: #e3e1e9;
    transition: all 0.5s linear;
  }
  .user-count {
    background-color: #e3e1e9;
    padding: 3px 5px;
    border-radius: 15px;
  }
  .user-card {
    display: flex;
    box-sizing: border-box;
    flex-shrink: 0;
    width: 100%;
    height: min-content; /* 120px */
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    align-items: center;
    padding: 16px 5px 24px 5px;
    flex: 1 0 0px;
    position: relative;
    align-content: center;
    flex-wrap: wrap;
    gap: 12px;
    border-radius: 0px 0px 0px 0px;
    border-color: var(--Gray_100, #e3e1e9);
    border-style: solid;
    border-top-width: 0px;
    border-bottom-width: 1px;
    border-left-width: 0px;
    border-right-width: 0px;
  }
  .user-avatar {
    height: 10vh;
    width: 10vh;
    border-radius: 50%;
  }
  .user-profile {
    height: 10vh;
    width: 10vh;
    border-radius: 50%;
  }
  .user-details {
    flex: auto;
  }
  .user-identity {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .user-role {
    background-color: #f1f0f4;
    padding: 3px;
    border-radius: 5px;
    font-size: 0.8rem;
  }
  .user-dataflows {
    display: flex;
  }
  .dataflow-divider {
    margin-left: 10px;
    margin-right: 10px;
    flex-shrink: 0;
    width: 3px;
    height: inherit;
    display: block;
    background-color: #e3e1e9;
    overflow: hidden;
    flex: 1 0 0px;
    position: relative;
    border-radius: 0px 0px 0px 0px;
  }
</style>
