<script>
	import '../../app.postcss';
	import { goto } from '$app/navigation';
	import OnBoardModal from '../../components/onBoardModal.svelte';
	import Sidebar from '../../components/Sidebar.svelte';
	import Navbar from '../../components/Navbar.svelte';
	import { store } from '../../lib/store.js';

	let client;
	let activeUrl = '/home';

	$: metadata = '';
	$: user_info = '';

	async function loadGlobalInfos() {
		// Load JSSerde from local Storage
		store.subscribe(async (value) => {
			if (value) {
				client  = value.client;
			} else{
				goto('/login')
			}
		});

		// Load metadata from session Storage
		metadata = JSON.parse(window.sessionStorage.getItem('metadata'));
		if (!metadata){
			metadata = await client.metadata;
			window.sessionStorage.setItem('metadata', JSON.stringify(metadata))
		}

		user_info = JSON.parse(window.sessionStorage.getItem('user_info'));
		if (!user_info){
			user_info = await client.user
			window.sessionStorage.setItem('user_info', JSON.stringify(user_info))
		}
	}
</script>

<main>
	{#await loadGlobalInfos() then none}
		<Navbar bind:user_info />
		<Sidebar bind:activeUrl bind:metadata bind:user_info />
		<OnBoardModal
			client={client}
			bind:user_info
			bind:metadata
		/>
	{/await}
</main>
<svelte:window />
<slot />
