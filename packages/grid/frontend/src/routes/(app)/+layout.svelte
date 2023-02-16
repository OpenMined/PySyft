<script>
	import '../../app.postcss';
	import { goto } from '$app/navigation';
	import { Input, Label, Modal, Textarea, Button, Helper } from 'flowbite-svelte';
	import OnBoardModal from '../../components/onBoardModal.svelte';
	import Sidebar from '../../components/Sidebar.svelte';
	import Navbar from '../../components/Navbar.svelte';
	import { store, getSerde } from '../../lib/store.js';

	let jsSerde;
	let activeUrl = '/home';

	$: metadata = '';
	$: user_info = '';

	async function loadGlobalInfos(url) {
		// Load metadata from session Storage
		metadata = JSON.parse(window.sessionStorage.getItem('metadata'));

		// Load JSSerde from local Storage
		store.subscribe((value) => {
			if (!value.jsserde) {
				jsSerde = getSerde();
			} else {
				jsSerde = value.jsserde;
			}
		});

		// Get current user info from users/me API
		await fetch(url, {
			method: 'GET',
			headers: { Authorization: window.sessionStorage.getItem('token') }
		}).then((response) => {
			if (response.status === 401) {
				goto('/login');
			} else {
				return response.json().then((body) => {
					user_info = body;
				});
			}
		});
	}
</script>

<main>
	{#await loadGlobalInfos('http://localhost:8081/api/v1/users/me') then none}
		<Navbar bind:user_info />
		<Sidebar bind:activeUrl bind:metadata bind:user_info />
		<OnBoardModal
			token={window.sessionStorage.getItem('token')}
			jsserde={jsSerde}
			bind:user_info
			bind:metadata
		/>
	{/await}
</main>
<svelte:window />
<slot />
