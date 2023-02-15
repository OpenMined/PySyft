<script>
	import LoginHeader from '../../components/LoginHeader.svelte';
	import RegisterModal from '../../components/RegisterModal.svelte';
	import { Input, Label, Helper, Button } from 'flowbite-svelte';
	import { getSerde, store } from '../../lib/store.js';
	import { prettyName } from '../../lib/utils.js';
	import { goto } from '$app/navigation';
	let email = '';
	let password = '';
	$: formModal = false;
	let inputColor = 'base';
	let displayError = 'none';
	let localStore;

	store.subscribe((value) => {
		localStore = value;
	});

	async function loadScreenInfo(serde) {
		return fetch('http://localhost:8081/api/v1/syft/metadata')
			.then((response) => response.arrayBuffer())
			.then(function (response) {
				let metadata = serde.deserialize(response);

				let nodeAddrObj = {};
				metadata.get('id').forEach((value, key) => {
					nodeAddrObj[key] = value;
				});

				let metadataObj = {};
				metadata.forEach((value, key) => {
					metadataObj[key] = value;
				});

				metadataObj.id = nodeAddrObj;
				window.sessionStorage.setItem('metadata', JSON.stringify(metadataObj));
				return metadata;
			});
	}

	async function login(email, password, metadata) {
		await fetch('http://localhost:8081/api/v1/login', {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ email: email, password: password })
		}).then((response) => {
			if (response.status === 401) {
				inputColor = 'red';
				displayError = 'block';
			} else {
				response.json().then((body) => {
					window.sessionStorage.setItem('token', 'Bearer ' + body['access_token']);
					localStore.credentials = body['access_token'];
					store.set(localStore);
					goto('/home');
				});
			}
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
	{#await getSerde() then jsserde}
		{#await loadScreenInfo(jsserde) then metadata}
			<!-- Login Screen Header -->
			<LoginHeader version={metadata.get('version')} />

			<!-- Login Screen Body -->
			<div id="login-screen">
				<!-- Background Circles -->
				<div id="login-orange-circle-bg" />
				<div id="login-blue-circle-bg" />

				<!-- Register Modal -->
				<RegisterModal bind:formModal {jsserde} nodeId={metadata.get('id').get('value')} />

				<!-- Login Window -->
				<div id="login-window">
					<div class="space-y-3" style="margin-top: 5%;">
						<h1 style="font-size:20px; text-align:center"><b> Welcome </b></h1>
						<h3 style="text-align:center;"><span class="dot" />Domain Online</h3>
						<Label class="space-y-2" style="margin-top: 50px;width:60vh;">
							<span>Email</span>
							<Input
								color={inputColor}
								type="email"
								bind:value={email}
								placeholder="info@openmined.org"
								size="md"
							>
								<svg
									slot="left"
									aria-hidden="true"
									class="w-4 h-4"
									fill="currentColor"
									viewBox="0 0 20 20"
									xmlns="http://www.w3.org/2000/svg"
									><path
										d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z"
									/><path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" /></svg
								>
							</Input>
						</Label>
						<Label class="space-y-2" style="width:60vh;">
							<span>Password</span>
							<Input
								color={inputColor}
								type="password"
								placeholder="**********"
								bind:value={password}
								size="md"
							>
								<svg
									slot="left"
									width="24"
									height="24"
									xmlns="http://www.w3.org/2000/svg"
									fill-rule="evenodd"
									clip-rule="evenodd"
									><path
										d="M24 11.5c0 3.613-2.951 6.5-6.475 6.5-2.154 0-4.101-1.214-5.338-3h-2.882l-1.046-1.013-1.302 1.019-1.362-1.075-1.407 1.081-4.188-3.448 3.346-3.564h8.841c1.145-1.683 3.104-3 5.339-3 3.497 0 6.474 2.866 6.474 6.5zm-10.691 1.5c.98 1.671 2.277 3 4.217 3 2.412 0 4.474-1.986 4.474-4.5 0-2.498-2.044-4.5-4.479-4.5-2.055 0-3.292 1.433-4.212 3h-9.097l-1.293 1.376 1.312 1.081 1.38-1.061 1.351 1.066 1.437-1.123 1.715 1.661h3.195zm5.691-3.125c.828 0 1.5.672 1.5 1.5s-.672 1.5-1.5 1.5-1.5-.672-1.5-1.5.672-1.5 1.5-1.5z"
									/></svg
								>
							</Input>
							<Helper
								color="red"
								class="text-sm"
								style="text-align: center;display: {displayError}"
							>
								Incorrect email or password
							</Helper>
							<Helper class="text-sm" style="text-align: center"
								>Don't you have an account yet? Apply for an account <a
									on:click={() => {
										formModal = true;
									}}
									class="font-medium text-blue-600 hover:underline dark:text-blue-500">here</a
								>.</Helper
							>
						</Label>
						<div style="display:flex; justify-content:center">
							<Button
								on:click={() => login(email, password, metadata)}
								style="width: 30vh;"
								color="dark">Login</Button
							>
						</div>
					</div>
				</div>

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
						<p
							id="gridUID"
							on:click={() => copyToClipBoard()}
							style="margin-left: 5px; color: black;padding-left:10px; padding-right:10px; background-color: #DDDDDD"
						>
							{metadata.get('id').get('value')}
						</p>
					</h3>
					<h3 class="info-foot"><b> DEPLOYED ON:&nbsp;&nbsp;</b> {metadata.get('deployed_on')}</h3>
				</div>
			</div>

			<a href="https://www.openmined.org/">
				<div id="login-footer">
					<h3>Empowered by</h3>
					<img
						style="margin-left: 10px; margin-right: 8vh;"
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

	#login-window {
		border-radius: 5px;
		justify-content: center;
		display: flex;
		padding: 10px;
		box-shadow: 5px 10px 18px #888888;
		position: absolute;
		top: 24%;
		left: 50%;
		width: 40%;
		height: 55%;
		z-index: 2;
		background-color: white;
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
