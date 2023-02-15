<script>
	import { Input, Label, Modal, Textarea, Button, Helper } from 'flowbite-svelte';
	import { SyftMessageWithoutReply } from '../lib/jsserde/objects/syftMessage.ts';
	import { Progressbar } from 'flowbite-svelte';
	export let onBoardModal = false;
	export let user_info;
	export let metadata;
	export let client;

	let domainInfo = {
		domain_name: '',
		organization: '',
		description: ''
	};

	let userInfo = {
		user_id: '',
		email: '',
		password: '',
		name: '',
		institution: '',
		website: ''
	};

	let steps = ['inline', 'none', 'none', 'none'];
	let stepIndex = 0;

	function nextStep() {
		steps[stepIndex] = 'none';
		steps[stepIndex + 1] = 'inline';
		stepIndex = stepIndex + 1;
	}

	function previousStep() {
		steps[stepIndex] = 'none';
		steps[stepIndex - 1] = 'inline';
		stepIndex = stepIndex - 1;
	}

	function checkOnBoard() {
		if (user_info.role === 'Owner' && !metadata['on_board']) {
			setTimeout(() => {
				onBoardModal = true;
			}, 1000);
		}
	}

	async function submitChanges() {
		// Set Domain name, organization and description
		await client.updateConfigs(domainInfo)

		userInfo.user_id = user_info.id;
		await client.updateUser(userInfo)

		// Update layout metadata variable
		await client.metadata
		metadata = JSON.parse(window.sessionStorage.getItem('metadata'))
		
		// Update user info
		user_info = await client.user

		nextStep();
	}

	checkOnBoard();
</script>

<main>
	<Modal bind:open={onBoardModal} placement="top-center" size="md" class="w-full">
		<!-- Modal First step -->
		<div class="space-y-9" style="display: {steps[0]}">
			<div style="display:flex; justify-content: center; align-items:center">
				<!-- Header -->
				<div id="onboard-square" />
			</div>
			<div style="display:flex; justify-content: center;">
				<h1 color="black"><b> Welcome to PyGrid Admin!</b></h1>
			</div>
			<Progressbar size="h-1.5" progress="25" />

			<h2>
				Congratulations on deploying {metadata['name']} node. This wizard will help get you started in
				setting up your domain node and user account. You can skip this wizard by pressing "Cancel" below.
				You can edit any of your responses later by going to "Domain Settings" indicated in the gear
				icon in the top left corner of your navigation or by going to "Account Settings" indicated by
				your avatar in the top right corner of the navigation.
			</h2>
			<h2>Click "Next" to begin.</h2>

			<div style="display:flex; justify-content: right">
				<div class="cancel-button" on:click={(onBoardModal = false)}><h1>Cancel</h1></div>

				<Button pill={true} on:click={() => nextStep()} color="dark">Next</Button>
			</div>
		</div>
		<!-- Modal Second Step -->
		<div class="space-y-4" style="display:{steps[1]}; justify-content: center; align-items:center">
			<div style="display:flex;justify-content:center;align-items:center">
				<div
					style="background-color: rgb(59 130 246); border-radius:50%; height: 10vh; width: 10vh; display:flex;justify-content:center;align-items:center;"
				>
					<svg width="48" height="48" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"
						><path
							d="m21 7.702-8.5 4.62v9.678c1.567-.865 6.379-3.517 7.977-4.399.323-.177.523-.519.523-.891zm-9.5 4.619-8.5-4.722v9.006c0 .37.197.708.514.887 1.59.898 6.416 3.623 7.986 4.508zm-8.079-5.629 8.579 4.763 8.672-4.713s-6.631-3.738-8.186-4.614c-.151-.085-.319-.128-.486-.128-.168 0-.335.043-.486.128-1.555.876-8.093 4.564-8.093 4.564z"
							fill-rule="nonzero"
						/></svg
					>
				</div>
			</div>
			<div style="display:flex; justify-content: center;">
				<h1 style="color:black; text-align:center"><b> Domain Profile</b></h1>
			</div>
			<div style="display:flex; justify-content: center;">
				<h1 style="color: rgb(59 130 246); text-align:center"><b> Steps 2 of 4</b></h1>
			</div>
			<Progressbar size="h-1.5" progress="50" />
			<h2>
				Let's begin by describing some basic information about this domain node. This information
				will be shown to outside users to help them find and understand what your domain offers.
			</h2>

			<form class="flex flex-col space-y-6" action="#">
				<Label class="space-y-2">
					<Label class="block mb-2"
						>Domain Name <span style="color: rgb(59 130 246);">*</span></Label
					>
					<Input
						bind:value={domainInfo.domain_name}
						type="text"
						name="name"
						placeholder=" Oxford Parkinson's Disease Center"
						required
					/>
				</Label>

				<Label class="space-y-2">
					<Label class="block mb-2">Organization</Label>
					<Input
						bind:value={domainInfo.organization}
						type="text"
						name="name"
						placeholder="ABC University"
						required
					/>
				</Label>

				<Label class="space-y-2">
					<Label class="block mb-2">Description</Label>
					<Textarea
						bind:value={domainInfo.description}
						style="background-color: #f9fafb"
						id="textarea-id"
						placeholder="Domain description"
						rows="4"
						name="message"
					/>
				</Label>
			</form>

			<div style="display:flex; justify-content: space-between;">
				<Button pill={true} on:click={() => previousStep()} color="dark">Back</Button>
				<div style="display: flex">
					<div class="cancel-button"><h1>Cancel</h1></div>
					<Button
						disabled={domainInfo.domain_name === ''}
						pill={true}
						on:click={() => nextStep()}
						color="dark">Next</Button
					>
				</div>
			</div>
		</div>

		<!-- Modal Third Step -->
		<div class="space-y-4" style="display:{steps[2]};">
			<div style="display:flex;justify-content:center;align-items:center">
				<div
					style="background-color: rgb(59 130 246); border-radius:50%; height: 10vh; width: 10vh; display:flex;justify-content:center;align-items:center;"
				>
					<svg width="48" height="48" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"
						><path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
						/></svg
					>
				</div>
			</div>
			<div style="display:flex; justify-content: center;">
				<h1 style="color:black; text-align:center"><b> User Account</b></h1>
			</div>
			<div style="display:flex; justify-content: center;">
				<h1 style="color: rgb(59 130 246); text-align:center"><b> Steps 3 of 4</b></h1>
			</div>
			<Progressbar size="h-1.5" progress="75" />
			<h2>
				Now that we have described our domain, let's update our password and describe some basic
				information about ourselves for our "User Profile".User profile information will be shown to
				teammates and collaborators when working on studies together.
			</h2>

			<form class="flex flex-col space-y-6" action="#">
				<Label class="space-y-2">
					<Label class="block mb-2">Email <span style="color: rgb(25, 179, 230);">*</span></Label>
					<Input
						bind:value={userInfo.email}
						type="email"
						name="email"
						placeholder="info@openmined.org"
						required
					/>
				</Label>

				<Label class="space-y-2">
					<Label class="block mb-2"
						><h6>Password <span style="color: rgb(25, 179, 230);">*</span></h6></Label
					>
					<Helper class="text-sm">
						To make your account more secure please update your password. Passwords should be at
						least 7 characters long and contain alphanumeric characters.
					</Helper>
					<Input
						bind:value={userInfo.password}
						type="password"
						name="name"
						placeholder="********"
						required
					/>
				</Label>
				<h3><b> PROFILE INFORMATION </b></h3>
				<Label class="space-y-2">
					<Label class="block mb-2"
						>Full Name <span style="color: rgb(25, 179, 230);">*</span>
					</Label>
					<Input
						bind:value={userInfo.name}
						type="email"
						name="email"
						placeholder="Jana Doe"
						required
					/>
				</Label>
				<Label class="space-y-2">
					<Label class="block mb-2">Team</Label>
					<Helper class="text-sm">
						Please identify the team or department you primarily work with at this organization.</Helper
					>
					<Input
						bind:value={userInfo.institution}
						type="text"
						name="institution"
						placeholder="Team name here"
						required
					/>
				</Label>
				<Label class="space-y-2">
					<Label class="block mb-2">Website</Label>
					<Helper
						>To help others verify who you are you can add a link to your university profile, Google
						Scholar profile, or any other profile page that helps showcase your work.</Helper
					>
					<Input
						bind:value={userInfo.website}
						type="text"
						name="website"
						placeholder="www.abc.com"
						required
					/>
				</Label>
			</form>

			<div style="display:flex; justify-content: space-between;">
				<Button pill={true} on:click={() => previousStep()} color="dark">Back</Button>
				<div style="display: flex">
					<div class="cancel-button"><h1>Cancel</h1></div>
					<Button
						disabled={!(userInfo.email !== '' && userInfo.password !== '' && userInfo.name !== '')}
						pill={true}
						on:click={() => submitChanges()}
						color="dark">Finish</Button
					>
				</div>
			</div>
		</div>

		<div class="space-y-9" style="display: {steps[3]}">
			<div style="display:flex; justify-content: center; align-items:center">
				<!-- Header -->
				<div id="onboard-square" />
			</div>
			<div style="display:flex; justify-content: center;">
				<h1 color="black"><b> Setup Complete!</b></h1>
			</div>
			<Progressbar size="h-1.5" progress="100" />

			<h2>
				Congratulations on setting up {domainInfo.name} node. To edit any of your responses you can go
				to "Domain Settings" indicated by a gear icon in the top left corner of your navigation or by
				going to "Account Settings" indicated by your avatar in the top right corner of the navigation.
			</h2>
			<br />
			<div style="display:flex; justify-content: right">
				<Button
					on:click={() => (onBoardModal = false)}
					style="width: 10vh; border-radius:5px;"
					color="dark">Close</Button
				>
			</div>
		</div>
	</Modal>
</main>

<style>
	#onboard-square {
		background: linear-gradient(to bottom left, rgb(146, 247, 51), rgb(30, 155, 251));
		height: 15vh;
		width: 15vh;
	}

	.cancel-button {
		margin-right: 5vh;
		display: flex;
		justify-content: center;
		align-items: center;
		cursor: pointer;
	}
</style>
