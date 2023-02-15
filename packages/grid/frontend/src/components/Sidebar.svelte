<script>
	import { Sidebar, SidebarGroup, SidebarItem, SidebarWrapper, SidebarCta } from 'flowbite-svelte';
	import { prettyName, shortName } from '../lib/utils.js';
	export let user_info;
	export let metadata;
	export let activeUrl;

	let spanClass = 'flex-1 ml-3 whitespace-nowrap';
</script>

<main>
	<Sidebar>
		<SidebarWrapper
			style="position: absolute;height: 100%; width: 15%;border-right:solid 1px #fff;"
		>
			<SidebarGroup>
				<a href="/config"
					><div style="width:100%;display:flex;justify-content:right">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="rgb(25, 179, 230)"
							width="24"
							height="24"
							viewBox="0 0 24 24"
							><path
								d="M24 14v-4h-3.23c-.229-1.003-.624-1.94-1.156-2.785l2.286-2.286-2.83-2.829-2.286 2.286c-.845-.532-1.781-.928-2.784-1.156v-3.23h-4v3.23c-1.003.228-1.94.625-2.785 1.157l-2.286-2.286-2.829 2.828 2.287 2.287c-.533.845-.928 1.781-1.157 2.784h-3.23v4h3.23c.229 1.003.624 1.939 1.156 2.784l-2.286 2.287 2.829 2.829 2.286-2.286c.845.531 1.782.928 2.785 1.156v3.23h4v-3.23c1.003-.228 1.939-.624 2.784-1.156l2.286 2.286 2.828-2.829-2.285-2.286c.532-.845.928-1.782 1.156-2.785h3.231zm-12 2c-2.209 0-4-1.791-4-4s1.791-4 4-4 4 1.791 4 4-1.791 4-4 4z"
							/></svg
						>
					</div></a
				>
				<div id="domain-profile-space">
					<div class="space-y-2" style="align-items:center;">
						<!-- Box to limit border since its a relative element. -->
						<div style="display: flex;  justify-content:center;">
							<div id="domain-profile-border">
								<div id="domain-profile-circle">
									<h3 style="color:white"><b> {shortName(prettyName(metadata['name']))} </b></h3>
								</div>
							</div>
						</div>
						<h5 style="text-align: center;font-size: 8;font-weight: bold;">
							{prettyName(metadata['name'])}
						</h5>
						<h5 style="text-align: center;margin-bottom: 35px;">
							<a href="/login"> <u> Logout </u> </a>
						</h5>
					</div>
				</div>
			</SidebarGroup>
			<SidebarGroup border>
				<h2 style="color:gray; margin-left:10px;font-weight:bold;font-size:12px;">ASSETS</h2>
				<SidebarItem
					label="Datasets"
					href="/datasets"
					on:click={() => {
						activeUrl = '/datasets';
					}}
					active={activeUrl === '/datasets'}
				>
					<svelte:fragment slot="icon">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="rgb(25, 179, 230)"
							width="20"
							height="20"
							viewBox="0 0 20 20"
							><path
								d="M2 23h-2v-2h2v2zm0-12h-2v2h2v-2zm0 5h-2v2h2v-2zm0-15h-2v2h2v-2zm2 0v2h20v-2h-20zm-2 5h-2v2h2v-2zm2 7h20v-2h-20v2zm0 10h20v-2h-20v2zm0-15h20v-2h-20v2zm0 10h20v-2h-20v2z"
							/></svg
						>
					</svelte:fragment>
				</SidebarItem>
				<SidebarItem
					label="Requests"
					href="/requests"
					on:click={() => {
						activeUrl = '/requests';
					}}
					active={activeUrl === '/requests'}
					{spanClass}
				>
					<svelte:fragment slot="icon">
						<svg
							clip-rule="evenodd"
							fill="rgb(25, 179, 230)"
							width="24"
							height="24"
							stroke-linejoin="round"
							stroke-miterlimit="2"
							viewBox="0 0 24 24"
							xmlns="http://www.w3.org/2000/svg"
							><path
								d="m22 17.75c0-.414-.336-.75-.75-.75h-9.5c-.414 0-.75.336-.75.75s.336.75.75.75h9.5c.414 0 .75-.336.75-.75zm-19.749-1.377 2.172 1.936c.144.128.324.191.502.191.204 0 .408-.082.557-.244l3.677-4.02c.131-.145.197-.326.197-.506 0-.413-.334-.751-.753-.751-.205 0-.408.082-.558.244l-3.174 3.47-1.616-1.44c-.144-.128-.323-.191-.502-.191-.418 0-.753.338-.753.751 0 .206.084.412.251.56zm15.749-2.623c0-.414-.336-.75-.75-.75h-5.5c-.414 0-.75.336-.75.75s.336.75.75.75h5.5c.414 0 .75-.336.75-.75zm-15.749-5.356 2.172 1.936c.144.128.324.191.502.191.204 0 .408-.082.557-.244l3.677-4.02c.131-.144.197-.326.197-.506 0-.412-.334-.751-.753-.751-.205 0-.408.082-.558.244l-3.174 3.471-1.616-1.44c-.144-.128-.323-.192-.502-.192-.418 0-.753.338-.753.752 0 .206.084.411.251.559zm17.749 1.356c0-.414-.336-.75-.75-.75h-7.5c-.414 0-.75.336-.75.75s.336.75.75.75h7.5c.414 0 .75-.336.75-.75zm2-4c0-.414-.336-.75-.75-.75h-9.5c-.414 0-.75.336-.75.75s.336.75.75.75h9.5c.414 0 .75-.336.75-.75z"
								fill-rule="nonzero"
							/></svg
						>
						<!--<svg xmlns="http://www.w3.org/2000/svg" fill='rgb(59 130 246)' viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-6 h-6"><path stroke-linecap="round" stroke-linejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" /></svg>-->
					</svelte:fragment>
				</SidebarItem>
				<SidebarItem
					label="Policies"
					href="/policies"
					on:click={() => {
						activeUrl = '/policies';
					}}
					active={activeUrl === '/policies'}
					{spanClass}
				>
					<svelte:fragment slot="icon">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="rgb(25, 179, 230)"
							width="24"
							height="24"
							viewBox="0 0 24 24"
							><path
								d="M15.762 8.047l-4.381 4.475-2.215-2.123-1.236 1.239 3.451 3.362 5.619-5.715-1.238-1.238zm-3.762-5.503c2.5 1.805 4.555 2.292 7 2.416v9.575c0 3.042-1.686 3.827-7 7.107-5.309-3.278-7-4.065-7-7.107v-9.575c2.447-.124 4.5-.611 7-2.416zm0-2.544c-3.371 2.866-5.484 3-9 3v11.535c0 4.603 3.203 5.804 9 9.465 5.797-3.661 9-4.862 9-9.465v-11.535c-3.516 0-5.629-.134-9-3z"
							/></svg
						>
					</svelte:fragment>
					<svelte:fragment slot="subtext">
						<span
							class="inline-flex justify-center items-center px-2 ml-3 text-sm font-medium text-gray-800 bg-gray-200 rounded-full dark:bg-gray-700 dark:text-gray-300"
							>Coming Soon</span
						>
					</svelte:fragment>
				</SidebarItem>
			</SidebarGroup>
			<!-- ADMIN Section -->
			{#if user_info.role === 'Owner'}
				<SidebarGroup style="margin-top:25px;">
					<h2 style="color:gray; margin-left:10px;font-weight:bold;font-size:12px;">ADMIN</h2>
					<SidebarItem
						href="/users"
						rel="no-refresh"
						label="Users"
						on:click={() => {
							activeUrl = '/users';
						}}
						active={activeUrl === '/users'}
					>
						<svelte:fragment slot="icon">
							<svg
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
								stroke-width="1.5"
								stroke="currentColor"
								class="w-6 h-6"
								><path
									stroke-linecap="round"
									stroke-linejoin="round"
									d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
								/></svg
							>
						</svelte:fragment>
					</SidebarItem>
				</SidebarGroup>
			{/if}
		</SidebarWrapper>
	</Sidebar>
</main>

<style>
	#domain-profile-space {
		height: 15vh;
		width: 100%;
		display: flex;
		justify-content: center;
		align-items: center;
	}
	#domain-profile-circle {
		height: 10vh;
		width: 10vh;
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
</style>
