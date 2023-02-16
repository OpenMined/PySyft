<script>
	import { getPath } from '../../../lib/utils.js';
	import { browser } from '$app/environment';
	let path = '';
	if (browser) {
		path = getPath();
	}
</script>

<main>
	<div class="page-container">{path}</div>
</main>

<style>
	.page-container {
		width: 85%;
		position: absolute;
		height: 93%;
		top: 7%;
		left: 15%;
		display: flex;
		justify-content: center;
		align-items: center;
	}
</style>
