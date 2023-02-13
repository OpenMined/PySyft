<script>
	import { browser } from '$app/environment';
	import { getPath } from '../../../lib/utils.js';

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
