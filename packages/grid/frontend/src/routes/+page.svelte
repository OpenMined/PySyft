<script>
    import { JSSerde } from "../lib/jsserde/jsserde.svelte";
    async function initJSSerde(url){
        let type_bank = await fetch(url)
				 .then(response => response.json())
				 .then(function(response) {return response['bank']})
        return new JSSerde(type_bank);
    }

    let promise = initJSSerde('http://localhost:8081/api/v1/syft/serde');
</script>

<main>
    {#await promise then jsserde}
    <h1>Welcome to SvelteKit</h1>
    <p>Visit <a href="https://kit.svelte.dev">kit.svelte.dev</a> to read the documentation</p>    
    {/await}
</main>
