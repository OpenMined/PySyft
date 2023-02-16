import { writable } from 'svelte/store';
import { JSSerde } from './jsserde/jsserde.svelte';

export const credentials = writable('');
export const jsSerde = writable('');

export const store = writable({
	jsserde: ''
});

export async function getSerde() {
	let newStore = '';
	store.subscribe((value) => {
		newStore = value;
	});

	if (!newStore.jsserde) {
		let type_bank = await fetch('http://localhost:8081/api/v1/syft/serde')
			.then((response) => response.json())
			.then(function (response) {
				return response['bank'];
			});
		newStore.jsserde = new JSSerde(type_bank);
		store.set(newStore);
	}
	return newStore.jsserde;
}
