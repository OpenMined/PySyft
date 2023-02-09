import { writable } from 'svelte/store';
import { JSSerde } from './jsserde/jsserde.svelte';

export const credentials = writable('');
export const jsSerde = writable('');

export const store = writable({
	user: {},
	tenantDetail: {},
	jsserde: {},
	credentials: {}
});

export async function getSerde() {
	let jsValue = '';
	jsSerde.subscribe((value) => {
		jsValue = value;
	});

	if (!jsValue) {
		let type_bank = await fetch('http://localhost:8081/api/v1/syft/serde')
			.then((response) => response.json())
			.then(function (response) {
				return response['bank'];
			});
		jsSerde.set(new JSSerde(type_bank));
	}
	return jsValue;
}
