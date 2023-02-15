import { writable } from 'svelte/store';
import { JSClient } from './jsserde/jsClient.svelte';

export const store = writable({
  client: ''
});

export async function getClient() {
  let newStore = '';
  store.subscribe((value) => {
    newStore = value;
  });

  if (!newStore.client) {
    newStore.client = await new JSClient('http://localhost:8081');
    store.set(newStore);
  }
  return newStore.client;
}
