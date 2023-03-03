import { writable } from 'svelte/store';
import { JSClient } from './client/jsclient/jsClient.svelte';

export const store = writable({
  client: '',
  session_token: '',
  metadata: {},
  user_info: {}
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
