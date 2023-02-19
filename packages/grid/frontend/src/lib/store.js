import { writable } from 'svelte/store';
import { JSClient } from './jsserde/jsClient.js';

export const store = writable({
  client: JSClient
});

export async function getClient() {
  let newClient = JSClient;
  store.subscribe((value) => {
    newClient = value.client;
  });

  if (!newClient) {
    newClient = await new JSClient('http://localhost:8081');
    store.set(newClient);
  }
  return newClient;
}
