import { writable } from "svelte/store"
import { JSClient } from "./client/jsclient/jsClient.svelte"

export const store = writable({
  client: "",
  metadata: {},
  user_info: {},
})

export const isLoading = writable(false)
export const metadata = writable()
export const user = writable()

export async function getClient() {
  let newStore = ""
  store.subscribe((value) => {
    newStore = value
  })

  if (!newStore.client) {
    newStore.client = await new JSClient(
      `${window.location.protocol}//${window.location.host}`
    )

    const session = window.sessionStorage.getItem("session")

    if (session) {
      newStore.client.recoverSession(session)
    }

    store.set(newStore)
  }
  return newStore.client
}
