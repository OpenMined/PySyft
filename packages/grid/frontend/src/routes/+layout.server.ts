import { getMetadata } from "$lib/api/metadata"
import { unload_cookies } from "$lib/utils"
import type { LayoutServerLoad } from "./$types"

export const load: LayoutServerLoad = async ({ cookies, fetch }) => {
  let current_user
  let metadata

  try {
    metadata = await getMetadata()
    const { uid, signing_key, node_id } = unload_cookies(cookies)
    console.log("req", uid)
    const res = await fetch(`/_syft_api/users/${uid}`)
    current_user = await res.json()
  } catch (error) {
    console.log(error)
  } finally {
    return {
      metadata,
      current_user,
    }
  }
}
