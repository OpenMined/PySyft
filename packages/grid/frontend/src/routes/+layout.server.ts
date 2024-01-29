import { unload_cookies } from "$lib/utils"
import type { LayoutServerLoad } from "./$types"

export const load: LayoutServerLoad = async ({ cookies, fetch }) => {
  // let current_user
  // let metadata

  // try {
  //   metadata = await fetch("/_syft_api/metadata")
  //   metadata = await metadata.json()

  //   const { uid } = unload_cookies(cookies)
  //   const res = await fetch(`/_syft_api/users/${uid}`)
  //   current_user = await res.json()
  // } catch (err) {
  //   console.log(err)
  // }

  return {
    // metadata,
    // current_user,
  }
}
