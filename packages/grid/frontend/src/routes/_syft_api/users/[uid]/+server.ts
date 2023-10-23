import { jsSyftCall } from "$lib/api/syft_api"
import { unload_cookies } from "$lib/utils"
import { error, json } from "@sveltejs/kit"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async ({ cookies, params }) => {
  try {
    const requested_uid = params.uid

    const { signing_key, node_id } = unload_cookies(cookies)

    const user = await jsSyftCall({
      path: "user.view",
      payload: { uid: { value: requested_uid, fqn: "syft.types.uid.UID" } },
      node_id,
      signing_key,
    })

    const user_view = {
      name: user?.name,
      uid: user.id.value,
      email: user.email,
      role: user.role.value,
      website: user.website,
      institution: user.institution,
    }

    return json(user_view)
  } catch (err) {
    console.log(err)
    throw error(400, "invalid user")
  }
}
