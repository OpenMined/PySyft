import { jsSyftCall } from "$lib/api/syft_api"
import { unload_cookies } from "$lib/utils"
import { error, json } from "@sveltejs/kit"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async ({ cookies, params }) => {
  try {
    const requested_uid = params.uid

    const { signing_key, server_id } = unload_cookies(cookies)

    const user = await jsSyftCall({
      path: "user.view",
      payload: { uid: { value: requested_uid, fqn: "syft.types.uid.UID" } },
      server_id,
      signing_key,
    })

    const user_view = {
      name: user?.name,
      uid: user.id?.value,
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

export const PUT: RequestHandler = async ({ cookies, params, request }) => {
  try {
    const { signing_key, server_id } = unload_cookies(cookies)
    const requested_uid = params.uid

    const body = await request.json()

    const { name, email, password, institution, website } = body

    const user = await jsSyftCall({
      path: "user.update",
      payload: {
        uid: { value: requested_uid, fqn: "syft.types.uid.UID" },
        user_update: {
          name,
          email,
          password,
          institution,
          website,
          fqn: "syft.service.user.user.UserUpdate",
        },
      },
      server_id,
      signing_key,
    })

    const user_view = {
      name: user?.name,
      uid: user.id?.value,
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
