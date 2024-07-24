import { jsSyftCall } from "$lib/api/syft_api"
import { unload_cookies } from "$lib/utils"
import { error, json } from "@sveltejs/kit"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async ({ cookies, url }) => {
  try {
    const page_size = parseInt(url.searchParams.get("page_size") || "10")
    const page_index = parseInt(url.searchParams.get("page_index") || "0")

    const { signing_key, server_id } = unload_cookies(cookies)

    const users = await jsSyftCall({
      path: "user.get_all",
      payload: { page_size, page_index },
      server_id,
      signing_key,
    })

    return json({ list: users.users, total: users.total })
  } catch (err) {
    console.log(err)
    throw error(400, "invalid user")
  }
}

export const POST: RequestHandler = async ({ cookies, request }) => {
  try {
    const { signing_key, server_id } = unload_cookies(cookies)

    const new_user = await request.json()

    const user = await jsSyftCall({
      path: "user.create",
      payload: {
        user_create: { ...new_user, fqn: "syft.service.user.user.UserCreate" },
      },
      server_id,
      signing_key,
    })

    return json(user)
  } catch (err) {
    console.log(err)
    throw error(400, "invalid user")
  }
}
