import { jsSyftCall } from "$lib/api/syft_api"
import { unload_cookies } from "$lib/utils"
import { error, json } from "@sveltejs/kit"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async ({ cookies, url }) => {
  try {
    const page_size = parseInt(url.searchParams.get("page_size") || "10")
    const page_index = parseInt(url.searchParams.get("page_index") || "0")

    const { signing_key, node_id } = unload_cookies(cookies)

    const users = await jsSyftCall({
      path: "user.get_all",
      payload: { page_size, page_index },
      node_id,
      signing_key,
    })

    return json({ list: users.users, total: users.total })
  } catch (err) {
    console.log(err)
    throw error(400, "invalid user")
  }
}
