import { jsSyftCall } from "$lib/api/syft_api"
import { get_url_page_params, unload_cookies } from "$lib/utils"
import { error, json } from "@sveltejs/kit"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async ({ cookies, url }) => {
  try {
    const { page_size, page_index } = get_url_page_params(url)
    const { signing_key, node_id } = unload_cookies(cookies)

    const dataset = await jsSyftCall({
      path: "dataset.get_all",
      payload: { page_size, page_index },
      node_id,
      signing_key,
    })

    console.log({ dataset })

    const dataset_view = { ...dataset }
    console.log({ dataset_view })

    return json(dataset_view)
  } catch (err) {
    console.log(err)
    throw error(400, "invalid dataset")
  }
}
