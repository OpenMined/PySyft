import { deserialize } from "$lib/api/serde"
import { jsSyftCall } from "$lib/api/syft_api"
import { API_BASE_URL } from "$lib/constants"
import { unload_cookies } from "$lib/utils"
import { error, json } from "@sveltejs/kit"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/metadata_capnp`)
    const metadata_raw = await deserialize(res)

    return json({
      admin_email: metadata_raw?.admin_email,
      deployed_on: metadata_raw?.deployed_on,
      description: metadata_raw?.description,
      highest_version: metadata_raw?.highest_version,
      lowest_version: metadata_raw?.lowest_version,
      name: metadata_raw?.name,
      server_id: metadata_raw?.id?.value,
      server_side: metadata_raw?.server_side_type,
      server_type: metadata_raw?.server_type?.value,
      organization: metadata_raw?.organization,
      signup_enabled: metadata_raw?.signup_enabled,
      syft_version: metadata_raw?.syft_version,
    })
  } catch (err) {
    console.log(err)
    throw error(400, "unable to fetch metadata")
  }
}

export const PATCH: RequestHandler = async ({ cookies, request }) => {
  try {
    const { signing_key, server_id } = unload_cookies(cookies)

    const metadata = await request.json()

    const new_metadata = await jsSyftCall({
      path: "settings.update",
      payload: {
        settings: {
          ...metadata,
          fqn: "syft.service.settings.settings.ServerSettingsUpdate",
        },
      },
      server_id,
      signing_key,
    })

    return json(new_metadata)
  } catch (err) {
    console.log(err)
    throw error(400, "invalid metadata")
  }
}
