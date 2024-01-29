import { deserialize } from "$lib/api/serde"
import { API_BASE_URL } from "$lib/constants"
import { error, json } from "@sveltejs/kit"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/metadata_capnp`)
    const metadata_raw = await deserialize(res)

    return json({
      admin_email: metadata_raw?.admin_email,
      description: metadata_raw?.description,
      highest_version: metadata_raw?.highest_version,
      lowest_version: metadata_raw?.lowest_version,
      name: metadata_raw?.name,
      node_id: metadata_raw?.id?.value,
      node_side: metadata_raw?.node_side_type,
      node_type: metadata_raw?.node_type?.value,
      organization: metadata_raw?.organization,
      signup_enabled: metadata_raw?.signup_enabled,
      syft_version: metadata_raw?.syft_version,
    })
  } catch (err) {
    console.log(err)
    throw error(400, "unable to fetch metadata")
  }
}
