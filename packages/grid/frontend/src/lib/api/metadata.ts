import ky from "ky"
import { jsSyftCall } from "./syft_api"
import { API_BASE_URL } from "../constants"
import { deserialize } from "./serde"

export async function getMetadata() {
  try {
    const res = await ky.get(`${API_BASE_URL}/metadata_capnp`)

    const metadata = await deserialize(res)

    return {
      admin_email: metadata?.admin_email,
      deployed_on: metadata?.deployed_on,
      description: metadata?.description,
      highest_version: metadata?.highest_version,
      lowest_version: metadata?.lowest_version,
      name: metadata?.name,
      node_id: metadata?.id?.value,
      node_side: metadata?.node_side_type,
      node_type: metadata?.node_type?.value,
      organization: metadata?.organization,
      signup_enabled: metadata?.signup_enabled,
      syft_version: metadata?.syft_version,
    }
  } catch (error) {
    console.log(error)
    throw error
  }
}

export async function updateMetadata(newMetadata) {
  const payload = {
    settings: {
      ...newMetadata,
      fqn: "syft.service.settings.settings.NodeSettingsUpdate",
    },
  }
  return await jsSyftCall({ path: "settings.update", payload })
}
