import { getMetadata } from "$lib/api/metadata"
import { getUser } from "$lib/api/users"
import { COOKIES } from "$lib/constants"

/** @type {import('./$types').LayoutServerLoad} */
export async function load({ cookies }) {
  const user_id = cookies.get(COOKIES.UID)
  const signing_key = cookies.get(COOKIES.SIGNING_KEY)

  let user
  let metadata

  try {
    metadata = await getMetadata()

    if (user_id) {
      user = await getUser(user_id, signing_key, metadata.node_id)
    }
  } catch (error) {
    console.log(error)
  } finally {
    return {
      metadata,
      user,
    }
  }
}
