import type { UserView } from "../../../../types/datasite/users"
import type { PageServerLoad } from "./$types"

export const load: PageServerLoad = async ({ params, fetch, depends }) => {
  const slug = params.slug
  const res = await fetch(`/_syft_api/users/${slug}`)
  const json = (await res.json()) as UserView

  depends(`user:${slug}`)

  return { user_requested: json }
}
