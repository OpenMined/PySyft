import { unload_cookies } from "$lib/utils"
import { redirect } from "@sveltejs/kit"
import type { LayoutServerLoad } from "./$types"

export const load: LayoutServerLoad = async ({ parent }) => {
  const { current_user } = await parent()

  if (!current_user || !current_user.email) {
    throw redirect(302, "/login")
  }
}
