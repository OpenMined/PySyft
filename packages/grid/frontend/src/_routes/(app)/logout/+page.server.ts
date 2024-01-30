import { redirect } from "@sveltejs/kit"
import { COOKIES } from "$lib/constants.js"

export const load = async ({ cookies }) => {
  for (const cookie of Object.values(COOKIES)) {
    cookies.delete(cookie)
  }
  throw redirect(302, "/login")
}
