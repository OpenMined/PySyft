import { fail, redirect } from "@sveltejs/kit"
import { login } from "$lib/api/auth"
import { default_cookie_config } from "$lib/utils"
import { COOKIES } from "$lib/constants"
import type { Actions } from "./$types"

export const actions: Actions = {
  default: async ({ cookies, request }) => {
    const data = await request.formData()

    const email = data.get("email")
    const password = data.get("password")
    const node_id = data.get("node_id")

    if (
      !email ||
      !password ||
      typeof email !== "string" ||
      typeof password !== "string"
    ) {
      return fail(400, { invalid: true })
    }

    try {
      const { signing_key, uid } = await login({ email, password })

      const cookie_user = {
        uid,
        node_id,
      }

      cookies.set(
        COOKIES.USER,
        JSON.stringify(cookie_user),
        default_cookie_config
      )
      cookies.set(COOKIES.KEY, signing_key, default_cookie_config)
    } catch (error) {
      return fail(400, { invalid: true })
    }

    throw redirect(302, "/datasets")
  },
}
