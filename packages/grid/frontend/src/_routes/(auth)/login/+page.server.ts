import { fail, redirect } from "@sveltejs/kit"
import { login } from "$lib/api/auth"
import { default_cookie_config, get_form_data_values } from "$lib/utils"
import { COOKIES } from "$lib/constants"
import type { Actions } from "./$types"

export const actions: Actions = {
  default: async ({ cookies, request }) => {
    try {
      const data = await request.formData()
      const { email, password, server_id } = get_form_data_values(data)

      if (
        !email ||
        !password ||
        typeof email !== "string" ||
        typeof password !== "string"
      ) {
        throw new Error(`invalid form data: email:${email} server:${server_id}`)
      }

      const { signing_key, uid } = await login({ email, password })

      const cookie_user = {
        uid,
        server_id,
      }

      cookies.set(
        COOKIES.USER,
        JSON.stringify(cookie_user),
        default_cookie_config
      )
      cookies.set(COOKIES.KEY, signing_key, default_cookie_config)
    } catch (err) {
      console.log(err)
      return fail(400, { invalid: true })
    }

    throw redirect(302, "/users")
  },
}
