import { fail, redirect } from "@sveltejs/kit"
import { login } from "$lib/api/auth"
import { COOKIES } from "$lib/constants"
import { default_cookie_config } from "$lib/utils"

export const load = async ({ parent }) => {
  const { metadata, user } = await parent()

  if (!metadata.signup_enabled) {
    throw redirect(302, user ? "/users" : "/login")
  }

  return {
    metadata,
    user,
  }
}

/** @type {import('./$types').Actions} */
export const actions = {
  default: async ({ cookies, request }) => {
    const data = await request.formData()

    const email = data.get("email")
    const password = data.get("password")

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
      cookies.set(COOKIES.UID, uid, default_cookie_config)
      cookies.set(COOKIES.SIGNING_KEY, signing_key, default_cookie_config)
    } catch (error) {
      return fail(400, { invalid: true })
    }

    throw redirect(302, "/datasets")
  },
}
