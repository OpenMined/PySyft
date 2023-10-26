import { fail, redirect } from "@sveltejs/kit"
import { API_BASE_URL } from "$lib/constants"
import { get_form_data_values } from "$lib/utils"
import { serialize, deserialize } from "$lib/api/serde"
import { throwIfError, getErrorMessage } from "$lib/api/syft_error_handler"
import type { Actions } from "./$types"

export const load = async ({ parent }) => {
  const { metadata } = await parent()

  if (!metadata.signup_enabled) {
    throw redirect(302, "/login")
  }
}

export const actions: Actions = {
  default: async ({ request }) => {
    try {
      const data = await request.formData()
      const formDataValues = get_form_data_values(data)
      const { email, password, confirm_password, fullName } = formDataValues

      if (!email || !password || !confirm_password || !fullName) {
        return fail(400, { invalid: true, message: "Missing fields!" })
      }

      if (password !== confirm_password) {
        return fail(400, { invalid: true, message: "Passwords do not match!" })
      }

      if (
        typeof email !== "string" ||
        typeof password !== "string" ||
        typeof fullName !== "string"
      ) {
        return fail(400, { invalid: true, message: "Invalid fields!" })
      }

      Object.keys(formDataValues).forEach(
        (k) => formDataValues[k] == "" && delete formDataValues[k]
      )

      // register user
      const res = await fetch(`${API_BASE_URL}/register`, {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: serialize({
          ...formDataValues,
          fqn: "syft.service.user.user.UserCreate",
        }),
      })

      if (res.ok) {
        const json = await deserialize(res)
        throwIfError(json)
        throw redirect(302, "/login")
      }

      throw Error("Unknown error")
    } catch (err) {
      return fail(400, { invalid: true, message: getErrorMessage(err) })
    }
  },
}
