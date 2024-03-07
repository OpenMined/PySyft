export const API_BASE_URL = import.meta.env.BACKEND_API_BASE_URL || "/api/v2"

export const syftRoles = {
  1: "Guest",
  2: "Data Scientist",
  32: "Data Owner",
  128: "Admin",
}

export const COOKIES = {
  USER: "_user",
  KEY: "_signing_key",
}
