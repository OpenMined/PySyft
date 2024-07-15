import { ServiceRoles } from "../types/datasite/users"
import { COOKIES } from "./constants"
import type { CookieSerializeOptions } from "cookie"
import type { Cookies } from "@sveltejs/kit"
import { onDestroy } from "svelte"

export function shortName(name: string) {
  const nameList = name.split(" ")
  let letters = ""
  nameList[0].charAt(0).toUpperCase()
  if (nameList.length < 2) {
    letters += nameList[0][0]
  } else {
    nameList[1].charAt(0).toUpperCase()
    letters += nameList[0][0]
    letters += nameList[1][0]
  }

  return letters
}

export function getInitials(name: string) {
  return name
    ? name
        .split(" ")
        .map((n, index, arr) => {
          if (index === 0 || index === arr.length - 1) return n[0]
        })
        .filter((n) => n)
        .join("")
    : ""
}

export function logout() {
  window.localStorage.removeItem("id")
  window.localStorage.removeItem("serverId")
  window.localStorage.removeItem("key")
}

export function getUserRole(value: ServiceRoles) {
  return ServiceRoles[value]
}

export const default_cookie_config: CookieSerializeOptions = {
  path: "/",
  httpOnly: true,
  sameSite: "strict",
  secure: import.meta.env.NODE_ENV === "production",
  maxAge: 60 * 60 * 24 * 30, // 30 days
}

interface CookieData {
  uid: string
  server_id: string
  signing_key: string
}

export function unload_cookies(cookies: Cookies): CookieData {
  const cookieUser = cookies.get(COOKIES.USER)
  const signing_key = cookies.get(COOKIES.KEY)

  if (!cookieUser || !signing_key) throw Error("Cookie is empty")

  return { ...JSON.parse(cookieUser), signing_key }
}

export function get_url_page_params(url: URL) {
  const page_size = parseInt(url.searchParams.get("page_size") || "10")
  const page_index = parseInt(url.searchParams.get("page_index") || "0")
  return { page_size, page_index }
}

export function get_form_data_values(data: FormData) {
  return Object.fromEntries(data.entries())
}

export function onInterval(callback: () => void, ms: number) {
  const interval = setInterval(callback, ms)

  onDestroy(() => {
    clearInterval(interval)
  })
}
