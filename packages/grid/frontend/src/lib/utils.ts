import { ServiceRoles } from "../types/domain/users"

export function shortName(name) {
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
  window.localStorage.removeItem("nodeId")
  window.localStorage.removeItem("key")
}

export function getUserRole(value: ServiceRoles) {
  return ServiceRoles[value]
}

export const default_cookie_config = {
  path: "/",
  httpOnly: true,
  sameSite: "strict",
  secure: import.meta.env.NODE_ENV === "production",
  maxAge: 60 * 60 * 24 * 30, // 30 days
}
