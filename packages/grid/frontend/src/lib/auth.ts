import jwtDecode from 'jwt-decode'
import api from '@/utils/api'
import { parseCookies, setCookie, destroyCookie } from 'nookies'
import type { Credentials } from '@/types/Login'

const GRID_KEY = 'grid'

export function getToken() {
  const parsedCookies = parseCookies()
  return parsedCookies?.[GRID_KEY]
}

export function logout() {
  destroyCookie(null, GRID_KEY, { path: '/' })
  // TODO: Add private routes
  if (typeof window !== 'undefined') window.location = '/login'
}

export function decodeToken() {
  const token = getToken()
  if (!token) return null
  return jwtDecode(token)
}

export async function login({
  email,
  password,
}: {
  email: string
  password: string
}) {
  try {
    const token: Credentials = await api
      .post('login', { json: { email, password } })
      .json()
    setCookie(null, GRID_KEY, token.access_token, {
      maxAge: 30 * 24 * 60 * 60 * 5,
      path: '/',
    })
    return 'ok'
  } catch (err) {
    throw err
  }
}
