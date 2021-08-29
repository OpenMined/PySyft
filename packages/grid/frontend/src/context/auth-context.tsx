import type {FunctionComponent} from 'react'
import {createContext, useContext} from 'react'
import {useQueryClient} from 'react-query'
import {logout, getToken, login} from '@/lib/auth'

const AuthContext = createContext(null)

AuthContext.displayName = 'AuthenticationContext'

export const useAuth = () => useContext(AuthContext)

export const AuthProvider: FunctionComponent = ({children}) => {
  const queryClient = useQueryClient()
  const appLogout = () => {
    logout()
    queryClient.clear()
  }

  return <AuthContext.Provider value={{getToken, logout: appLogout, login}}>{children}</AuthContext.Provider>
}
