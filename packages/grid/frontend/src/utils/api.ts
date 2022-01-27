import ky from 'ky-universal'
import { getToken } from '@/lib/auth'

const api = ky.extend({
  hooks: {
    beforeRequest: [
      req => {
        const token = getToken()
        if (token) {
          req.headers.set('Authorization', `Bearer ${token}`)
        }
      },
    ],
  },
  prefixUrl: '/api/v1',
})

export default api
