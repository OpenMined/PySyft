import ky from 'ky'
/* import { getToken } from '@/lib/auth' */

const prefixUrl = `${process.env.NEXT_PUBLIC_HOST || ''}${
  process.env.NEXT_PUBLIC_API_URL || '/api/v1'
}`

export const api = ky.extend({
  /* hooks: { */
  /*   beforeRequest: [ */
  /*     req => { */
  /*       const token = getToken() */
  /*       if (token) { */
  /*         req.headers.set('Authorization', `Bearer ${token}`) */
  /*       } */
  /*     }, */
  /*   ], */
  /* }, */
  prefixUrl,
})
