import ky from 'ky'
/* import { getToken } from '@/lib/auth' */

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
  prefixUrl: '/api/v1',
})
