/** @type {import('./$types').PageServerLoad} */
export const load = async ({ params }) => {
  return {
    slug: params.slug,
  }
}
