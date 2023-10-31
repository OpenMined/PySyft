import type { PageServerLoad } from "./$types"

export const load: PageServerLoad = async ({ params }) => {
  return {
    slug: params.slug,
  }
}
