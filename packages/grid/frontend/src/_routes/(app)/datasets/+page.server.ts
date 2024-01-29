import type { PageServerLoad } from "./$types"

export const load: PageServerLoad = async ({ fetch, depends }) => {
  const res = await fetch(`/_syft_api/datasets`)
  const json = (await res.json()) as { list: any[]; total: number }

  depends("dataset:list")

  return json
}
