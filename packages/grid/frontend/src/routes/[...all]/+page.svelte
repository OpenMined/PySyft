<script>
  import { onMount } from "svelte"
  import DatasiteMetadataPanel from "$lib/components/authentication/DatasiteMetadataPanel.svelte"
  import AuthCircles from "$lib/components/AuthCircles.svelte"
  import Nav from "$lib/components/authentication/Nav.svelte"
  import Footer from "$lib/components/authentication/Footer.svelte"
  import { API_BASE_URL } from "$lib/constants"
  import { deserialize } from "$lib/api/serde"

  let _metadata = null

  const get_metadata = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/metadata_capnp`)
      const metadata_raw = await deserialize(res)

      return {
        admin_email: metadata_raw?.admin_email,
        description: metadata_raw?.description,
        highest_version: metadata_raw?.highest_version,
        lowest_version: metadata_raw?.lowest_version,
        name: metadata_raw?.name,
        server_id: metadata_raw?.id?.value,
        server_side: metadata_raw?.server_side_type,
        server_type: metadata_raw?.server_type?.value,
        organization: metadata_raw?.organization,
        signup_enabled: metadata_raw?.signup_enabled,
        syft_version: metadata_raw?.syft_version,
      }
    } catch (err) {
      console.log(err)
    }
  }

  onMount(async () => {
    _metadata = await get_metadata()
  })

  $: metadata = _metadata
</script>

<title>PySyft</title>
<div
  class="fixed top-0 right-0 w-full h-full max-w-[808px] max-h-[880px] z-[-1]"
>
  <AuthCircles />
</div>
<main class="flex flex-col p-10 gap-10 h-screen">
  <Nav version={_metadata?.syft_version} />
  <div class="grow flex-shrink-0">
    <div
      class="flex flex-col xl:flex-row w-full h-full xl:justify-around items-center gap-12"
    >
      <DatasiteMetadataPanel {metadata} />
    </div>
  </div>
  <Footer />
</main>
