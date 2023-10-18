import { jsSyftCall } from "./syft_api"

export async function getAllUsers(page_size = 0, page_index = 0) {
  return await jsSyftCall({
    path: "user.get_all",
    payload: { page_size: page_size, page_index: page_index },
  })
}

export async function getUser(
  uid: string,
  signing_key: string | Uint8Array,
  node_id: string
) {
  try {
    const user = await jsSyftCall({
      path: "user.view",
      payload: { uid: { value: uid, fqn: "syft.types.uid.UID" } },
      node_id,
      signing_key,
    })

    return {
      uid: user.id.value,
      email: user.email,
      role: user.role.value,
      website: user.website,
      institution: user.institution,
    }
  } catch (error) {
    console.error(error)
    return undefined
  }
}

export async function searchUsersByName(
  name: string,
  page_size = 0,
  page_index = 0
) {
  return await jsSyftCall({
    path: "user.search",
    payload: {
      user_search: { name: name, fqn: "syft.service.user.user.UserSearch" },
      page_size: page_size,
      page_index: page_index,
    },
  })
}

export async function updateCurrentUser(
  name: string,
  email: string,
  password: string,
  institution: string,
  website: string
) {
  const userUpdate = {
    name: name,
    email: email,
    password: password,
    institution: institution,
    website: website,
    fqn: "syft.service.user.user.UserUpdate",
  }

  return await jsSyftCall({
    path: "user.update",
    payload: {
      uid: makeSyftUID(getUserIdFromStorage()),
      user_update: userUpdate,
    },
  })
}
