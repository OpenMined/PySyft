import { metadata, user, isLoading } from '$lib/store';
import { getMetadata } from '$lib/api/metadata';
import { getUserIdFromStorage } from '$lib/api/keys';
import { getSelf } from '$lib/api/users';

async function domainStartup() {
  try {
    isLoading.set(true);
    const updatedMetadata = await getMetadata();
    metadata.set(updatedMetadata);
    if (getUserIdFromStorage()) {
      const updatedUser = await getSelf();
      user.set(updatedUser);
    }
  } catch (error) {
    console.log(error);
  } finally {
    isLoading.set(false);
  }
}

domainStartup();
