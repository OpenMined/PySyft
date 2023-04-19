import { metadata, isLoading } from '$lib/store';
import { getMetadata } from '$lib/api/metadata';

async function domainStartup() {
  try {
    isLoading.set(true);
    const updatedMetadata = await getMetadata();
    metadata.set(updatedMetadata);
  } catch (error) {
    console.log(error);
  } finally {
    isLoading.set(false);
  }
}

domainStartup();
