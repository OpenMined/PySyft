import type { SyftUID } from './syft';

export type DomainMetadata = {
  name: string;
  description: string;
  organization: string;
  id: SyftUID;
  deployed_on: string;
  tags: string[];
};
