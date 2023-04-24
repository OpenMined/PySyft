import type { SyftUID } from './syft';

export interface UserListView {
  id: SyftUID;
  name: string;
  email: string;
}

export interface UserView {
  id: SyftUID;
  name: string;
  email: string;
  organization?: string;
  website?: string;
  role?: number;
}
