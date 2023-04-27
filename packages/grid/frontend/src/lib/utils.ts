import { onDestroy } from 'svelte';
import { ServiceRoles } from '../types/domain/users';

export function getPath() {
  return window.location.pathname;
}

export function getInitials(name: string) {
  return name
    ? name
      .split(' ')
      .map((n, index, arr) => {
        if (index === 0 || index === arr.length - 1) return n[0];
      })
      .filter((n) => n)
      .join('')
    : '';
}

export function logout() {
  window.localStorage.removeItem('id');
  window.localStorage.removeItem('key');
}

export function getUserRole(value: ServiceRoles) {
  return ServiceRoles[value];
}

export function onInterval(callback: () => void, ms: number) {
  const interval = setInterval(callback, ms);

  onDestroy(() => {
    clearInterval(interval);
  });
}
