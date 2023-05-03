import { ServiceRoles } from '../types/domain/users';

export function prettyName(name) {
  let nameList = name.split('_');
  for (var i = 0; i < nameList.length; i++) {
    nameList[i] = nameList[i].charAt(0).toUpperCase() + nameList[i].slice(1);
  }
  return nameList.join(' ');
}

export function shortName(name) {
  let nameList = name.split(' ');
  let letters = '';
  nameList[0].charAt(0).toUpperCase();
  if (nameList.length < 2) {
    letters += nameList[0][0];
  } else {
    nameList[1].charAt(0).toUpperCase();
    letters += nameList[0][0];
    letters += nameList[1][0];
  }

  return letters;
}

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
  window.localStorage.removeItem('nodeId');
  window.localStorage.removeItem('key');
}

export function getUserRole(value: ServiceRoles) {
  return ServiceRoles[value];
}
