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

export function MapRoles(role) {
  const roleMap = Object.freeze({
    1: 'Guest',
    2: 'Data Scientist',
    32: 'Data Owner',
    128: 'Admin'
  });
  return roleMap[role];
}
export function getPath() {
  return window.location.pathname;
}
