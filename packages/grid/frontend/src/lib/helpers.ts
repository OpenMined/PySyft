export function parseActiveRoute(route: string): string {
  if (route === '/') {
    return route;
  }

  return route.slice(1);
}

export function parseBadgeForNav(badge: string): string {
  if (badge.length <= 9) {
    return badge;
  }

  return badge.slice(0, 9).concat('...');
}
