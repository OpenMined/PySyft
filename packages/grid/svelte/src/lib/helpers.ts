export function parseActiveRoute(route: string): string {
  if (route === '/') {
    return route;
  }

  return route.slice(1);
}
