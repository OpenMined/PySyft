# PyGrid UI

PyGrid Admin is the user interface that allows data owners to manage their
**deployed** PyGrid domains and networks.

## Installation

You need to install all dependencies first.

```bash
cd <pysyft-root>/packages/grid/svelte
pnpm install
```

`svelte` is a temporary name. Check back down the road as we'll probably rename it.

## Developing

Once you've installed the project and all dependencies, start a development server:

```bash
pnpm dev
```

Add the `host` flag to expose the dev server and the `port` flag to specify a port:

```bash
pnpm dev --host --port=4200
```

## Building

We use [Vite](https://vitejs.dev/) with the svelte-kit plugin. Vite exports to `./out`.

You can preview the production build with `npm run preview`.

```bash
pnpm build
```

## Testing

Our tests use [Playwright](https://playwright.dev/).

```bash
pnpm test
```
