# Syft UI

The Syft UI is the user interface that allows data owners to manage their
**deployed** Syft datasites and gateways.

## Installation

```bash
cd <syft-root>/packages/grid/frontend
pnpm install
```

You can use other package managers such as yarn or npm.

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
