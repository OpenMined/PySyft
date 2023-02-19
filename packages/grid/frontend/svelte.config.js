import adapter from '@sveltejs/adapter-auto';
import nodeAdapter from '@sveltejs/adapter-node';
import preprocess from 'svelte-preprocess';
import tailwind from 'tailwindcss';
import autoprefixer from 'autoprefixer';
// import { vitePreprocess } from '@sveltejs/kit/vite';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const Docker = process.env.DOCKER;

/** @type {import('@sveltejs/kit').Config} */
const config = {
  kit: {
    adapter: Docker ? nodeAdapter({ out: 'out' }) : adapter()
  },
  preprocess: vitePreprocess()
};

export default config;
