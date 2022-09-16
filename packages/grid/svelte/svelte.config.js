import adapter from '@sveltejs/adapter-auto';
import nodeAdapter from '@sveltejs/adapter-node';
import preprocess from 'svelte-preprocess';
import tailwind from 'tailwindcss'
import autoprefixer from 'autoprefixer'

const Docker = process.env.DOCKER;

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: preprocess({
   postcss: {
      plugins: [
         tailwind,
         autoprefixer
      ]
   }
  }),
  kit: {
    adapter: Docker ? nodeAdapter({ out: 'out' }) : adapter()
  }
};

export default config;
