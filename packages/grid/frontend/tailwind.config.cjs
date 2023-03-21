const openminedColors = require('./src/lib/colors/hsl-colors.cjs');

module.exports = {
  content: [
    './src/**/*.{html,js,svelte,ts}',
    './node_modules/flowbite-svelte/**/*.{html,js,svelte,ts}'
  ],
  theme: {
    extend: {
      fontFamily: {
        roboto: ['Roboto', 'Roboto-offline', 'system-ui', 'sans-serif'],
        rubik: ['Rubik', 'Rubik-offline', 'system-ui', 'sans-serif'],
        fira: ['Fira Code', 'Fira Code-offline', 'ui-monospace', 'monospace'],
        mono: ['Fira Code', 'Fira Code-offline', 'ui-monospace', 'monospace']
      },
      colors: {
        ...openminedColors,
        primary: openminedColors.cyan
      },
      spacing: {
        4.5: '1.125rem',
        13: '3.25rem'
      },
      lineHeight: {
        120: '1.2',
        160: '1.6'
      },
      boxShadow: {
        'neutral-1': '-2px 4px 8px rgba(13, 12, 17, 0.25)',
        'modal-1': '-1px 2px 4px 2px rgba(0, 0, 0, 0.08);'
      },
      minHeight: {
        screen: '100vh'
      },
      minWidth: {
        screen: '100vw'
      }
    }
  },
  plugins: [require('flowbite/plugin')],
  darkMode: 'class'
};
