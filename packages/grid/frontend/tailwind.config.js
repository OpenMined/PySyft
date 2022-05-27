const omuiColors = require('./src/omui/styles/colors')
const themes = require('./src/omui/themes')

module.exports = {
  purge: ['src/**/*.{js,jsx,ts,tsx}'],
  // mode: 'jit',
  darkMode: 'class', // or 'media' or 'class'
  theme: {
    extend: {
      fontFamily: {
        roboto: ['"Roboto"', 'sans-serif'],
        rubik: ['"Rubik"', 'sans-serif'],
        firacode: ['"Fira Code"', 'monospace'],
      },
      lineHeight: {
        12: '3rem',
      },
      minWidth: {
        lg: '32rem',
        '270px': '270px',
      },
      maxWidth: {
        112: '28rem',
        '270px': '270px',
        42: '10.5rem',
        modal: '646px',
        mbig: '906px',
      },
      marginLeft: {
        sidebar: '270px',
      },
      container: (theme) => ({
        padding: theme('padding.4'),
      }),
      keyframes: {
        punch: {
          '0%, 100%': { transform: 'rotate(-38deg)' },
          '50%': { transform: 'rotate(71deg)' },
        },
      },
      animation: {
        punch: 'punch 250ms ease-in-out',
      },
      fontSize: {
        xxs: ['.5rem', '1.6'],
        xs: ['.75rem', '1.6'],
        sm: ['.875rem', '1.5'],
        tiny: '.875rem',
        md: ['1rem', '1.5'],
        base: '1rem',
        lg: ['1.125rem', '1.5'],
        xl: ['1.25rem', '1.5'],
        '2xl': ['1.5rem', '1.5'],
        '3xl': ['1.75rem', '1.4'],
        '4xl': ['2.25rem', '1.1'],
        '4xl-upper': ['2.25rem', '1.4'],
        '4xl-mono': ['2.25rem', '1.3'],
        '5xl': ['3rem', '1.1'],
        '5xl-upper': ['3rem', '1.4'],
        '5xl-mono': ['3rem', '1.3'],
        '6xl': ['4rem', '1.1'],
        '6xl-upper': ['4rem', '1.3'],
        '7xl': '5rem',
      },
      colors: {
        ...omuiColors,
        ...themes.cyan,
        white: '#fff',
        black: '#000',
        transparent: 'transparent',
      },
      fill: {
        transparent: 'transparent',
      },
      boxShadow: (theme) => ({
        'icon-border': '0 0 1px transparent',
        'button-focus': '0px 0px 8px 1px text-primary-500',
        'primary-focus': `0px 0px 8px 1px ${theme('colors.primary.500')}`,
        card: '-2px 4px 8px 0px rgba(13, 12, 17, 0.25)',
        'card-hover': `-4px 4px 16px ${theme('colors.primary.500')}`,
        modal: '-2px 4px 8px rgba(13, 12, 17, 0.25)',
      }),
      gradientColorStops: {
        ...omuiColors,
        ...themes.cyan,
        'gradient-white': 'rgba(255, 255, 255, 0.5)',
        gbsc: 'rgba(0, 0, 0, 0.75)',
        gbuz: 'rgba(0, 0, 0, 0.10)',
      },
      spacing: {
        2.5: '0.625rem',
        3.5: '0.875rem',
        4.5: '1.125rem',
        13: '3.25rem',
      },
      dropShadow: {
        'button-hover': '-4px 4px 8px --color-primary-500',
      },
      backgroundImage: {
        radio: `url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='currentColor' xmlns='http://www.w3.org/2000/svg'%3e%3ccircle cx='8' cy='8' r='3'/%3e%3c/svg%3e")`,
        'scrim-white': `linear-gradient(90deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.5) 100%), #F1F0F4`,
      },
    },
  },
  variants: {
    extend: {
      animation: ['responsive', 'focus', 'hover', 'active'],
      transitionProperty: ['hover'],
      backgroundColor: ['active', 'disabled', 'group-hover'],
      border: ['first', 'last'],
      opacity: ['active', 'group-hover'],
      cursor: ['disabled'],
      textColor: ['active', 'disabled'],
      ringWidth: ['hover', 'active'],
      ringColor: ['hover', 'active'],
      display: ['hover', 'group-hover'],
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/line-clamp'),
    require('@tailwindcss/aspect-ratio'),
  ],
}
