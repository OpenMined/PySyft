const plugin = require('tailwindcss/plugin')
const twcolors = require('tailwindcss/colors')

module.exports = {
  purge: {
    content: ['./src/**/*.{ts,tsx}']
  },
  darkMode: false, // or 'media' or 'class'
  theme: {
    extend: {
      fontFamily: {
        rubik: 'Rubik, sans-serif'
      },
      lineHeight: {
        12: '3rem'
      },
      minWidth: {
        lg: '32rem'
      },
      maxWidth: {
        112: '28rem'
      },
      container: theme => ({
        padding: theme('padding.4')
      }),
      keyframes: {
        punch: {
          '0%, 100%': {transform: 'rotate(-38deg)'},
          '50%': {transform: 'rotate(71deg)'}
        }
      },
      animation: {
        punch: 'punch 250ms ease-in-out'
      },
      colors: {
        ...twcolors,
        cyan: {
          50: 'hsl(195, 75%, 95%)',
          100: 'hsl(195, 75%, 90%)',
          200: 'hsl(195, 75%, 80%)',
          300: 'hsl(195, 75%, 70%)',
          400: 'hsl(195, 75%, 60%)',
          500: 'hsl(195, 75%, 50%)',
          600: 'hsl(195, 75%, 40%)',
          700: 'hsl(195, 75%, 30%)',
          800: 'hsl(195, 75%, 20%)',
          900: 'hsl(195, 75%, 10%)'
        }
      }
    }
  },
  variants: {
    extend: {
      animation: ['responsive', 'focus', 'hover', 'active'],
      transitionProperty: ['hover'],
      backgroundColor: ['active', 'disabled', 'group-hover'],
      opacity: ['active', 'group-hover'],
      cursor: ['disabled'],
      textColor: ['active', 'disabled'],
      ringWidth: ['hover', 'active'],
      ringColor: ['hover', 'active'],
      display: ['hover', 'group-hover']
    },
    textColor: ({after}) => after(['invalid'])
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/line-clamp'),
    require('@tailwindcss/aspect-ratio'),

    plugin(function ({addVariant, e}) {
      addVariant('invalid', ({modifySelectors, separator}) => {
        modifySelectors(({className}) => {
          return `.${e(`invalid${separator}${className}`)}:invalid`
        })
      })
    })
  ]
}
